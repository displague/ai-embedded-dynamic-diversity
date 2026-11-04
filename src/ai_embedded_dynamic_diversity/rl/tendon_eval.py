"""The gate: a policy trained on the stand-in, run on the full tendon model (displague/robot tools/sim_tendon.py).

Every 20 ms the policy reads the stand-in's observation computed from the tendon rig (the same layout as
displague/robot tools/record_teacher.py, plus the previous action) and its targets replace the hand-written
controller's: the rig's balance loop is off, so the policy is the balance. The tendon model then reports what
the stand-in cannot: line tensions against their ratings, slack, clutch slip, locks and electrical power.

    python -m ai_embedded_dynamic_diversity.rl.tendon_eval artifacts/rl/stand-grown-bc.pt
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import torch

from ai_embedded_dynamic_diversity.rl.actor import CoreActor, RunningNorm, build_policy_net
from ai_embedded_dynamic_diversity.sim.robot_tendon import _load_sim_tendon, find_robot_repo


def load_policy(path: str, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    core = build_policy_net(saved)
    actor = CoreActor(core)
    actor.log_std.data = saved["log_std"]
    norm = RunningNorm(core.signal_dim)
    norm.load_state_dict(saved["obs_norm"])
    return actor.to(device).eval(), norm.to(device), saved


class TendonPolicyRunner:
    def __init__(self, policy_path: str, repo: str | None = None, rate_hz: float = 50.0, locks: bool = True,
                 deadband: float = 0.0, rest_v: float = 0.0, rest_s: float = 0.5,
                 hand_ankles: bool | None = None):
        """`deadband` (rad): a joint's target holds until the policy asks for a change larger than this, so a
        policy that is nearly still gives steady targets and the drums can lock (issue #56)."""
        self.actor, self.norm, self.saved = load_policy(policy_path)
        self._ankle_gain = None
        self.locks = locks
        self.deadband = deadband
        # hand_ankles: the robot's hand-written balance loop drives the ankles (its torque mode, CoP loop and hip
        # lean) and the policy every other joint: a hybrid (displague/robot docs/32). None: as the policy was trained
        # (a `finetune --hand-ankles` policy saves it), else off
        self.hand_ankles = bool(self.saved["env"].get("hand_ankles")) if hand_ankles is None else hand_ankles
        self.rest_v, self.rest_s = rest_v, rest_s   # rest_v > 0: the deadband's own looser stillness gate
        self.rest_t = 0.0
        st = _load_sim_tendon(find_robot_repo(repo))
        self.st = st
        if not hasattr(st, "_OUTER_GAIN_FOOT"):
            st._OUTER_GAIN_FOOT = st.OUTER_GAIN.get("foot", 1.0)    # the model's own ankle gain, kept once
        self._ankle_gain = st._OUTER_GAIN_FOOT
        self.rig = st.Rig()
        self.names = [w["name"] for w in self.rig.w]
        assert self.names == self.saved["dof_names"], "the policy was trained on a different joint order"
        env = self.saved["env"]
        frac = env["action_fraction"] if env.get("action_fraction") is not None else (0.2 if env["task"] == "stand" else 1.0)
        lo = np.array([w["lo"] for w in self.rig.w])
        hi = np.array([w["hi"] for w in self.rig.w])
        self.lo, self.hi, self.half = lo, hi, 0.5 * (hi - lo) * frac
        if env.get("action_half") is not None:
            self.half = np.asarray(env["action_half"], dtype=float)[:len(lo)]
        self.every = max(1, int(round(1.0 / (rate_hz * self.rig.model.opt.timestep))))

    def reset(self):
        r = self.rig
        r.reset()
        r.balance = self.hand_ankles
        # the drums' stillness gate (the robot's D-038) runs without the balance loop, so the drums can lock under
        # a policy that holds still, as the firmware would let them (robot checkouts from its docs/30 on)
        r.still_gate = self.locks
        # a policy trained with the ankles in torque mode runs them so here too: no outer-loop trim at the ankles.
        # Not in the hybrid: there the hand-written loop drives the ankles, and its ankle roll in double stance runs
        # in position mode on the model's own gain
        torque = self.saved["env"].get("ankle_torque") and not self.hand_ankles
        self.st.OUTER_GAIN["foot"] = 0.0 if torque else self._ankle_gain
        if not self.locks:
            r.lock_enabled = False
        self.default = np.array([r.target[n] for n in self.names])
        self.is_ankle = np.array([n.startswith("foot.") for n in self.names])
        self.prev = np.zeros(len(self.names))
        self.mem = self.actor.initial_state(1, "cpu")
        self.count = 0
        self.rest_t = 0.0
        r.hook = self._hook

    def _obs(self):
        r, d = self.rig, self.rig.data
        q = np.array([d.qpos[w["q"]] for w in r.w])
        qd = np.array([d.qvel[w["dq"]] for w in r.w])
        R = d.xmat[r.pelvis].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.qvel[3:6]
        _, loads = r.cop()
        return np.concatenate([q - self.default, 0.1 * qd, grav, 0.25 * gyro, np.array(loads) / (r.mass * 9.81),
                               self.prev]).astype(np.float32)

    @torch.no_grad()
    def _hook(self, r):
        self.count += 1
        if (self.count - 1) % self.every:
            return
        o = torch.as_tensor(self._obs())[None]
        mean, self.mem = self.actor(self.norm(o), self.mem)
        a = mean[0].clamp(-1, 1).numpy()
        tgt = np.clip(self.default + a * self.half, self.lo, self.hi)
        # only while the body is still: the drums' own stillness gate, or (rest_v > 0) a looser one on the CoM
        # speed, which the policy's own fidget does not hold shut; a push (above the drums' release speed) opens it
        if self.rest_v > 0.0:
            v = float(np.linalg.norm(r._com_v))
            self.rest_t = 0.0 if v > self.rest_v else self.rest_t + self.every * r.model.opt.timestep
            resting = self.rest_t > self.rest_s
        else:
            resting = getattr(r, "_still", 0.0) > self.st.LOCK_STILL
        if self.deadband > 0.0 and resting:
            held = np.array([r.target[n] for n in self.names])
            tgt = np.where(np.abs(tgt - held) > self.deadband, tgt, held)
        if self.hand_ankles:                # the ankles keep the balance loop's own targets
            tgt = np.where(self.is_ankle, self.default, tgt)
        for n, v in zip(self.names, tgt):
            r.target[n] = float(v)
        self.prev = a


WIND_SCENES = (5, 8)                    # m/s mean head wind, 1.5 m/s gusts, 10 s standing


class _TeacherRunner:
    """The hand-written controller on the same scenes: the rig as it comes, balance loop on."""

    def __init__(self, repo: str | None = None):
        self.st = _load_sim_tendon(find_robot_repo(repo))
        if not hasattr(self.st, "_OUTER_GAIN_FOOT"):
            self.st._OUTER_GAIN_FOOT = self.st.OUTER_GAIN.get("foot", 1.0)
        self.rig = self.st.Rig()

    def reset(self):
        # the model's own ankle gain, should a torque-mode policy have run in this process before
        self.st.OUTER_GAIN["foot"] = self.st._OUTER_GAIN_FOOT
        self.rig.reset()


def power(rig) -> tuple[float, float]:
    """(mean electrical power in W over the scene so far, fraction of drum-steps locked), as the tendon model
    counts them (its `report`)."""
    st = rig.stats.values()
    t = max(rig.data.time, 1e-9)
    n = sum(s["n"] for s in st)
    return sum(s["e_j"] for s in st) / t, (sum(s["locked"] for s in st) / n if n else 0.0)


REST_BAG_KG = 3.9                       # a gallon of milk in the left hand, arm hanging (the robot's docs/28)
POWER_PASS_W = 7.2                      # the rest scenes' pass line: 2 x the hand-written controller standing


HYBRID = dict(hand_ankles=True)     # the hand-written ankle balance under the policy


REST_MODE = dict(deadband=math.radians(2.0), rest_v=0.05)   # the rest mode: targets held within 2 deg once the
                                                            # CoM has been under 0.05 m/s for 0.5 s (issue #56)


def _runner(policy_path: str, locks: bool, bag_kg: float = 0.0, repo: str | None = None, rest_mode: bool = False):
    """A runner whose rig carries `bag_kg` in the left hand (the robot's export_mjcf.BAGS, read at build)."""
    import sys as _sys
    _load_sim_tendon(find_robot_repo(repo))
    ex = _sys.modules["export_mjcf"]
    before = dict(ex.BAGS)
    if bag_kg > 0:
        ex.BAGS.update({"hand.L": bag_kg})
    try:
        return (_TeacherRunner(repo) if policy_path == "teacher"
                else TendonPolicyRunner(policy_path, repo=repo, locks=locks, **(REST_MODE if rest_mode else {})))
    finally:
        ex.BAGS.clear()
        ex.BAGS.update(before)


def evaluate_rest(policy_path: str, locks: bool = True, repo: str | None = None, rest_mode: bool = False) -> dict:
    """The rest scenes (issue #56): standing idle 30 s, and holding a 3.9 kg bag 20 s. Each: standing at the end,
    mean electrical power and the fraction of drum-steps locked over the last 20 s (after settling)."""
    out = {}
    for name, kg, secs in (("idle30", 0.0, 30.0), ("bag20", REST_BAG_KG, 20.0)):
        runner = _runner(policy_path, locks, kg, repo, rest_mode)
        runner.reset()
        r = runner.rig
        r.run(secs - 20.0 if secs > 20.0 else 0.0)
        e0 = sum(s["e_j"] for s in r.stats.values())
        n0 = sum(s["n"] for s in r.stats.values())
        l0 = sum(s["locked"] for s in r.stats.values())
        t0 = r.data.time
        r.run(min(20.0, secs))
        st_ = r.stats.values()
        w = (sum(s["e_j"] for s in st_) - e0) / max(r.data.time - t0, 1e-9)
        n = sum(s["n"] for s in st_) - n0
        lk = (sum(s["locked"] for s in st_) - l0) / n if n else 0.0
        out[name] = dict(stands=bool(r.standing()), power_w=float(w), locked=float(lk),
                         passes=bool(r.standing() and w <= POWER_PASS_W * (1.0 if kg == 0 else 3.0)))
    return out


def evaluate(policy_path: str, pushes: int = 20, seed: int = 0, locks: bool = True, rest_mode: bool = False) -> dict:
    """The gate scenes for a policy, or for the hand-written controller when `policy_path` is 'teacher'.
    `locks`: let the drums lock under the policy (the stillness gate on); False reproduces the gate before
    the robot's docs/30, where no lock could engage under a policy."""
    teacher = policy_path == "teacher"
    runner = _TeacherRunner() if teacher else TendonPolicyRunner(policy_path, locks=locks,
                                                                 **(REST_MODE if rest_mode else {}))
    who = "the hand-written controller" if teacher else "policy %s" % policy_path
    rng = np.random.default_rng(seed)
    results = {}

    def scene(name, fn):
        runner.reset()
        fn(runner.rig)
        z, tilt = runner.rig.pelvis_state()
        rows = runner.rig.report("%s (%s)" % (name, who), top=5)
        w, locked = power(runner.rig)
        results[name] = dict(stands=bool(runner.rig.standing()), pelvis_z=float(z), tilt_deg=float(tilt),
                             power_w=float(w), locked=float(locked),
                             worst_line=rows[0][1], worst_line_n=float(rows[0][2]), worst_line_cont=float(rows[0][3]),
                             worst_hot_s=float(max(r[11] for r in rows)))

    scene("stand10", lambda r: r.run(10.0))
    for f in (10, 20, 30):
        scene("push%d" % f, lambda r, f=f: (r.run(1.5), r.push("spine", (0, -f, 0), 0.2), r.run(3.0)))
    if hasattr(runner.st, "WIND"):         # a robot checkout with wind (its docs/25): a head wind with gusts
        before = runner.st.WIND
        for mean in WIND_SCENES:
            runner.st.WIND = dict(mean=mean, gust=1.5, dir="head", seed=seed)
            try:
                scene("wind%d" % mean, lambda r: r.run(10.0))
            finally:
                runner.st.WIND = before
    ok = 0
    for k in range(pushes):
        ang, f = rng.uniform(-math.pi, math.pi), rng.uniform(10, 30)

        def fn(r, ang=ang, f=f):
            r.run(1.5)
            r.push("spine", (f * math.cos(ang), f * math.sin(ang), 0.0), 0.2)
            r.run(4.5)                    # 4.5 s after the push: a slow fall shows by then (2.5 s missed some)
        runner.reset()
        fn(runner.rig)
        ok += bool(runner.rig.standing())
    results["random_pushes"] = dict(n=pushes, stands=ok)
    return results


def evaluate_teacher(pushes: int = 20, seed: int = 0) -> dict:
    """The same random pushes (same seed) under the hand-written controller, for comparison."""
    st = _load_sim_tendon(find_robot_repo(None))
    rig = st.Rig()
    rng = np.random.default_rng(seed)
    ok = 0
    for k in range(pushes):
        ang, f = rng.uniform(-math.pi, math.pi), rng.uniform(10, 30)
        rig.reset()
        rig.run(1.5)
        rig.push("spine", (f * math.cos(ang), f * math.sin(ang), 0.0), 0.2)
        rig.run(2.5)
        ok += bool(rig.standing())
    return dict(random_pushes=dict(n=pushes, stands=ok))


if __name__ == "__main__":
    rm = "restmode" in sys.argv[3:]           # the policy with the rest mode (REST_MODE) in front of the drives
    if "handankles" in sys.argv[3:]:          # the hybrid: the hand-written ankle balance under the policy
        _orig_init = TendonPolicyRunner.__init__

        def _init(self, *a, **k):
            k.setdefault("hand_ankles", True)
            _orig_init(self, *a, **k)
        TendonPolicyRunner.__init__ = _init
    if len(sys.argv) > 2 and sys.argv[2] == "rest":
        print(json.dumps(evaluate_rest(sys.argv[1], rest_mode=rm), indent=1))
        sys.exit(0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = evaluate(sys.argv[1], pushes=n, rest_mode=rm)   # 'teacher': the hand-written controller on every scene
    print(json.dumps(out, indent=1))

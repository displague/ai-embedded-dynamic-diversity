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

from ai_embedded_dynamic_diversity.models.core import ModelCore
from ai_embedded_dynamic_diversity.rl.actor import CoreActor, RunningNorm, load_checkpoint
from ai_embedded_dynamic_diversity.sim.robot_tendon import _load_sim_tendon, find_robot_repo


def load_policy(path: str, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, state = load_checkpoint(path)
    core = ModelCore(**cfg)
    core.load_state_dict(state)
    actor = CoreActor(core)
    actor.log_std.data = saved["log_std"]
    norm = RunningNorm(core.signal_dim)
    norm.load_state_dict(saved["obs_norm"])
    return actor.to(device).eval(), norm.to(device), saved


class TendonPolicyRunner:
    def __init__(self, policy_path: str, repo: str | None = None, rate_hz: float = 50.0):
        self.actor, self.norm, self.saved = load_policy(policy_path)
        st = _load_sim_tendon(find_robot_repo(repo))
        self.st = st
        self.rig = st.Rig()
        self.names = [w["name"] for w in self.rig.w]
        assert self.names == self.saved["dof_names"], "the policy was trained on a different joint order"
        env = self.saved["env"]
        frac = env["action_fraction"] if env.get("action_fraction") is not None else (0.2 if env["task"] == "stand" else 1.0)
        lo = np.array([w["lo"] for w in self.rig.w])
        hi = np.array([w["hi"] for w in self.rig.w])
        self.lo, self.hi, self.half = lo, hi, 0.5 * (hi - lo) * frac
        self.every = max(1, int(round(1.0 / (rate_hz * self.rig.model.opt.timestep))))

    def reset(self):
        r = self.rig
        r.reset()
        r.balance = False
        self.default = np.array([r.target[n] for n in self.names])
        self.prev = np.zeros(len(self.names))
        self.mem = self.actor.initial_state(1, "cpu")
        self.count = 0
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
        for n, v in zip(self.names, tgt):
            r.target[n] = float(v)
        self.prev = a


def evaluate(policy_path: str, pushes: int = 20, seed: int = 0) -> dict:
    runner = TendonPolicyRunner(policy_path)
    rng = np.random.default_rng(seed)
    results = {}

    def scene(name, fn):
        runner.reset()
        fn(runner.rig)
        z, tilt = runner.rig.pelvis_state()
        rows = runner.rig.report("%s (policy %s)" % (name, policy_path), top=5)
        results[name] = dict(stands=bool(runner.rig.standing()), pelvis_z=float(z), tilt_deg=float(tilt),
                             worst_line=rows[0][1], worst_line_n=float(rows[0][2]), worst_line_cont=float(rows[0][3]),
                             worst_hot_s=float(max(r[11] for r in rows)))

    scene("stand10", lambda r: r.run(10.0))
    for f in (10, 20, 30):
        scene("push%d" % f, lambda r, f=f: (r.run(1.5), r.push("spine", (0, -f, 0), 0.2), r.run(3.0)))
    ok = 0
    for k in range(pushes):
        ang, f = rng.uniform(-math.pi, math.pi), rng.uniform(10, 30)

        def fn(r, ang=ang, f=f):
            r.run(1.5)
            r.push("spine", (f * math.cos(ang), f * math.sin(ang), 0.0), 0.2)
            r.run(2.5)
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
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    out = evaluate_teacher(n) if sys.argv[1] == "teacher" else evaluate(sys.argv[1], pushes=n)
    print(json.dumps(out, indent=1))

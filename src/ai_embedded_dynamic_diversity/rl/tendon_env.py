"""PPO directly on the full tendon model (displague/robot tools/sim_tendon.py), for fine-tuning a stand policy where
the fast stand-in falls short: the drums' self-locks and the electrical power (issue #56).

`TendonVecEnv` has the interface rl/ppo.py uses from `StandinEnv` (reset, observe, step, obs_dim, priv_dim, act_dim,
N, dt_ctrl, half, dof_names). Each worker process holds `per_worker` rigs and steps them one control tick at a time
(the tendon model runs at about 2x real time per process, so 20 workers give ~2,000 control steps per second).

The observation is the gate's (rl/tendon_eval.py), the same as the stand-in's for the tendon humanoid, so a policy
trained on the stand-in fine-tunes here unchanged. The balance loop is off and the drums' stillness gate on (the
robot's `still_gate`), as in the gate. Scenes: pushes of 10 to 30 N for 0.2 s every `push_interval_s` (quiet rigs:
every `quiet_push_s`), and wind on some workers. The reward is the stand-in's (height, upright, standing, alive,
action rate) minus `power_cost * log(1 + P / 10 W)`, P the tendon model's own electrical power; a fall ends it.
"""
from __future__ import annotations

import contextlib
import io
import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np

# no package imports at module level: the workers import this module, and the sibling's sim package loads torch


@dataclass
class TendonEnvConfig:
    workers: int = 20
    per_worker: int = 8
    control_hz: float = 50.0
    episode_s: float = 20.0
    action_fraction: float = 0.2
    action_half: tuple | None = None       # per-DOF half-spans (the fine-tuned policy's own), overrides the fraction
    ankle_torque: bool = False             # the ankles in torque mode (no outer-loop trim), as the policy was trained
    push_interval_s: tuple = (2.0, 5.0)
    push_n: tuple = (10.0, 30.0)
    quiet_fraction: float = 0.5
    quiet_push_s: tuple = (8.0, 16.0)
    wind_workers: float = 0.3              # this fraction of workers blows a wind (mean 0 to wind_max, gusts to 1.5)
    wind_max: float = 8.0
    hand_ankles: bool = False              # the robot's hand-written balance loop drives the ankles, the policy the rest
    power_cost: float = 0.5
    power_ref_w: float = 10.0
    seed: int = 0
    repo: str | None = None


def _worker(conn, cfg: TendonEnvConfig, wid: int):
    import os
    import sys

    tools = os.path.join(cfg.repo, "tools")       # resolved by the parent
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import sim_tendon as st
    if cfg.ankle_torque:
        st.OUTER_GAIN["foot"] = 0.0
    rng = np.random.default_rng([cfg.seed, wid])
    K = cfg.per_worker
    with contextlib.redirect_stdout(io.StringIO()):
        rigs = [st.Rig() for _ in range(K)]
    names = [w["name"] for w in rigs[0].w]
    lo = np.array([w["lo"] for w in rigs[0].w])
    hi = np.array([w["hi"] for w in rigs[0].w])
    half = 0.5 * (hi - lo) * cfg.action_fraction
    is_ankle = np.array([n.startswith("foot.") for n in names])
    if cfg.action_half is not None:
        half = np.asarray(cfg.action_half, dtype=float)[:len(lo)]
    every = max(1, int(round(1.0 / (cfg.control_hz * rigs[0].model.opt.timestep))))
    dt_ctrl = every * rigs[0].model.opt.timestep
    windy = wid < int(round(cfg.wind_workers * cfg.workers))
    quiet = np.arange(K) < int(round(cfg.quiet_fraction * K))
    spine = [rigs[0].model.body("spine").id] * K
    S = dict(t=np.zeros(K), push_at=np.zeros(K), push_left=np.zeros(K, dtype=int), force=np.zeros((K, 3)),
             prev=np.zeros((K, len(names))), e=np.zeros(K), stood=np.zeros(K), ever=np.zeros(K, dtype=bool), default=None, z0=None)

    def gap(k):
        return rng.uniform(*(cfg.quiet_push_s if quiet[k] else cfg.push_interval_s))

    def reset(k):
        r = rigs[k]
        r.reset()
        r.balance = cfg.hand_ankles
        r.still_gate = True
        st.mujoco.mj_forward(r.model, r.data)   # positions for the first observation (reset leaves them stale)
        if windy:
            r.set_wind(dict(mean=float(rng.uniform(0, cfg.wind_max)), gust=float(rng.uniform(0, 1.5)),
                            dir=float(rng.uniform(0, 360)), seed=int(rng.integers(1 << 30))))
        S["t"][k] = 0.0
        S["push_at"][k] = gap(k)
        S["push_left"][k] = 0
        S["prev"][k] = 0.0
        S["e"][k] = 0.0
        S["stood"][k] = 0.0
        S["ever"][k] = False

    def obs(k):
        r = rigs[k]
        d = r.data
        q = np.array([d.qpos[w["q"]] for w in r.w])
        qd = np.array([d.qvel[w["dq"]] for w in r.w])
        R = d.xmat[r.pelvis].reshape(3, 3)
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        gyro = d.qvel[3:6]
        _, loads = r.cop()
        o = np.concatenate([q - S["default"], 0.1 * qd, grav, 0.25 * gyro, np.array(loads) / (r.mass * 9.81),
                            S["prev"][k]])
        z, tilt = r.pelvis_state()
        locked = np.mean([w.get("_locked", False) for w in r.w])
        p = np.concatenate([o, [z / S["z0"], math.cos(math.radians(tilt)), min(r._still, 5.0) / 5.0, locked]])
        return o, p

    for k in range(K):
        reset(k)
    S["default"] = np.array([rigs[0].target[n] for n in names])
    S["z0"] = rigs[0].pelvis_state()[0]
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        if cmd == "info":
            conn.send(dict(names=names, half=half, dt_ctrl=dt_ctrl, obs_dim=obs(0)[0].size, priv_dim=obs(0)[1].size))
            continue
        if cmd == "observe":
            o, p = zip(*[obs(k) for k in range(K)])
            conn.send((np.array(o), np.array(p)))
            continue
        # step: arg = (K, nu) actions
        out = []
        for k in range(K):
            r = rigs[k]
            a = np.clip(arg[k], -1.0, 1.0)
            tgt = np.clip(S["default"] + a * half, lo, hi)
            if cfg.hand_ankles:
                tgt = np.where(is_ankle, S["default"], tgt)
            for n, v in zip(names, tgt):
                r.target[n] = float(v)
            if S["push_left"][k] == 0 and S["t"][k] >= S["push_at"][k]:
                ang = rng.uniform(-math.pi, math.pi)
                f = rng.uniform(*cfg.push_n)
                S["force"][k] = (f * math.cos(ang), f * math.sin(ang), 0.0)
                S["push_left"][k] = int(round(0.2 / dt_ctrl))
                S["push_at"][k] = S["t"][k] + gap(k)
            if S["push_left"][k] > 0:
                r.data.xfrc_applied[spine[k]][:3] = S["force"][k]
                S["push_left"][k] -= 1
            else:
                r.data.xfrc_applied[spine[k]][:3] = 0.0
            e0 = sum(s["e_j"] for s in r.stats.values())
            for _ in range(every):
                r.step()
            e1 = sum(s["e_j"] for s in r.stats.values())
            power = (e1 - e0) / dt_ctrl
            S["t"][k] += dt_ctrl
            z, tilt = r.pelvis_state()
            up = math.cos(math.radians(tilt))
            standing = bool(r.standing())
            S["stood"][k] = S["stood"][k] + dt_ctrl if standing else 0.0
            S["ever"][k] = S["ever"][k] or S["stood"][k] > 1.0
            rew = 0.5 * min(z / S["z0"], 1.0) + 0.5 * max(up, 0.0) + 1.0 * standing + 0.5
            rew -= 0.05 * float(np.square(a - S["prev"][k]).mean()) * len(names)
            rew -= cfg.power_cost * math.log1p(power / cfg.power_ref_w)
            fell = z < 0.65 * S["z0"]
            timeout = S["t"][k] >= cfg.episode_s - 1e-9
            if fell:
                rew = -10.0
            locked = float(np.mean([w.get("_locked", False) for w in r.w]))
            S["prev"][k] = a
            done = fell or timeout
            out.append((rew, done, fell, timeout and not fell, standing, power, locked, bool(S["ever"][k])))
            if done:
                reset(k)
        o, p = zip(*[obs(k) for k in range(K)])
        conn.send((np.array(o), np.array(p), out))


class TendonVecEnv:
    def __init__(self, cfg: TendonEnvConfig):
        from dataclasses import replace

        from ai_embedded_dynamic_diversity.sim.robot_tendon import find_robot_repo

        cfg = replace(cfg, repo=find_robot_repo(cfg.repo))
        self.cfg = cfg
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        # spawned workers would re-import the parent's __main__ (the trainer, so torch: ~0.5 GB each); hide it
        # while they start, so a worker loads only this module, numpy and mujoco
        import sys
        main = sys.modules["__main__"]
        saved = {k: getattr(main, k) for k in ("__spec__", "__file__") if hasattr(main, k)}
        try:
            main.__spec__ = None
            if hasattr(main, "__file__"):
                del main.__file__
            for w in range(cfg.workers):
                a, b = ctx.Pipe()
                p = ctx.Process(target=_worker, args=(b, cfg, w), daemon=True)
                p.start()
                self.conns.append(a)
                self.procs.append(p)
        finally:
            for k, v in saved.items():
                setattr(main, k, v)
        self.conns[0].send(("info", None))
        info = self.conns[0].recv()
        self.dof_names = info["names"]
        self.half = info["half"]
        self.dt_ctrl = info["dt_ctrl"]
        self._obs_dim, self._priv_dim = info["obs_dim"], info["priv_dim"]
        self.N = cfg.workers * cfg.per_worker
        self.nu = len(self.dof_names)

    @property
    def obs_dim(self):
        return self._obs_dim

    @property
    def priv_dim(self):
        return self._priv_dim

    @property
    def act_dim(self):
        return self.nu

    def reset(self, idx=None):
        pass                                # workers start reset and reset each rig when its episode ends

    def observe(self):
        for c in self.conns:
            c.send(("observe", None))
        o, p = zip(*[c.recv() for c in self.conns])
        return np.concatenate(o).astype(np.float32), np.concatenate(p).astype(np.float32)

    def step(self, action: np.ndarray, mean: np.ndarray | None = None):
        from ai_embedded_dynamic_diversity.rl.standin_env import StepInfo

        K = self.cfg.per_worker
        for w, c in enumerate(self.conns):
            c.send(("step", action[w * K:(w + 1) * K, :self.nu]))
        res = [c.recv() for c in self.conns]
        obs = np.concatenate([r[0] for r in res]).astype(np.float32)
        priv = np.concatenate([r[1] for r in res]).astype(np.float32)
        rows = [x for r in res for x in r[2]]
        rew, done, fell, timeout, standing, power, locked, stood = (np.array(v) for v in zip(*rows))
        info = StepInfo(standing=standing.astype(bool), damage=fell.astype(bool), timeout=timeout.astype(bool),
                        line_over_max=np.zeros(self.N), impact=np.zeros(self.N),
                        extras=dict(ever_stood=stood.astype(bool), power_w=power, locked=locked))
        return obs, priv, rew.astype(np.float32), done.astype(bool), info

    def close(self):
        for c in self.conns:
            with contextlib.suppress(Exception):
                c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)

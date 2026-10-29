"""Batched environments on the tendon humanoid's fast stand-in (displague/robot sim/robot_standin.xml).

The stand-in is the robot's skeleton with each winch-and-line drive train replaced by one position servo that
keeps the loop's stiffness, the motor's back-EMF damping and speed, and the clutch's slip torque (the robot
repository's tools/export_standin.py). Physics runs in MuJoCo's multithreaded rollout on the CPU; the policy
runs in torch on any device. Everything here is numpy over the batch.

Damage gates, from the per-DOF figures in sim/robot_standin.json, the same ones the tendon model reports:

- a line above its continuous rating costs in proportion to the excess and, for longer than `line_over_s`
  (2 s in training), ends the episode; the time above `line_report_s` (0.5 s, the tendon model's exit test)
  is reported; the
  force on the loaded run for a joint torque tau is pretension + |tau| / (2 r) while both runs are taut
  (|tau| below the linear torque), else |tau| / r, with r the moment arm (tackle x pulley radius);
- an impact on the head, trunk or pelvis shell above `impact_end_n` ends it; above `impact_cost_n` it costs;
- a joint pressed into its stop costs in proportion to the stop force.

The W-mini joints (neck, wrist) are excluded from the line gate: they are above their rating standing still
(displague/robot D-023), and gating on them would end every episode. Their overload is reported instead.

Tasks:
- `stand`: start standing, random pushes; survive and stay upright.
- `getup`: start fallen (dropped from a random orientation and pose, settled), reach standing and hold it;
  falling does not end the episode, damage does.

Wind (off by default; `wind_mean` / `wind_gust`), as on the robot (displague/robot tools/sim_tendon.py WIND,
Rig.blow): MuJoCo's fluid model (`opt.density`, the inertia-box drag of each body) under `opt.wind`, a mean
speed in a random direction plus Ornstein-Uhlenbeck gusts on the along-wind (sigma `gust`) and cross-wind
(0.75 `gust`) speed with time constant `wind_tau`. The models are shared by the envs mapped to them, so the
wind is per model variant: every non-nominal variant draws its own mean, direction and gust sigma once and
carries its own gust state, advanced each control tick; variant 0 stays calm. The policy observation has no
wind channel (the robot has no anemometer); with wind on, the privileged observation gains two channels, the
wind's world x and y speed / 10 m/s.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from ai_embedded_dynamic_diversity.sim.robot_tendon import find_robot_repo

STAND_LEAN = math.radians(-5.0)    # the robot's standing ankle angle (displague/robot sim_tendon.STAND_LEAN)
STAND_Z = 0.922                    # pelvis height standing on the stand-in (measured)


BODIES = ("tendon32", "rpo23")


def body_paths(body: str, repo: str | None = None) -> tuple[str, str]:
    """(model xml, spec json) of a stand-in body: `tendon32`, the displague/robot tendon humanoid (its
    tools/export_standin.py), or `rpo23`, Roboparty's RPO (rl/rpo_body.py)."""
    if body == "tendon32":
        r = find_robot_repo(repo)
        return os.path.join(r, "sim", "robot_standin.xml"), os.path.join(r, "sim", "robot_standin.json")
    if body == "rpo23":
        from ai_embedded_dynamic_diversity.rl.rpo_body import RPO_DIR
        x = os.path.join(RPO_DIR, "mjcf", "rpo_standin.xml")
        if not os.path.exists(x):
            raise FileNotFoundError("%s missing: run python -m ai_embedded_dynamic_diversity.rl.rpo_body" % x)
        return x, x.replace(".xml", ".json")
    raise ValueError("unknown body %r (one of %s)" % (body, ", ".join(BODIES)))


@dataclass
class EnvConfig:
    task: str = "getup"
    body: str = "tendon32"
    pad_dofs: int = 0                      # pad every per-joint channel block to this many slots (one network for
                                           # bodies with different joint counts: 32 covers both); 0: no padding
    num_envs: int = 2048
    control_hz: float = 50.0
    episode_s: float = 10.0
    nthread: int = 0                       # 0: all cores
    variants: int = 64                     # randomised model copies (envs share them round-robin)
    seed: int = 0
    action_fraction: float | None = None  # actions span this fraction of each joint's half-range; None: 0.2
                                           # for stand (small corrections; the full span turned policy noise
                                           # into falls), 1.0 for getup (it needs the whole range)
    max_delay_steps: int = 2               # actuation delay, uniform in [0, this] control ticks per env
    kp_scale: tuple = (0.5, 1.05)          # servo stiffness: down to half (one line run slack), per DOF
    kv_scale: tuple = (0.8, 1.2)
    torque_scale: tuple = (0.85, 1.0)      # clutch slip torque
    friction: tuple = (0.4, 1.2)
    mass_scale: tuple = (0.9, 1.1)
    pelvis_mass_add: tuple = (-1.0, 1.0)   # kg
    push_interval_s: tuple = (2.0, 4.0)    # stand: a push every this often
    push_speed: float = 0.5                # m/s added to the pelvis
    line_over_s: float = 2.0            # training: this long above rating ends an episode (the tendon
                                           # exit test's 0.5 s is reported as `over_half_s`; ending on it from
                                           # the first step left an untrained policy no time to learn)
    line_report_s: float = 0.5
    impact_cost_n: float = 150.0
    impact_end_n: float = 1500.0
    stand_height: float = 0.92             # pelvis above this fraction of its standing height, upright, counts as
                                           # standing (0.85 m on the tendon humanoid)
    fall_height: float = 0.65              # stand task: below this fraction of standing height is a fall
    stand_upright: float = 0.95
    settle_s: float = 1.0                  # getup: the drop settles this long before the policy acts
    gated_classes: tuple = ("W-max", "W-mid")
    wind_mean: tuple = (0.0, 0.0)          # m/s, each non-nominal variant's mean wind speed, uniform in this range
                                           # (direction uniform); (0, 0) with no gusts: no wind, no air
    wind_gust: tuple = (0.0, 0.0)          # m/s, each variant's gust sigma (along-wind; 0.75 of it across)
    wind_tau: float = 2.0                  # s, the gusts' time constant
    wind_ramp_s: float = 2.0               # s, the mean ramps in over this long from the env's construction
    air_density: float = 1.2               # kg/m^3, MuJoCo opt.density on the windy variants
    repo: str | None = None


@dataclass
class StepInfo:
    standing: np.ndarray
    damage: np.ndarray
    timeout: np.ndarray
    line_over_max: np.ndarray
    impact: np.ndarray
    extras: dict = field(default_factory=dict)


class StandinEnv:
    """N copies of the stand-in, stepped together. reset() -> (obs, priv); step(actions) -> (obs, priv, reward,
    done, info). Actions are in [-1, 1] per DOF, mapped around the standing pose."""

    def __init__(self, cfg: EnvConfig):
        import mujoco
        from mujoco import rollout

        self.mj, self._rollout_mod = mujoco, rollout
        self.cfg = cfg
        self.xml_path, spec_path = body_paths(cfg.body, cfg.repo)
        with open(spec_path) as f:
            self.spec = json.load(f)
        self.base = mujoco.MjModel.from_xml_path(self.xml_path)
        m = self.base
        self.stand_z = float(self.spec.get("stand_height", STAND_Z))
        self.root = self.spec.get("root_body", "pelvis")
        self.rng = np.random.default_rng(cfg.seed)
        self.N, self.nu = cfg.num_envs, m.nu
        self.dt_ctrl = 1.0 / cfg.control_hz
        self.substeps = max(1, int(round(self.dt_ctrl / m.opt.timestep)))
        self.max_steps = int(round(cfg.episode_s / self.dt_ctrl))
        dofs = self.spec["dofs"]
        assert [d["name"] for d in dofs] == [m.actuator(i).name for i in range(m.nu)]
        self.dof_names = [d["name"] for d in dofs]
        self.lo = np.array([d["lo"] for d in dofs])
        self.hi = np.array([d["hi"] for d in dofs])
        if "default_ctrl" in self.spec:
            self.default = np.array(self.spec["default_ctrl"], dtype=float)
        else:                              # the tendon humanoid: everything at zero, ankles at the standing lean
            self.default = np.zeros(m.nu)
            for side in "LR":
                self.default[self.dof_names.index("foot.%s_pitch" % side)] = STAND_LEAN
        self.P = max(cfg.pad_dofs, m.nu)   # slots per per-joint channel block
        self.n_impact_slots = 3            # impact channels in the privileged observation (padded)
        frac = cfg.action_fraction if cfg.action_fraction is not None else (0.2 if cfg.task == "stand" else 1.0)
        self.half = 0.5 * (self.hi - self.lo) * frac
        self.arm = np.array([d["moment_arm_m"] for d in dofs])
        self.pre = np.array([d["pretension_n"] for d in dofs])
        self.lin = np.array([d["linear_torque_nm"] for d in dofs])
        self.f_cont = np.array([d["line_cont_n"] for d in dofs])
        self.gated = np.array([d["cls"] in cfg.gated_classes for d in dofs])
        self.tau_max = np.array([d["torque_max"] for d in dofs])
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.dof_names])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.dof_names])
        sadr = lambda n: (m.sensor_adr[m.sensor(n).id], m.sensor_dim[m.sensor(n).id])
        self.s_quat, self.s_gyro, self.s_acc = sadr("pelvis_quat"), sadr("pelvis_gyro"), sadr("pelvis_acc")
        self.s_feet = [sadr("%s_touch" % f)[0] for f in self.spec["feet"]]
        self.s_impact = [sadr("%s_impact" % b)[0] for b in self.spec["impact_bodies"]]
        self.s_tau = np.array([sadr("%s_tau" % n)[0] for n in self.dof_names])
        self.s_stop = np.array([sadr("%s_stop" % n)[0] for n in self.dof_names])
        self.weight = float(m.body_subtreemass[0] * 9.81)
        self.nq, self.nv = m.nq, m.nv
        self.state_kind = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(m, self.state_kind)
        # the full-physics state is [time, qpos, qvel, act, ...]
        self.i_qpos, self.i_qvel = 1, 1 + m.nq
        self.models = self._variants()
        self._wind_setup()
        self.model_of_env = [self.models[i % len(self.models)] for i in range(self.N)]
        nthread = cfg.nthread or os.cpu_count() or 1
        self.nthread = nthread
        self.datas = [mujoco.MjData(m) for _ in range(nthread)]
        self.roller = rollout.Rollout(nthread=nthread)
        self.state = np.zeros((self.N, self.nstate))
        self.sens = np.zeros((self.N, m.nsensordata))
        self.target = np.tile(self.default, (self.N, 1))
        self.prev_action = np.zeros((self.N, self.nu))
        self.delay = self.rng.integers(0, cfg.max_delay_steps + 1, size=self.N)
        self.queue = np.zeros((cfg.max_delay_steps + 1, self.N, self.nu))
        self.t = np.zeros(self.N, dtype=np.int64)
        self.over_t = np.zeros((self.N, self.nu))
        self.stand_t = np.zeros(self.N)
        self.push_at = np.zeros(self.N)
        self.ever_stood = np.zeros(self.N, dtype=bool)

    # ---------------------------------------------------------------- randomisation
    def _variants(self):
        cfg, mj = self.cfg, self.mj
        out = []
        pelvis = self.base.body(self.root).id
        for k in range(cfg.variants):
            m = mj.MjModel.from_xml_path(self.xml_path)
            if k == 0:                     # variant 0 is the nominal model
                out.append(m)
                continue
            u = lambda r, n=None: self.rng.uniform(r[0], r[1], size=n)
            kp = m.actuator_gainprm[:, 0] * u(cfg.kp_scale, m.nu)
            m.actuator_gainprm[:, 0] = kp
            m.actuator_biasprm[:, 1] = -kp
            m.actuator_biasprm[:, 2] *= u(cfg.kv_scale, m.nu)
            m.actuator_forcerange[:] *= u(cfg.torque_scale, m.nu)[:, None]
            m.geom_friction[:, 0] = u(cfg.friction)
            m.body_mass[1:] *= u(cfg.mass_scale, m.nbody - 1)
            m.body_mass[pelvis] = max(0.5, m.body_mass[pelvis] + u(cfg.pelvis_mass_add))
            out.append(m)
        return out

    def _wind_setup(self):
        """Per-variant wind (cfg.wind_*). Its own random stream, so the rest of the randomisation is the same
        with wind on or off."""
        cfg = self.cfg
        self.wind_on = max(*cfg.wind_mean, *cfg.wind_gust) > 0.0
        V = len(self.models)
        self.wind_v = np.zeros((V, 2))     # the wind each variant blows now, world x, y (m/s)
        if not self.wind_on:
            return
        rng = self._wind_rng = np.random.default_rng([cfg.seed, 0x57494E44])
        self.wind_mean = rng.uniform(cfg.wind_mean[0], cfg.wind_mean[1], V)
        self.wind_sigma = rng.uniform(cfg.wind_gust[0], cfg.wind_gust[1], V)
        a = rng.uniform(-math.pi, math.pi, V)
        self.wind_u = np.stack([np.cos(a), np.sin(a)], axis=1)   # the direction the wind blows towards
        self.wind_mean[0] = self.wind_sigma[0] = 0.0           # variant 0, the nominal model, stays calm
        self.wind_g = rng.standard_normal((V, 2)) * np.array([1.0, 0.75]) * self.wind_sigma[:, None]  # stationary
        self.wind_clock = 0.0
        self.variant_of_env = np.arange(self.N) % V
        for m in self.models[1:]:
            m.opt.density = cfg.air_density
        self._blow(0.0)

    def _blow(self, dt):
        """Advance each variant's gusts by `dt` (exact Ornstein-Uhlenbeck update) and set its opt.wind."""
        cfg = self.cfg
        if dt > 0.0:
            k = math.exp(-dt / cfg.wind_tau)
            sig = np.array([1.0, 0.75]) * self.wind_sigma[:, None]
            self.wind_g = self.wind_g * k + sig * math.sqrt(1.0 - k * k) * self._wind_rng.standard_normal(
                self.wind_g.shape)
            self.wind_clock += dt
        f = min(1.0, self.wind_clock / cfg.wind_ramp_s) if cfg.wind_ramp_s > 0 else 1.0
        u = self.wind_u
        across = np.stack([-u[:, 1], u[:, 0]], axis=1)
        self.wind_v = (f * self.wind_mean + self.wind_g[:, 0])[:, None] * u + self.wind_g[:, 1:2] * across
        self.wind_v[0] = 0.0
        for i, m in enumerate(self.models):
            m.opt.wind[0:2] = self.wind_v[i]

    def wind_of_env(self):
        """(N, 2) the wind each env stands in now, world x, y (m/s); zeros with wind off."""
        if not self.wind_on:
            return np.zeros((self.N, 2))
        return self.wind_v[self.variant_of_env]

    # ---------------------------------------------------------------- physics
    def _roll(self, idx, ctrl, nstep):
        """Advance envs `idx` by `nstep` physics steps holding `ctrl` (len(idx), nu)."""
        models = [self.model_of_env[i] for i in idx]
        control = np.repeat(ctrl[:, None, :], nstep, axis=1)
        st, sd = self.roller.rollout(models, self.datas, self.state[idx], control, nstep=nstep)
        self.state[idx] = st[:, -1, :]
        self.sens[idx] = sd[:, -1, :]

    def _standing_state(self, n):
        m, d = self.base, self.mj.MjData(self.base)
        if "stand_qpos" in self.spec:
            d.qpos[:] = self.spec["stand_qpos"]
        else:                              # the tendon humanoid: leaned about the ankles by the standing lean
            th = -STAND_LEAN
            ankle_z = 0.08
            d.qpos[0:3] = (0.0, -(0.95 - ankle_z) * math.sin(th), ankle_z + (0.95 - ankle_z) * math.cos(th) + 0.01)
            d.qpos[3:7] = (math.cos(th / 2), math.sin(th / 2), 0.0, 0.0)
            d.qpos[self.qadr] = self.default
        self.mj.mj_forward(m, d)
        s = np.empty(self.nstate)
        self.mj.mj_getState(m, d, s, self.state_kind)
        return np.tile(s, (n, 1))

    def _fallen_state(self, idx):
        """Drop from a random orientation and pose, settle with the servos holding that pose."""
        n = len(idx)
        st = self._standing_state(n)
        q = st[:, self.i_qpos:self.i_qpos + self.nq]
        q[:, 2] = 0.5 * self.stand_z
        # random orientation: lying on the back, front or a side, with a random heading and tilt
        roll_choice = self.rng.choice([math.pi / 2, -math.pi / 2, 0.0, math.pi], size=n)
        pitch_choice = self.rng.choice([math.pi / 2, -math.pi / 2], size=n)
        ax = np.where(self.rng.random(n) < 0.5, 0, 1)
        ang = np.where(ax == 0, pitch_choice, roll_choice) + self.rng.normal(0, 0.3, n)
        yaw = self.rng.uniform(-math.pi, math.pi, n)
        for k in range(n):
            a = np.zeros(3)
            a[ax[k]] = ang[k]
            qa = _axis_angle_quat(a)
            qy = _axis_angle_quat(np.array([0.0, 0.0, yaw[k]]))
            q[k, 3:7] = _quat_mul(qy, qa)
        pose = self.default + self.rng.uniform(-0.6, 0.6, (n, self.nu)) * 0.5 * (self.hi - self.lo)
        pose = np.clip(pose, self.lo, self.hi)
        q[:, 7:][:, self.qadr - 7] = pose
        st[:, self.i_qpos:self.i_qpos + self.nq] = q
        st[:, self.i_qvel:self.i_qvel + self.nv] = 0.0
        self.state[idx] = st
        self._roll(idx, pose, int(round(self.cfg.settle_s / self.base.opt.timestep)))
        return pose

    # ---------------------------------------------------------------- observations
    def _pelvis(self):
        q = self.state[:, self.i_qpos:self.i_qpos + self.nq]
        v = self.state[:, self.i_qvel:self.i_qvel + self.nv]
        quat = q[:, 3:7]
        R = _quat_to_mat(quat)                     # (N, 3, 3) body -> world
        up = R[:, 2, 2]                            # world z component of the pelvis z axis
        grav_body = np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -1.0]))
        lin_body = np.einsum("nji,nj->ni", R, v[:, 0:3])
        return q, v, R, up, grav_body, lin_body

    def observe(self):
        q, v, R, up, grav, lin = self._pelvis()
        jq = q[:, self.qadr] - self.default
        jv = v[:, self.vadr]
        s = self.sens
        gyro = s[:, self.s_gyro[0]:self.s_gyro[0] + 3]
        feet = s[:, self.s_feet] / self.weight
        pad = self._pad
        obs = np.concatenate([pad(jq), pad(0.1 * jv), grav, 0.25 * gyro, feet, pad(self.prev_action)], axis=1)
        tau = s[:, self.s_tau]
        line = self._line_force(tau) / self.f_cont
        impact = np.zeros((self.N, self.n_impact_slots))
        impact[:, :len(self.s_impact)] = s[:, self.s_impact] / self.weight
        parts = [obs, (q[:, 2] / self.stand_z)[:, None], up[:, None], lin, impact, pad(line),
                 (self.t / self.max_steps)[:, None]]
        if self.wind_on:                   # only with wind, so priv_dim is unchanged without it
            parts.append(0.1 * self.wind_of_env())
        priv = np.concatenate(parts, axis=1)
        return obs.astype(np.float32), priv.astype(np.float32)

    def _pad(self, x):
        if self.P == self.nu:
            return x
        out = np.zeros((x.shape[0], self.P))
        out[:, :self.nu] = x
        return out

    def _line_force(self, tau):
        a = np.abs(tau)
        return np.where(a < self.lin, self.pre + a / (2 * self.arm), a / self.arm)

    # ---------------------------------------------------------------- episode control
    def reset(self, idx=None):
        idx = np.arange(self.N) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return
        if self.cfg.task == "getup":
            pose = self._fallen_state(idx)
            self.target[idx] = pose
        else:
            self.state[idx] = self._standing_state(len(idx))
            self._roll(idx, np.tile(self.default, (len(idx), 1)), self.substeps)
            self.target[idx] = self.default
        self.prev_action[idx] = 0.0
        self.queue[:, idx] = (self.target[idx] - self.default) / self.half
        self.t[idx] = 0
        self.over_t[idx] = 0.0
        self.stand_t[idx] = 0.0
        self.ever_stood[idx] = False
        self.delay[idx] = self.rng.integers(0, self.cfg.max_delay_steps + 1, size=len(idx))
        self.push_at[idx] = self.rng.uniform(*self.cfg.push_interval_s, size=len(idx))

    def step(self, action: np.ndarray):
        cfg = self.cfg
        action = np.clip(action[:, :self.nu], -1.0, 1.0)     # padded slots beyond this body's joints are unused
        # actuation delay: each env applies the action from `delay` ticks ago
        self.queue = np.roll(self.queue, 1, axis=0)
        self.queue[0] = action
        applied = self.queue[self.delay, np.arange(self.N)]
        ctrl = np.clip(self.default + applied * self.half, self.lo, self.hi)
        if cfg.task == "stand":
            self._push()
        if self.wind_on:
            self._blow(self.dt_ctrl)
        self._roll(np.arange(self.N), ctrl, self.substeps)
        self.t += 1
        q, v, R, up, grav, lin = self._pelvis()
        s = self.sens
        tau = s[:, self.s_tau]
        line = self._line_force(tau) / self.f_cont
        over = (line > 1.0) & self.gated
        self.over_t = np.where(over, self.over_t + self.dt_ctrl, 0.0)
        line_over_max = self.over_t.max(axis=1)
        impact = s[:, self.s_impact].max(axis=1)
        stop = np.abs(s[:, self.s_stop]) / np.maximum(self.tau_max, 1.0)
        z = q[:, 2]
        standing = (z > cfg.stand_height * self.stand_z) & (up > cfg.stand_upright)
        self.stand_t = np.where(standing, self.stand_t + self.dt_ctrl, 0.0)
        self.ever_stood |= self.stand_t > 1.0
        # reward: progress to upright, then staying there; costs keep the hardware alive
        height = np.clip(z / self.stand_z, 0.0, 1.0)
        upright = np.clip(up, 0.0, 1.0)
        r = 0.5 * height + 0.5 * upright + 1.0 * standing
        if cfg.task == "stand":
            r = r + 0.5                                           # alive bonus
        r -= 0.5 * np.clip(line - 1.0, 0.0, None)[:, self.gated].sum(axis=1)
        r -= 0.05 * np.square(action - self.prev_action).mean(axis=1) * self.nu
        r -= 0.01 * np.square(tau / np.maximum(self.tau_max, 1e-3)).mean(axis=1) * self.nu
        r -= 0.02 * np.clip(stop - 0.2, 0.0, None).sum(axis=1)
        r -= 1.0 * np.clip((impact - cfg.impact_cost_n) / cfg.impact_cost_n, 0.0, 3.0)
        damage = (line_over_max > cfg.line_over_s) | (impact > cfg.impact_end_n)
        if cfg.task == "stand":
            damage |= (z < cfg.fall_height * self.stand_z)        # in the stand task a fall ends it
        r = np.where(damage, -10.0, r)
        timeout = self.t >= self.max_steps
        done = damage | timeout
        self.prev_action = action.copy()
        info = StepInfo(standing=standing, damage=damage, timeout=timeout & ~damage, line_over_max=line_over_max,
                        impact=impact, extras=dict(z=z, up=up, ever_stood=self.ever_stood.copy(),
                                                   over_half_s=line_over_max > cfg.line_report_s,
                                                   wmini_over=((line > 1.0) & ~self.gated).any(axis=1)))
        if self.wind_on:
            info.extras["wind"] = self.wind_of_env().copy()
        obs_done = done.copy()
        if done.any():
            self.reset(np.nonzero(done)[0])
        obs, priv = self.observe()
        return obs, priv, r.astype(np.float32), obs_done, info

    def _push(self):
        t = self.t * self.dt_ctrl
        due = t >= self.push_at
        if not due.any():
            return
        idx = np.nonzero(due)[0]
        ang = self.rng.uniform(-math.pi, math.pi, len(idx))
        self.state[idx, self.i_qvel + 0] += self.cfg.push_speed * np.cos(ang)
        self.state[idx, self.i_qvel + 1] += self.cfg.push_speed * np.sin(ang)
        self.push_at[idx] = t[idx] + self.rng.uniform(*self.cfg.push_interval_s, size=len(idx))

    @property
    def obs_dim(self):
        return 3 * self.P + 3 + 3 + len(self.s_feet)

    @property
    def act_dim(self):
        return self.P

    @property
    def priv_dim(self):
        return self.obs_dim + 1 + 1 + 3 + self.n_impact_slots + self.P + 1 + (2 if self.wind_on else 0)

    def close(self):
        self.roller.close()


# ---------------------------------------------------------------- small quaternion helpers (w, x, y, z)
def _axis_angle_quat(a):
    th = float(np.linalg.norm(a))
    if th < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = a / th
    return np.concatenate([[math.cos(th / 2)], math.sin(th / 2) * ax])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_to_mat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

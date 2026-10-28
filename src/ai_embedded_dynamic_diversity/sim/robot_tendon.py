"""Bridge to the tendon-driven humanoid of the sibling `robot` repository (displague/robot).

That repository holds a MuJoCo model in which every one of the robot's 32 body degrees of freedom is driven
the way the hardware will be: a DC motor on a gear, a self-locking stage, a drag clutch, a drum and two
fishing-line runs that stretch and go slack. Its `tools/sim_tendon.py` runs the firmware's control cascade
(gear position loop, joint-sensor trim, zero-power lock, ankle/hip balance). This module wraps that rig as an
embodiment of this project, so the model core sees a physically grounded set of anonymous channels:

- controls: the 32 joint targets (one per DOF, normalised to the joint's range), applied on top of the
  firmware balance loop, as a teleoperation or a learned policy would set them;
- sensors: joint angles and rates, winch gear angles, line tensions (the `strain` channel family), slack
  and clutch-slip flags, lock state, motor torque, a pelvis IMU, foot loads and the centre of pressure (the
  `pressure` family), the centre of mass against the support.

Nothing here is imported at package import time beyond the standard library and torch; MuJoCo and the robot
repository are looked up when a `TendonHumanoid` is created. The robot repository is found through the
`ROBOT_REPO` environment variable or as `../robot` next to this repository.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from ai_embedded_dynamic_diversity.sim.embodiments import Embodiment, get_embodiment, register_embodiment

EMBODIMENT_NAME = "humanoid_tendon32"


def find_robot_repo(explicit: str | None = None) -> str:
    """Path of the robot repository, or raise with instructions."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("ROBOT_REPO"):
        candidates.append(os.environ["ROBOT_REPO"])
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    candidates.append(os.path.join(os.path.dirname(project_root), "robot"))
    for c in candidates:
        if os.path.isfile(os.path.join(c, "tools", "sim_tendon.py")):
            return c
    raise FileNotFoundError(
        "robot repository not found; set ROBOT_REPO to a checkout of github.com/displague/robot "
        f"(tried: {', '.join(candidates)})"
    )


def _load_sim_tendon(repo: str):
    tools = os.path.join(repo, "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    try:
        import mujoco  # noqa: F401
    except ImportError as exc:
        raise ImportError("mujoco is required for the tendon humanoid: pip install mujoco") from exc
    import sim_tendon  # type: ignore

    return sim_tendon


@dataclass
class SensorLayout:
    """Names of every observation channel, in order, with the family each belongs to."""

    names: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)

    def add(self, family: str, *names: str) -> None:
        for n in names:
            self.names.append(n)
            self.families.append(family)

    def __len__(self) -> int:
        return len(self.names)


class TendonHumanoid:
    """The robot repository's tendon rig as an embodiment: reset()/step() over normalised channels.

    Controls (one per DOF) are in [0, 1]. In `delta` mode (default) 0.5 holds the current target and the
    ends move it at `max_rate` within the joint range, which keeps a random or untrained policy from
    slamming the drive train; in `absolute` mode the value maps straight onto the joint range. The firmware
    balance loop stays active underneath unless `balance=False`.
    """

    def __init__(
        self,
        control_hz: float = 50.0,
        action_mode: str = "delta",
        max_rate_deg: float = 30.0,
        balance: bool = True,
        repo: str | None = None,
        device: str = "cpu",
    ):
        self.repo = find_robot_repo(repo)
        self.st = _load_sim_tendon(self.repo)
        self.rig = self.st.Rig()
        self.rig.balance = balance
        self.device = torch.device(device)
        self.control_dt = 1.0 / float(control_hz)
        self.substeps = max(1, int(round(self.control_dt / self.rig.model.opt.timestep)))
        self.action_mode = action_mode
        self.max_rate = math.radians(max_rate_deg)
        self.winches = self.rig.w
        self.control_names = [w["name"] for w in self.winches]
        self.layout = self._layout()
        self.embodiment = Embodiment(
            name=EMBODIMENT_NAME, controls=tuple(self.control_names), sensors=tuple(sorted(set(self.layout.families)))
        )
        static = get_embodiment(EMBODIMENT_NAME)
        if static != self.embodiment:
            # the robot repository changed its joint table or this module its channels: the static entry in
            # embodiments.py has to follow, or checkpoints trained on one layout meet another
            raise RuntimeError(
                f"{EMBODIMENT_NAME}: the rig's layout no longer matches the static registration in embodiments.py "
                f"(controls {self.embodiment.controls} vs {static.controls})"
            )
        register_embodiment(self.embodiment)
        self._prev_v = None

    # ---- channel layout
    def _layout(self) -> SensorLayout:
        lay = SensorLayout()
        n = self.control_names
        lay.add("joint_angle", *[f"{k}.angle" for k in n])
        lay.add("joint_rate", *[f"{k}.rate" for k in n])
        lay.add("gear_angle", *[f"{k}.gear" for k in n])
        for k in n:
            lay.add("strain", f"{k}.lineA", f"{k}.lineB")
        for k in n:
            lay.add("slack", f"{k}.slackA", f"{k}.slackB")
        lay.add("clutch_slip", *[f"{k}.slip" for k in n])
        lay.add("lock", *[f"{k}.locked" for k in n])
        lay.add("current", *[f"{k}.motor" for k in n])
        lay.add("imu", "pelvis.qw", "pelvis.qx", "pelvis.qy", "pelvis.qz", "pelvis.wx", "pelvis.wy", "pelvis.wz",
                "pelvis.ax", "pelvis.ay", "pelvis.az")
        lay.add("pressure", "foot.L.load", "foot.R.load", "cop.x", "cop.y")
        lay.add("com", "com.dx", "com.dy", "com.height")
        return lay

    def spec(self) -> dict[str, Any]:
        """Embodiment spec in the shape `discover_from_spec` accepts, plus the full channel listing."""
        return {
            "name": EMBODIMENT_NAME,
            "controls": list(self.control_names),
            "sensors": list(self.embodiment.sensors),
            "sensor_channels": list(self.layout.names),
            "sensor_families": list(self.layout.families),
            "control_hz": 1.0 / self.control_dt,
            "action_mode": self.action_mode,
            "source": "github.com/displague/robot tools/sim_tendon.py (cad/skeleton.json)",
        }

    # ---- episode interface
    def reset(self) -> torch.Tensor:
        self.rig.reset()
        self._prev_v = None
        # let the locks settle as the robot does when it is switched on standing
        self.rig.run(0.6)
        return self.observe()

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        a = action.detach().to("cpu").flatten().clamp(0.0, 1.0).tolist()
        if len(a) != len(self.winches):
            raise ValueError(f"action has {len(a)} values, embodiment has {len(self.winches)} controls")
        for w, v in zip(self.winches, a):
            lo, hi = w["lo"], w["hi"]
            if self.action_mode == "absolute":
                tgt = lo + v * (hi - lo)
            else:
                tgt = self.rig.target[w["name"]] + (v - 0.5) * 2.0 * self.max_rate * self.control_dt
            self.rig.target[w["name"]] = min(max(tgt, lo), hi)
        e0 = sum(s["e_j"] for s in self.rig.stats.values())
        for _ in range(self.substeps):
            self.rig.step()
        e1 = sum(s["e_j"] for s in self.rig.stats.values())
        obs = self.observe()
        worst = max(((self.rig.line_force(w["tA"]), self.rig.line_force(w["tB"]), w) for w in self.winches),
                    key=lambda t: max(t[0], t[1]) / t[2]["p"]["f_cont"])
        info = {
            "time": self.rig.data.time,
            "standing": bool(self.rig.standing()),
            "power_w": (e1 - e0) / self.control_dt,
            "worst_line": worst[2]["name"],
            "worst_line_ratio": max(worst[0], worst[1]) / worst[2]["p"]["f_cont"],
        }
        return obs, info

    def observe(self) -> torch.Tensor:
        rig, d = self.rig, self.rig.data
        vals: list[float] = []
        for w in self.winches:
            lo, hi = w["lo"], w["hi"]
            vals.append((d.qpos[w["q"]] - lo) / (hi - lo))
        for w in self.winches:
            vals.append(math.tanh(d.qvel[w["dq"]] / 5.0) * 0.5 + 0.5)
        for w in self.winches:
            p = w["p"]
            ratio = p["tackle"] * p["r_p"] / p["r_d"]
            lo, hi = w["lo"], w["hi"]
            vals.append((d.qpos[w["gq"]] / ratio - lo) / (hi - lo))
        for w in self.winches:
            fc = w["p"]["f_cont"]
            vals += [min(2.0, rig.line_force(w["tA"]) / fc) / 2.0, min(2.0, rig.line_force(w["tB"]) / fc) / 2.0]
        for w in self.winches:
            vals += [1.0 if rig.line_force(w["tA"]) <= 0.0 else 0.0, 1.0 if rig.line_force(w["tB"]) <= 0.0 else 0.0]
        for w in self.winches:
            vals.append(1.0 if w.get("_slipping") else 0.0)
        for w in self.winches:
            vals.append(1.0 if w.get("_locked") else 0.0)
        for w in self.winches:
            vals.append(min(1.0, abs(d.actuator_force[w["act"]]) / w["p"]["t_stall"]))
        # pelvis IMU: orientation, angular velocity, linear acceleration (finite difference of the body velocity)
        q = d.xquat[rig.pelvis]
        vals += [0.5 + 0.5 * float(x) for x in q]
        import mujoco  # local import keeps the module importable without it

        v6 = torch.zeros(6, dtype=torch.float64)
        mujoco.mj_objectVelocity(rig.model, d, mujoco.mjtObj.mjOBJ_BODY, rig.pelvis, v6.numpy(), 0)
        w3, v3 = v6[:3], v6[3:]
        vals += [0.5 + 0.5 * math.tanh(float(x) / 3.0) for x in w3]
        if self._prev_v is None:
            acc = torch.zeros(3, dtype=torch.float64)
        else:
            acc = (v3 - self._prev_v) / self.control_dt
        self._prev_v = v3.clone()
        vals += [0.5 + 0.5 * math.tanh(float(x) / 10.0) for x in acc]
        cop, loads = rig.cop()
        weight = rig.mass * 9.81
        vals += [min(1.0, loads[0] / weight), min(1.0, loads[1] / weight)]
        e, heel, toe = rig.support()
        com = d.subtree_com[rig.pelvis]
        if cop is None:
            vals += [0.5, 0.5]
        else:
            vals += [0.5 + 0.5 * math.tanh((cop[0] - com[0]) / 0.1), 0.5 + 0.5 * math.tanh((cop[1] - com[1]) / 0.1)]
        vals += [0.5 + 0.5 * math.tanh(float(e[0]) / 0.1), 0.5 + 0.5 * math.tanh(float(e[1]) / 0.1), min(1.0, com[2] / 1.2)]
        assert len(vals) == len(self.layout), (len(vals), len(self.layout))
        return torch.tensor(vals, dtype=torch.float32, device=self.device).clamp(0.0, 1.0)


def smoke_rollout(steps: int = 100, policy: str = "hold", seed: int = 0, control_hz: float = 50.0) -> dict[str, Any]:
    """Run the rig under a trivial policy and report whether it stands and what its lines did.

    `hold` keeps every target (the balance loop alone keeps it up); `random` jitters every target within the
    rate limit, the kind of input an untrained core produces; `zero` sends 0 (every target drifts to its
    lower limit at the rate limit, which is what a saturated channel does to a tendon robot).
    """
    env = TendonHumanoid(control_hz=control_hz)
    g = torch.Generator().manual_seed(seed)
    obs = env.reset()
    n = len(env.control_names)
    standing_steps, worst, worst_ratio, power = 0, "", 0.0, 0.0
    for _ in range(steps):
        if policy == "random":
            a = torch.rand(n, generator=g)
        elif policy == "zero":
            a = torch.zeros(n)
        else:
            a = torch.full((n,), 0.5)
        obs, info = env.step(a)
        standing_steps += int(info["standing"])
        power += info["power_w"]
        if info["worst_line_ratio"] > worst_ratio:
            worst, worst_ratio = info["worst_line"], info["worst_line_ratio"]
    return {
        "embodiment": env.embodiment.name,
        "controls": n,
        "sensor_channels": len(env.layout),
        "sensor_families": list(env.embodiment.sensors),
        "steps": steps,
        "seconds": steps / control_hz,
        "policy": policy,
        "standing_fraction": standing_steps / max(1, steps),
        "standing_at_end": bool(env.rig.standing()),
        "mean_power_w": power / max(1, steps),
        "worst_line": worst,
        "worst_line_over_continuous": worst_ratio,
        "obs_shape": tuple(obs.shape),
    }

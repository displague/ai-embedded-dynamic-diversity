from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch


@dataclass(frozen=True)
class Embodiment:
    name: str
    controls: tuple[str, ...]
    sensors: tuple[str, ...]


def _indexed(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{i:03d}" for i in range(count))


EMBODIMENTS: dict[str, Embodiment] = {
    "hexapod": Embodiment(
        name="hexapod",
        controls=("leg_front_l", "leg_front_r", "leg_mid_l", "leg_mid_r", "leg_back_l", "leg_back_r", "arm_a", "arm_b", "arm_c", "arm_d"),
        sensors=("photo", "pressure", "imu", "audio", "mag"),
    ),
    "car": Embodiment(
        name="car",
        controls=("steer", "throttle", "brake", "gear", "camera_gimbal", "suspension"),
        sensors=("vision", "wheel_load", "imu", "lidar", "gps"),
    ),
    "drone": Embodiment(
        name="drone",
        controls=("rotor_fl", "rotor_fr", "rotor_rl", "rotor_rr", "pitch", "roll", "yaw", "thrust"),
        sensors=("vision", "imu", "altimeter", "wind", "rf"),
    ),
    "polymorph120": Embodiment(
        name="polymorph120",
        controls=(
            _indexed("locomotion_joint", 48)
            + _indexed("manipulator_joint", 40)
            + _indexed("spine_joint", 16)
            + _indexed("thruster", 16)
        ),
        sensors=(
            "vision",
            "stereo_audio",
            "imu",
            "pressure",
            "temperature",
            "proximity",
            "chemical",
            "magnetic",
            "rf",
            "current",
            "light",
            "strain",
        ),
    ),
    # Human-form constrained subclass profile of polymorph120.
    "humanoid120": Embodiment(
        name="humanoid120",
        controls=(
            _indexed("humanoid_leg_joint", 36)
            + _indexed("humanoid_arm_joint", 32)
            + _indexed("humanoid_spine_joint", 16)
            + _indexed("humanoid_hand_joint", 24)
            + _indexed("humanoid_neck_head_joint", 12)
        ),
        sensors=(
            "vision",
            "stereo_audio",
            "imu",
            "pressure",
            "temperature",
            "proximity",
            "chemical",
            "magnetic",
            "rf",
            "current",
            "light",
            "strain",
        ),
    ),
}


def get_embodiment(name: str) -> Embodiment:
    key = name.strip().lower()
    if key not in EMBODIMENTS:
        allowed = ", ".join(sorted(EMBODIMENTS))
        raise ValueError(f"Unknown embodiment '{name}'. Choose one of: {allowed}")
    return EMBODIMENTS[key]


def register_embodiment(embodiment: Embodiment) -> None:
    """Register a new embodiment at runtime."""
    EMBODIMENTS[embodiment.name.lower()] = embodiment


def discover_from_spec(spec: dict[str, object]) -> Embodiment:
    """Create an Embodiment from a dictionary specification."""
    name = str(spec.get("name", "unknown"))
    
    controls = spec.get("controls")
    if controls is None:
        count = int(spec.get("control_count", 0))
        controls = _indexed("control", count)
    
    sensors = spec.get("sensors")
    if sensors is None:
        count = int(spec.get("sensor_count", 0))
        sensors = _indexed("sensor", count)
        
    emb = Embodiment(name=name, controls=tuple(controls), sensors=tuple(sensors))
    register_embodiment(emb)
    return emb


def embodiment_dof_table() -> list[dict[str, int | str]]:
    rows = []
    for key in sorted(EMBODIMENTS):
        emb = EMBODIMENTS[key]
        rows.append(
            {
                "name": emb.name,
                "dof": len(emb.controls),
                "sensor_channels": len(emb.sensors),
            }
        )
    return rows


_DOF_ANATOMY: dict[str, list[tuple[float, float, float, float]]] = {
    # (cz, cy, cx, sigma) per DOF, in the same order as Embodiment.controls
    "hexapod": [
        # leg_front_l, leg_front_r
        (-0.7,  0.6, -0.5, 0.25), (-0.7,  0.6,  0.5, 0.25),
        # leg_mid_l, leg_mid_r
        (-0.7,  0.0, -0.6, 0.25), (-0.7,  0.0,  0.6, 0.25),
        # leg_back_l, leg_back_r
        (-0.7, -0.6, -0.5, 0.25), (-0.7, -0.6,  0.5, 0.25),
        # arm_a, arm_b, arm_c, arm_d
        ( 0.2,  0.3, -0.3, 0.30), ( 0.2,  0.3,  0.3, 0.30),
        ( 0.2, -0.3, -0.3, 0.30), ( 0.2, -0.3,  0.3, 0.30),
    ],
    "car": [
        # steer: lateral bias
        (-0.5,  0.0,  0.0, 0.40),
        # throttle: forward push
        (-0.5,  0.5,  0.0, 0.35),
        # brake: rearward
        (-0.5, -0.5,  0.0, 0.35),
        # gear: central
        ( 0.0,  0.0,  0.0, 0.50),
        # camera_gimbal: upper centre
        ( 0.5,  0.0,  0.0, 0.40),
        # suspension: full vertical column
        ( 0.0,  0.0,  0.0, 0.80),
    ],
    "drone": [
        # rotor_fl, rotor_fr, rotor_rl, rotor_rr
        ( 0.6,  0.5, -0.5, 0.25), ( 0.6,  0.5,  0.5, 0.25),
        ( 0.6, -0.5, -0.5, 0.25), ( 0.6, -0.5,  0.5, 0.25),
        # pitch: forward tilt
        ( 0.3,  0.4,  0.0, 0.35),
        # roll: lateral tilt
        ( 0.3,  0.0,  0.5, 0.35),
        # yaw: rotation — broad
        ( 0.5,  0.0,  0.0, 0.55),
        # thrust: full z-column
        ( 0.0,  0.0,  0.0, 0.85),
    ],
}


def _anatomy_for_indexed(prefix_groups: list[tuple[str, int, float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """Expand (prefix, count, cz, cy, cx, sigma) groups into per-DOF entries with slight jitter."""
    entries = []
    for prefix, count, cz, cy, cx, sigma in prefix_groups:
        for k in range(count):
            angle = 2.0 * math.pi * k / max(1, count)
            r = sigma * 0.4
            entries.append((
                cz + r * math.sin(angle) * 0.3,
                cy + r * math.cos(angle),
                cx + r * math.sin(angle),
                sigma,
            ))
    return entries


_DOF_ANATOMY["polymorph120"] = _anatomy_for_indexed([
    # locomotion_joint_000..047 — ground ring
    ("locomotion_joint", 48, -0.7,  0.0,  0.0, 0.20),
    # manipulator_joint_000..039 — mid-height reach
    ("manipulator_joint", 40,  0.1,  0.0,  0.0, 0.22),
    # spine_joint_000..015 — central z-axis
    ("spine_joint",       16,  0.0,  0.0,  0.0, 0.18),
    # thruster_000..015 — top hemisphere
    ("thruster",          16,  0.7,  0.0,  0.0, 0.28),
])

_DOF_ANATOMY["humanoid120"] = _anatomy_for_indexed([
    # humanoid_leg_joint_000..035 — lower body
    ("humanoid_leg_joint",      36, -0.6,  0.0,  0.0, 0.22),
    # humanoid_arm_joint_000..031 — lateral reach
    ("humanoid_arm_joint",      32,  0.1,  0.0,  0.6, 0.22),
    # humanoid_spine_joint_000..015 — vertical axis
    ("humanoid_spine_joint",    16,  0.0,  0.0,  0.0, 0.18),
    # humanoid_hand_joint_000..023 — distal reach
    ("humanoid_hand_joint",     24,  0.0,  0.0,  0.8, 0.18),
    # humanoid_neck_head_joint_000..011 — upper head
    ("humanoid_neck_head_joint",12,  0.8,  0.0,  0.0, 0.20),
])


def dof_spatial_map(
    embodiment: Union["Embodiment", str],
    z: int,
    y: int,
    x: int,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """
    Compute [control_dim, z*y*x] spatial influence tensor for an embodiment.

    Each DOF gets a Gaussian blob at its anatomical position in normalised [-1,1]³.
    Rows are L1-normalised so each DOF's total spatial influence sums to 1.
    The map is deterministic and sits at the world boundary — anonymous to the model.
    Falls back to uniform (equal weight per voxel) for DOFs with no anatomy entry.
    """
    if isinstance(embodiment, str):
        embodiment = get_embodiment(embodiment)

    dev = torch.device(device)
    control_dim = len(embodiment.controls)
    anatomy = _DOF_ANATOMY.get(embodiment.name.lower(), [])

    # Build normalised coordinate grid [3, z*y*x]
    zv = torch.linspace(-1.0, 1.0, z, device=dev)
    yv = torch.linspace(-1.0, 1.0, y, device=dev)
    xv = torch.linspace(-1.0, 1.0, x, device=dev)
    gz, gy, gx = torch.meshgrid(zv, yv, xv, indexing="ij")
    # Shape [z*y*x, 3]
    coords = torch.stack([gz.flatten(), gy.flatten(), gx.flatten()], dim=1)

    result = torch.zeros(control_dim, z * y * x, device=dev)
    uniform = torch.ones(z * y * x, device=dev) / (z * y * x)

    for i in range(control_dim):
        if i < len(anatomy):
            cz, cy, cx_val, sigma = anatomy[i]
            centre = torch.tensor([[cz, cy, cx_val]], device=dev)
            dist2 = ((coords - centre) ** 2).sum(dim=1)
            blob = torch.exp(-dist2 / (sigma ** 2 + 1e-8))
            total = blob.sum().clamp(min=1e-8)
            result[i] = blob / total
        else:
            result[i] = uniform

    return result  # [control_dim, z*y*x]


def device_map_for_embodiment(
    io_channels: int,
    embodiment: Embodiment,
    device: torch.device,
    permutation_seed: int,
) -> torch.Tensor:
    control_dim = len(embodiment.controls)
    mapping = torch.zeros(io_channels, control_dim, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(int(permutation_seed))
    perm = torch.randperm(control_dim, generator=gen, device=device)
    for i in range(io_channels):
        mapping[i, perm[i % control_dim]] = 1.0
    return mapping

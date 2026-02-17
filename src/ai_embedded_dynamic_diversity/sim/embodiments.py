from __future__ import annotations

from dataclasses import dataclass

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
}


def get_embodiment(name: str) -> Embodiment:
    key = name.strip().lower()
    if key not in EMBODIMENTS:
        allowed = ", ".join(sorted(EMBODIMENTS))
        raise ValueError(f"Unknown embodiment '{name}'. Choose one of: {allowed}")
    return EMBODIMENTS[key]


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

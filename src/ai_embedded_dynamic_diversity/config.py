from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    signal_dim: int = 64
    hidden_dim: int = 128
    edge_nodes: int = 96
    memory_slots: int = 64
    memory_dim: int = 64
    io_channels: int = 24
    max_remap_groups: int = 8
    gating_mode: str = "sigmoid"
    topk_gating: int = 0
    enable_dmd_gating: bool = False
    enable_phase_gating: bool = False
    enable_multi_scale_gating: bool = False
    emergent_signal_dim: int = 8


@dataclass
class HazardZoneConfig:
    """Configuration for a single environmental hazard zone."""
    # Fractional centre position in [0,1] for each axis
    cx: float = 0.5
    cy: float = 0.5
    cz: float = 0.5
    # Fractional half-extents in [0,1] for each axis
    rx: float = 0.1
    ry: float = 0.1
    rz: float = 0.5
    kind: str = "light_triggered"   # "light_triggered" | "airflow" | "periodic"
    threshold: float = 0.5
    shadow_safe: bool = True
    # For "periodic" kind: period in steps
    period: int = 16
    hazard_weight: float = 0.4


@dataclass
class WorldConfig:
    x: int = 20
    y: int = 20
    z: int = 10
    resource_channels: int = 5
    decay: float = 0.03
    actuation_delay_steps: int = 0
    actuation_noise_std: float = 0.0
    sensor_latency_steps: int = 0
    sensor_dropout_burst_prob: float = 0.0
    surface_friction_scale: float = 1.0
    disturbance_correlation_horizon: int = 0
    # Structural occlusion objects (T-shapes)
    num_occlusion_objects: int = 0
    occlusion_seed: int = 0
    # Anonymous physics objects (push/pull/stack/bridge)
    num_physics_objects: int = 0
    phys_mass: float = 1.0
    phys_friction: float = 0.85
    # Environmental hazard zones
    hazard_zones: List[HazardZoneConfig] = field(default_factory=list)


@dataclass
class TrainConfig:
    batch_size: int = 16
    unroll_steps: int = 12
    lr: float = 2e-4
    epochs: int = 20
    entropy_weight: float = 0.01
    energy_weight: float = 0.05
    memory_consistency_weight: float = 0.15
    remap_probability: float = 0.2
    device: str = "cuda"


@dataclass
class ExportConfig:
    opset: int = 17
    quantize_dynamic: bool = False


def model_config_for_profile(profile: str) -> ModelConfig:
    normalized = profile.strip().lower()
    if normalized in {"base", "laptop", "train"}:
        return ModelConfig()
    if normalized in {"pi5", "pi-5", "raspberry-pi-5", "edge"}:
        return ModelConfig(
            signal_dim=48,
            hidden_dim=64,
            edge_nodes=64,
            memory_slots=24,
            memory_dim=32,
            io_channels=16,
            max_remap_groups=8,
        )
    raise ValueError(f"Unknown model profile: {profile}")


def world_config_for_profile(profile: str) -> WorldConfig:
    normalized = profile.strip().lower()
    if normalized in {"base", "default", "train"}:
        return WorldConfig()
    if normalized in {"pi5", "pi-5", "raspberry-pi-5", "edge"}:
        # Memory-constrained: keep original 20×20×10
        return WorldConfig(x=20, y=20, z=10)
    if normalized in {"large_v1", "large-v1"}:
        return WorldConfig(
            x=56,
            y=56,
            z=28,
            resource_channels=6,
            decay=0.025,
            actuation_delay_steps=1,
            actuation_noise_std=0.02,
            sensor_latency_steps=1,
            sensor_dropout_burst_prob=0.03,
            surface_friction_scale=0.85,
            disturbance_correlation_horizon=5,
        )
    if normalized in {"new_env_v1", "new-env-v1"}:
        return WorldConfig(
            x=40,
            y=40,
            z=20,
            resource_channels=6,
            decay=0.028,
            actuation_delay_steps=1,
            actuation_noise_std=0.02,
            sensor_latency_steps=1,
            sensor_dropout_burst_prob=0.03,
            surface_friction_scale=0.9,
            disturbance_correlation_horizon=4,
            num_occlusion_objects=3,
            occlusion_seed=42,
            num_physics_objects=3,
            phys_mass=1.2,
            phys_friction=0.82,
            hazard_zones=[
                HazardZoneConfig(cx=0.75, cy=0.25, cz=0.5, rx=0.08, ry=0.08, rz=0.5,
                                 kind="light_triggered", threshold=0.45, shadow_safe=True, hazard_weight=0.35),
                HazardZoneConfig(cx=0.25, cy=0.75, cz=0.5, rx=0.08, ry=0.08, rz=0.5,
                                 kind="periodic", threshold=0.0, period=20, shadow_safe=False, hazard_weight=0.3),
                HazardZoneConfig(cx=0.5, cy=0.5, cz=0.3, rx=0.06, ry=0.06, rz=0.3,
                                 kind="airflow", threshold=0.4, shadow_safe=False, hazard_weight=0.25),
            ],
        )
    if normalized in {"large_v1_extreme", "large-v1-extreme"}:
        return WorldConfig(
            x=64,
            y=64,
            z=32,
            resource_channels=6,
            decay=0.022,
            actuation_delay_steps=2,
            actuation_noise_std=0.04,
            sensor_latency_steps=2,
            sensor_dropout_burst_prob=0.06,
            surface_friction_scale=0.75,
            disturbance_correlation_horizon=9,
        )
    raise ValueError(f"Unknown world profile: {profile}")

from dataclasses import dataclass


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
    if normalized in {"base", "default", "pi5", "edge", "train"}:
        return WorldConfig()
    if normalized in {"large_v1", "large-v1"}:
        return WorldConfig(
            x=28,
            y=28,
            z=14,
            resource_channels=6,
            decay=0.025,
            actuation_delay_steps=1,
            actuation_noise_std=0.02,
            sensor_latency_steps=1,
            sensor_dropout_burst_prob=0.03,
            surface_friction_scale=0.85,
            disturbance_correlation_horizon=5,
        )
    if normalized in {"large_v1_extreme", "large-v1-extreme"}:
        return WorldConfig(
            x=32,
            y=32,
            z=16,
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

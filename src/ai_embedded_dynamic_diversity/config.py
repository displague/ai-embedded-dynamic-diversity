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


@dataclass
class WorldConfig:
    x: int = 20
    y: int = 20
    z: int = 10
    resource_channels: int = 5
    decay: float = 0.03


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

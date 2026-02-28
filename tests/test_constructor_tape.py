from __future__ import annotations

from ai_embedded_dynamic_diversity.models import DescriptionCopier, UniversalConstructor
from ai_embedded_dynamic_diversity.models.constructor_tape import ConstructorTape, config_from_constructor_tape


def test_constructor_tape_roundtrip_and_config_parse() -> None:
    tape = ConstructorTape(
        version=1,
        tokens=[
            "signal_dim=32",
            "hidden_dim=48",
            "memory_slots=12",
            "memory_dim=24",
            "io_channels=10",
            "gating_mode=symplectic",
            "enable_dmd_gating=true",
            "topk_gating=3",
        ],
    )
    payload = tape.to_payload()
    recovered = ConstructorTape.from_payload(payload)
    cfg = config_from_constructor_tape(recovered, profile="base")
    assert cfg.signal_dim == 32
    assert cfg.hidden_dim == 48
    assert cfg.memory_slots == 12
    assert cfg.memory_dim == 24
    assert cfg.io_channels == 10
    assert cfg.gating_mode == "symplectic"
    assert cfg.enable_dmd_gating is True
    assert cfg.topk_gating == 3


def test_universal_constructor_builds_model_from_tape() -> None:
    tape = ConstructorTape(
        version=1,
        tokens=["signal_dim=24", "hidden_dim=40", "edge_nodes=18", "memory_slots=6", "memory_dim=8", "io_channels=4", "max_remap_groups=3"],
    )
    built = UniversalConstructor(base_profile="base").build(tape, seed_state=7)
    assert built.config.signal_dim == 24
    assert built.config.io_channels == 4
    assert built.model.router.io_channels == 4
    assert built.deterministic_seed >= 0


def test_description_copier_noise_profiles_affect_fidelity() -> None:
    tape = ConstructorTape(version=1, tokens=["signal_dim=48", "hidden_dim=64", "memory_slots=24"])
    copier = DescriptionCopier()
    copied_none, fidelity_none = copier.copy(tape, noise_profile="none", seed=11)
    copied_mild, fidelity_mild = copier.copy(tape, noise_profile="mild", seed=11)
    assert copied_none.tokens == tape.tokens
    assert fidelity_none == 1.0
    assert 0.0 <= fidelity_mild <= 1.0
    assert fidelity_mild <= fidelity_none

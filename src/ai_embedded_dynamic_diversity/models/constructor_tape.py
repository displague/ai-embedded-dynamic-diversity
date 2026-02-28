from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ai_embedded_dynamic_diversity.config import ModelConfig, model_config_for_profile


_INT_FIELDS = {
    "signal_dim",
    "hidden_dim",
    "edge_nodes",
    "memory_slots",
    "memory_dim",
    "io_channels",
    "max_remap_groups",
    "topk_gating",
    "emergent_signal_dim",
}
_BOOL_FIELDS = {"enable_dmd_gating", "enable_phase_gating", "enable_multi_scale_gating"}
_STR_FIELDS = {"gating_mode"}


@dataclass
class ConstructorTape:
    version: int
    tokens: list[str]

    def to_payload(self) -> dict:
        return {"version": int(self.version), "tokens": list(self.tokens)}

    @classmethod
    def from_payload(cls, payload: dict) -> "ConstructorTape":
        version = int(payload.get("version", 1))
        raw_tokens = payload.get("tokens", [])
        if not isinstance(raw_tokens, list):
            raise ValueError("Constructor tape tokens must be a list")
        tokens = [str(token).strip() for token in raw_tokens if str(token).strip()]
        if not tokens:
            raise ValueError("Constructor tape is empty")
        return cls(version=version, tokens=tokens)

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "ConstructorTape":
        return cls.from_payload(json.loads(raw))


def load_constructor_tape(path: str) -> ConstructorTape:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return ConstructorTape.from_payload(payload)


def save_constructor_tape(tape: ConstructorTape, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tape.to_json() + "\n", encoding="utf-8")


def config_from_constructor_tape(tape: ConstructorTape, profile: str = "base") -> ModelConfig:
    cfg = model_config_for_profile(profile)
    for token in tape.tokens:
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        field = key.strip()
        value = raw_value.strip()
        if field in _INT_FIELDS:
            setattr(cfg, field, int(value))
        elif field in _BOOL_FIELDS:
            lowered = value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                setattr(cfg, field, True)
            elif lowered in {"0", "false", "no", "off"}:
                setattr(cfg, field, False)
            else:
                raise ValueError(f"Invalid boolean token value for '{field}': {value}")
        elif field in _STR_FIELDS:
            setattr(cfg, field, value)
        else:
            raise ValueError(f"Unknown constructor token field: {field}")
    return cfg

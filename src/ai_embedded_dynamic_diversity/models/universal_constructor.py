from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from ai_embedded_dynamic_diversity.config import ModelConfig
from ai_embedded_dynamic_diversity.models.constructor_tape import ConstructorTape, config_from_constructor_tape
from ai_embedded_dynamic_diversity.models.core import ModelCore


@dataclass
class ConstructedCore:
    model: ModelCore
    config: ModelConfig
    tape: ConstructorTape
    deterministic_seed: int


class UniversalConstructor:
    def __init__(self, base_profile: str = "base"):
        self.base_profile = base_profile

    def build(self, tape: ConstructorTape, seed_state: int = 0) -> ConstructedCore:
        token_blob = "|".join(tape.tokens)
        digest = hashlib.sha256(f"{token_blob}:{seed_state}".encode("utf-8")).hexdigest()
        deterministic_seed = int(digest[:8], 16)
        config = config_from_constructor_tape(tape, profile=self.base_profile)
        model = ModelCore(**config.__dict__)
        return ConstructedCore(
            model=model,
            config=config,
            tape=tape,
            deterministic_seed=deterministic_seed,
        )


class DescriptionCopier:
    def copy(self, tape: ConstructorTape, noise_profile: str = "none", seed: int = 0) -> tuple[ConstructorTape, float]:
        normalized = noise_profile.strip().lower()
        rates = {
            "none": 0.0,
            "mild": 0.02,
            "aggressive": 0.08,
        }
        if normalized not in rates:
            raise ValueError(f"Unknown noise_profile '{noise_profile}'")

        mutate_rate = rates[normalized]
        if mutate_rate <= 0.0:
            return ConstructorTape(version=tape.version, tokens=list(tape.tokens)), 1.0

        rng = random.Random(seed)
        out_tokens: list[str] = []
        edits = 0
        total = 0
        for token in tape.tokens:
            chars = list(token)
            for idx, ch in enumerate(chars):
                total += 1
                if ch.isdigit() and rng.random() < mutate_rate:
                    edits += 1
                    chars[idx] = str(rng.randint(0, 9))
            out_tokens.append("".join(chars))
        fidelity = 1.0 - (edits / max(1, total))
        return ConstructorTape(version=tape.version, tokens=out_tokens), max(0.0, min(1.0, fidelity))

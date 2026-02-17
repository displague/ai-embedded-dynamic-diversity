# Research Backlog

1. Add policy-level objective heads for explicit downstream tasks.
2. Add true multi-agent competition inside the same world instance.
3. Implement quantization-aware training and ONNX Runtime benchmark on Raspberry Pi 5.
4. Replace UDP test bridge with hardware bus adapters (I2C/SPI/CAN/UART).
5. Add checkpoint comparison dashboards across multiple embodiments.
6. Tune curriculum schedules specifically against hardy-line profiles (`storm`, `blackout`, `crosswind`).
7. Add car-focused robustness objectives and weighting to reduce persistent high mismatch under hardy profiles.
8. Tune transfer-fitness coupling so evolutionary selection correlates with cross-eval transfer gains (current transfer can improve while legacy fitness degrades).
9. Add progressive embodiment curriculum (`hexapod/car/drone` -> `polymorph120`) and measure whether staged complexity reduces catastrophic remap mismatch.
10. Standardize CUDA package sourcing for `uv` lock/index so plain `uv run` remains on CUDA torch (strict-device guard already blocks silent fallback).
11. Add emergent mimicry research track (environmental/peer/threat mimicry) using self-supervised objectives and population interaction, avoiding explicit forced imitation labels.
12. Add emergent signaling objectives with anonymous channels (signal generation + detection) and quantify reliability under remaps/noise.
13. Add threat-evasion benchmark suite (disturbance adversaries + threat agents) and track evasion-success vs vitality/transfer tradeoffs.
14. Add multi-generation genetic-memory survival experiments with explicit long-horizon persistence metrics (recovery half-life, lineage survival score).
15. Add conjoining studies: tool-use coupling, genetic bonding proxies, and organism-formation behavior in multi-agent embodied environments.

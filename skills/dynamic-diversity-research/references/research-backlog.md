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

## Prioritized Extension (2026-02-17)

### Phase 1: Foundation Solidification (Now -> Near-term)
1. Embodiment profiling CLI (`add-sim profiler`) for bottleneck I/O channels, readiness/gating saturation, memory pressure, and P50/P95 step latency. Status: implemented.
2. Checkmate transfer suite: establish fixed acceptance threshold (`>=85%` normalized effectiveness) across `hexapod/car/drone/polymorph120` with high-repeat confidence intervals. Status: implemented (needs tighter thresholds/harder heldout regimes for discrimination).
3. Transfer matrix harness: train on subset embodiments and evaluate zero-shot transfer on held-out embodiments with matrix reports. Status: implemented.
4. Latency profile matrix: standardize P50/P95 runs across CPU/CUDA now; add Pi5/Jetson/mobile accelerators when hardware is available.
5. CUDA reproducibility in `uv` lock/index workflow so `uv run` remains CUDA-backed.

### Phase 2: Adaptive Diversity (Near -> Mid-term)
1. Online remap learning inside episode (learned remap policy vs static periodic remap).
2. Dynamic embodiment discovery at runtime (I/O schema inference without recompilation).
3. Few-shot embodiment adaptation (`<10` gradient steps) benchmark and adaptation-slope tracking.
4. Genetic memory bank for warm-start retrieval by morphology family.
5. Multi-scale gating (global policy gate + local edge response gate).

### Phase 3: Portability and Robustness (Mid -> Long-term)
1. Hardware-agnostic I/O abstraction for single-checkpoint runtime across Pi5/phone/embedded Linux.
2. Noisy signal resilience training (sensor dropout + quantization noise).
3. Embodiment interpolation benchmarks (morphology changes during deployment).
4. Energy-aware routing constraints in readiness and gating objectives.

### Phase 4: Drop-in Deployment (Long-term)
1. ONNX-first export/inference path for minimal runtime dependency.
2. Telemetry feedback loop from runtime adaptation metrics into training curriculum.
3. Lightweight ensemble routing across specialized sub-models.
4. Runtime embodiment switching without model reload (stateless memory and handoff policy).

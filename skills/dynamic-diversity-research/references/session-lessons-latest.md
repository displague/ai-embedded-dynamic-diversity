# Session Lessons (Latest)

Date: 2026-03-01

## High-Value Outcomes

1. **Multi-scale gating** (Global + Local) significantly improves long-horizon stability and recovery from signal shocks compared to single-scale sigmoid/symplectic baselines.
2. **INT8 Dynamic Quantization** provides the optimal balance for Pi 5 deployment, achieving ~0.43ms latency with a reduced 165KB footprint.
3. **Emergent Symbiogenesis** is quantitatively measurable via signal variance and decoding accuracy, but collapses under extreme physical survival pressure (>0.85 volatility) without balanced curriculum weighting.
4. **Hardware-Agnostic I/O Abstraction** (IOAdapter) successfully bridges ModelCore with synthetic and concrete (I2C/SMBus2) hardware interfaces.
5. **Few-Shot Adaptation** protocol effectively reduces reconstruction loss on novel embodiments in <10 gradient steps.
6. **New champion line** established at `artifacts/model-core-champion-v09.pt` with reported `0.8111` score after a `1200`-generation run.
7. **Adaptive Loss-Weighting Controller** prevented signaling collapse in extended extreme training, enabling the v09 breakthrough.
8. **Extreme-stress behavior evidence** captured in `artifacts/v09-vs-v08-car-storm.gif` and summarized in `artifacts/report-extreme-v02-cap.md`.
9. **Shared-environment 1200-generation retrain (v03)** completed with all five embodiments in `large_v1_extreme`, seeded from leading checkpoints; this improved weighted transfer but reduced blended capability score.
10. **Failure mode identified:** transfer gains can hide capability collapse when `signal_reliability` and `conjoining_gain` decay under aggressive objective weighting.
11. **Storyboard evidence expanded** with 30 compare GIFs + montage (`artifacts/viz-storyboard-v07-v03-calib-large-v1`) showing strongest v03 regressions in `latency-storm/storm` for `hexapod`, `car`, and `drone`.
12. **Capability guardrail controls implemented** in coevolution selection with floor-based penalties and per-generation proxy telemetry (`mean_signal_reliability`, `mean_conjoining_gain`).

## Immediate Handoff Continuation

1. Tune capability guardrail thresholds/weights and rerun a constrained shared-environment cycle to recover capability while preserving transfer.
2. Add periodic capability probe checkpoints during training and terminate/ratchet when capability decays while transfer rises.
3. Run targeted curriculum pass for `latency-storm` and `storm` (especially `hexapod`, `car`, `drone`) before next 1200-generation promotion attempt.

## Reuse Tags

- `multi-scale-gating`
- `int8-deployment`
- `symbiogenesis-audit`
- `hil-readiness`
- `survival-curriculum`
- `v09-champion`
- `1200-gen-run`
- `adaptive-loss-success`
- `shared-env-v03-regression`
- `capability-guardrail-needed`
- `storyboard-v07-v03-calib-large`

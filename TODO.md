# TODO

## High Priority

- [x] Run long-form RTX 5080 training sweep with curriculum schedules across `hexapod`, `car`, and `drone` embodiment mappings.
- [x] Run `add-train-parallel` on RTX 5080 with 6-8 variants and compare fitness/logs in `artifacts/parallel/summary.json`.
- [x] Capture and compare side-by-side adaptation visualizations for baseline vs co-evolution checkpoints.
- [x] Add cross-embodiment transfer ranking (`add-cross-eval`) over `hexapod`, `car`, and `drone`.
- [x] Add cross-eval report generation (`add-cross-report`) with rank/delta outputs.
- [x] Add and run hardy-line transfer evaluation profile across top checkpoints.
- [x] Add artifact interpretation guide (goals, "good" thresholds, failure patterns, decision rules).
- [x] Add higher-complexity embodiment (`polymorph120`) with DOF multiple of existing embodiments.
- [x] Add embodiment-aware transfer training objective and aggressive parallel launcher options (AMP/TF32/device pool/warm-start).
- [ ] Reduce hardy-line `car` mismatch further (improved to `~0.9585`, still dominant transfer bottleneck vs `hexapod`/`drone`; next target `< 0.90`).
- [x] Run car-priority hardy ranking using `--embodiment-weights` and compare top-1 stability vs unweighted champion.
- [x] Run focused warm-start fine-tune with embodiment-transfer loss and verify hardy transfer/mismatch impact.
- [ ] Measure Raspberry Pi 5 latency and memory for `model-core-pi5-int8.ts` with `add-bench` on-device.

## Medium Priority

- [ ] Add policy-level task objectives on top of adaptation (navigation, stability, manipulation).
- [ ] Implement quantization-aware training and compare against dynamic quantization export.
- [ ] Integrate ONNX Runtime benchmark path for Pi 5.
- [ ] Evaluate memory gating ablations: `sigmoid` vs `symplectic` vs `symplectic+dmd+phase` using `add-gating-bench`.
- [ ] Add manifold paging objective reduction beyond top-k slot gating (per-sample adaptive page budget).
- [ ] Tune curriculum schedule for hardy-line gains (current curriculum run improved but did not beat non-curriculum focused run).
- [ ] Continue car-focused objective weighting/sampling in training (storm/crosswind emphasis) to push hardy car mismatch below `0.9`.
- [ ] Make CUDA PyTorch install reproducible in `uv` workflow (avoid `uv run` reverting to CPU torch from lockfile).

## Hardware-In-The-Loop

- [ ] Replace UDP bridge with real transport adapters (I2C/SPI/CAN/UART) and protocol contracts.
- [ ] Build sensor normalization wrappers that preserve anonymous-channel semantics.
- [ ] Add actuator safety constraints and rate limits in the HIL path.

## Visualization & Analysis

- [ ] Add metric export (`csv/json`) from visualization runs.
- [ ] Add batch rendering scripts for multiple force modes (`poke`, `press`, `push`, `continuous-blow`, `thrust`, `move`).
- [ ] Add object trajectory overlays to comparison visualizations.

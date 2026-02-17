# TODO

## High Priority

- [ ] Run long-form RTX 5080 training sweep with curriculum schedules across `hexapod`, `car`, and `drone` embodiment mappings.
- [x] Run `add-train-parallel` on RTX 5080 with 6-8 variants and compare fitness/logs in `artifacts/parallel/summary.json`.
- [ ] Capture and compare side-by-side adaptation visualizations for baseline vs co-evolution checkpoints.
- [x] Add cross-embodiment transfer ranking (`add-cross-eval`) over `hexapod`, `car`, and `drone`.
- [x] Add cross-eval report generation (`add-cross-report`) with rank/delta outputs.
- [ ] Measure Raspberry Pi 5 latency and memory for `model-core-pi5-int8.ts` with `add-bench` on-device.

## Medium Priority

- [ ] Add policy-level task objectives on top of adaptation (navigation, stability, manipulation).
- [ ] Implement quantization-aware training and compare against dynamic quantization export.
- [ ] Integrate ONNX Runtime benchmark path for Pi 5.
- [ ] Evaluate memory gating ablations: `sigmoid` vs `symplectic` vs `symplectic+dmd+phase` using `add-gating-bench`.
- [ ] Add manifold paging objective reduction beyond top-k slot gating (per-sample adaptive page budget).

## Hardware-In-The-Loop

- [ ] Replace UDP bridge with real transport adapters (I2C/SPI/CAN/UART) and protocol contracts.
- [ ] Build sensor normalization wrappers that preserve anonymous-channel semantics.
- [ ] Add actuator safety constraints and rate limits in the HIL path.

## Visualization & Analysis

- [ ] Add metric export (`csv/json`) from visualization runs.
- [ ] Add batch rendering scripts for multiple force modes (`poke`, `press`, `push`, `continuous-blow`, `thrust`, `move`).
- [ ] Add object trajectory overlays to comparison visualizations.

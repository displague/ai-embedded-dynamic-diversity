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
- [x] Reduce hardy-line `car` mismatch below `0.90` (now `~0.7131` with `artifacts/parallel-cuda-long-v02/variant-01.pt`).
- [x] Reduce hardy-line `car` mismatch further below `< 0.70` with improved stability (`~0.509` in `artifacts/cross-eval-cuda-converge-v03-top4-hardy-r6.json`).
- [ ] Push hardy-line `car` mismatch toward `< 0.50` while improving car transfer variance under `storm/crosswind`.
- [x] Run car-priority hardy ranking using `--embodiment-weights` and compare top-1 stability vs unweighted champion.
- [x] Run focused warm-start fine-tune with embodiment-transfer loss and verify hardy transfer/mismatch impact.
- [ ] Measure Raspberry Pi 5 latency and memory for `model-core-pi5-int8.ts` with `add-bench` on-device.

## Medium Priority

- [ ] Add policy-level task objectives on top of adaptation (navigation, stability, manipulation).
- [ ] Add a learned (non-hardcoded) mimicry training track: environmental mimicry, peer mimicry, and threat mimicry from anonymous signals, optimized for transfer fitness rather than explicit imitation labels.
- [ ] Add emergent signaling track (no explicit channel labels): produce robust internal/external signals under noisy/remapped embodiment conditions.
- [ ] Add signal-detection track: distinguish peer/environment/threat signal patterns from anonymous streams with remap-robust decoding accuracy and low false positives.
- [ ] Add evasion track: learn adaptive evasive behavior under hostile force/crosswind/threat-agent perturbations while preserving vitality and transfer.
- [x] Add signaling+detection+evasion evaluation harness (`cross-eval` capability profile + tests) with reproducible scenario seeds and success metrics.
- [ ] Extend capability harness with explicit mimicry and conjoining proxies (peer/environment/threat imitation quality + cooperative gain metrics).
- [ ] Add genetic+memory survival curriculum: multi-generation selection with memory persistence objectives focused on hardy-line recovery and long-horizon stability.
- [ ] Add conjoining research track: environment/tool-use coupling, genetic bonding proxies, and multi-agent organism-formation objectives under anonymous I/O.
- [ ] Implement quantization-aware training and compare against dynamic quantization export.
- [ ] Integrate ONNX Runtime benchmark path for Pi 5.
- [ ] Evaluate memory gating ablations: `sigmoid` vs `symplectic` vs `symplectic+dmd+phase` using `add-gating-bench`.
- [ ] Add manifold paging objective reduction beyond top-k slot gating (per-sample adaptive page budget).
- [ ] Tune curriculum schedule for hardy-line gains (current curriculum run improved but did not beat non-curriculum focused run).
- [x] Continue car-focused objective weighting/sampling in training (storm/crosswind emphasis) to push hardy car mismatch below `0.9`.
- [ ] Make CUDA PyTorch install reproducible in `uv` lock workflow (strict-device guard is in place; remaining work is lock/index standardization so plain `uv run` keeps CUDA torch).
- [ ] Promote next champion after high-repeat validation (`runs_per_combo >= 6`) on both transfer-only and capability-weighted hardy profiles.

## Hardware-In-The-Loop

- [ ] Replace UDP bridge with real transport adapters (I2C/SPI/CAN/UART) and protocol contracts.
- [ ] Build sensor normalization wrappers that preserve anonymous-channel semantics.
- [ ] Add actuator safety constraints and rate limits in the HIL path.

## Visualization & Analysis

- [ ] Add metric export (`csv/json`) from visualization runs.
- [ ] Add batch rendering scripts for multiple force modes (`poke`, `press`, `push`, `continuous-blow`, `thrust`, `move`).
- [ ] Add object trajectory overlays to comparison visualizations.

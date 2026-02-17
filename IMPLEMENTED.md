# IMPLEMENTED

## Completed

- Added Pi-oriented model profile (`pi5`) with smaller core dimensions for edge deployment.
- Added co-evolution training mode with elite selection and mutation.
- Added curriculum schedules for remap severity and environment volatility in training.
- Added latent genetic memory persistence across episodes in training.
- Extended world simulation with environment controls:
  - wind vector
  - dynamic light source
  - force interaction and object movement
- Added reproducible embodiment schemas (`hexapod`, `car`, `drone`) and deterministic I/O remapping.
- Added visualization command for single-run adaptation with environment control traces.
- Added side-by-side checkpoint comparison visualization under the same remap/environment schedule.
- Added dynamic quantized TorchScript export path.
- Added Pi-focused benchmark CLI.
- Added UDP hardware-in-the-loop adapter CLI.
- Added configurable memory gating research modes:
  - `sigmoid` baseline
  - `symplectic` (tanh + max pooling + learnable scale)
  - optional `top-k` memory slot paging
  - optional phase-aware gate modulation
  - optional DMD-inspired gate modulation
- Added parallel diverse-fitness training launcher (`add-train-parallel`).
- Added symplectic/gating verification benchmark CLI (`add-gating-bench`) with long-horizon recovery score.
- Added structured run metric tracking (`*.metrics.json`) including active flags, per-epoch/generation performance (`mean_step_ms`), and fitness.
- Added parallel summary aggregation with flags/performance/fitness in `artifacts/parallel*/summary.json`.
- Added primitive principle tests in `tests/test_principles.py` for:
  - memory update from zero state
  - remap-driven adaptation signal shift
  - interconnected memory->I/O influence
  - top-k memory paging sparsity
  - environment control impact on stress/object motion
- Ran a longer diverse-fitness sweep in `artifacts/parallel-long` (8 variants, 30 epochs).
- Added cross-embodiment transfer evaluator (`add-cross-eval`) with scenario stressors and ranking output in `artifacts/cross-eval-summary.json`.
- Added transfer-focused tests in `tests/test_cross_eval.py` (recovery metric + rollout metric structure).
- Added cross-eval reporting utility (`add-cross-report`) to emit ranked Markdown/CSV with per-embodiment deltas.
- Added explicit checkpoint-list support in `add-cross-eval` for focused A/B/C comparisons.
- Ran focused long retrain (`artifacts/focused-variant03-long.pt`) and confirmed transfer improvement vs prior top checkpoints.
- Added hardy-line scenario profile support in cross-eval (`standard`, `hardy`, `extreme`) with stronger disturbance cases (`storm`, `blackout`, `crosswind`).
- Ran focused curriculum retrain and hardy-line ranking (`artifacts/cross-eval-hardy-focused-vs-top.json`) to compare robustness under harsher conditions.
- Ran additional curriculum sweep variants (`focused-curriculum-a`, `focused-curriculum-b`) and ranked them under hardy profile (`artifacts/cross-eval-hardy-curriculum-sweep.json`).
- Added tracking files (`TODO.md`, `IMPLEMENTED.md`) and updated backlog/docs.

## Lessons Learned

- Detaching rollout memory between steps is required to avoid graph reuse errors in unrolled training.
- Returning dictionaries from model forward paths complicates export; wrappers with tensor/tuple outputs simplify TorchScript/ONNX.
- Visual comparisons are only meaningful when both models share identical initial state and remap/control schedules.
- Anonymous-channel adaptation still benefits from embodiment schemas when they are used only as reproducible mapping generators.
- Environment perturbations (wind/light/force) need to be explicit controls rather than implicit randomness to debug adaptation behavior.
- A lightweight HIL bridge is useful early, but production robotics needs transport-specific safety and timing guarantees.
- Symplectic-style gating and DMD/phase modulation are easiest to integrate when they preserve backward compatibility with the baseline memory API.
- Top-k slot paging is a practical first step toward manifold paging, but objective-level paging still needs dedicated work.
- Curriculum helps hard-regime robustness, but schedule parameters need tuning; in current hardy evaluation, non-curriculum focused training remained slightly better on transfer score.
- Curriculum variant A improved recovery but still trailed the non-curriculum focused model on overall hardy transfer score.

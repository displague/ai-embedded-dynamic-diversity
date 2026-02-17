# Experiment Log

Use compact entries:
- date:
- hypothesis:
- change:
- result:
- next action:

- date: 2026-02-13
- hypothesis: Embodiment-aware remap schedules plus explicit environment controls will make adaptation behavior easier to observe and compare.
- change: Added world environment controls (wind/light/force/object move), embodiment schemas, and single+comparison visualizer commands; added curriculum schedules and latent genetic memory in training.
- result: Smoke training and visualization completed; generated `artifacts/adaptation-controlled.gif` and `artifacts/adaptation-compare-controlled.gif` with remap events and adaptation traces.
- next action: Run longer RTX 5080 training and compare checkpoints across all three embodiments.
- date: 2026-02-13
- hypothesis: Longer parallel co-evolution with optional gating flags plus curriculum/genetic-memory toggles should produce a clearer best-performing variant and better run-to-run comparability.
- change: Added principle tests (`tests/test_principles.py`) and ran `add-train-parallel` equivalent with 8 variants, 30 epochs, mixed optional flags into `artifacts/parallel-long`.
- result: All 8 variants completed successfully and produced checkpoints, logs, and metrics summary (`artifacts/parallel-long/summary.json`). Primitive principle tests passed (5/5).
- next action: Rank top variants by fitness and generate side-by-side visualization comparisons for best 2-3 models across `hexapod/car/drone`.
- date: 2026-02-13
- hypothesis: Direct cross-embodiment ranking across shared scenario schedules will expose transfer-generalization differences better than single-fitness ranking.
- change: Added `add-cross-eval` (`src/ai_embedded_dynamic_diversity/train/cross_eval_cli.py`) and transfer tests (`tests/test_cross_eval.py`), then evaluated `artifacts/parallel-long`.
- result: `artifacts/cross-eval-summary.json` ranked `artifacts/parallel-long/variant-03.pt` highest by overall transfer score (~0.2961) across `hexapod/car/drone` under `mild/gust/force` scenarios.
- next action: Generate side-by-side visual comparisons for top-ranked checkpoints (`variant-03` vs `variant-02`) across each embodiment and inspect failure modes where mismatch spikes after remaps.
- date: 2026-02-13
- hypothesis: Top transfer-ranked checkpoints should show more consistent vitality under remaps across all embodiments.
- change: Generated side-by-side visual comparisons for top ranked models (`variant-03` vs `variant-02`) across `hexapod`, `car`, and `drone`.
- result: Produced `artifacts/cross-eval-top-hexapod.gif`, `artifacts/cross-eval-top-car.gif`, and `artifacts/cross-eval-top-drone.gif`; left model (`variant-03`) showed slightly higher final vitality in all three.
- next action: Build automated delta charts (mismatch/vitality) from `cross-eval-summary.json` and inspect per-remap recovery dips.
- date: 2026-02-13
- hypothesis: A focused long retrain on the best-transfer configuration should improve cross-embodiment transfer score beyond the current top checkpoint.
- change: Added `add-cross-report`, extended `add-cross-eval` with `--checkpoints-list`, trained `artifacts/focused-variant03-long.pt` (symplectic + topk + dmd + phase + coevolution + genetic memory), and evaluated against `variant-03` and `variant-02`.
- result: Focused model ranked #1 in `artifacts/cross-eval-focused-vs-top.json` with transfer score ~0.32448 (vs ~0.29614 and ~0.29556). Generated comparison report/csv and visualization `artifacts/focused-vs-old-top-hexapod.gif`.
- next action: Re-run focused training with curriculum enabled and compare whether robustness gains persist across harsher scenario mixes.

# Artifact Interpretation Guide

This project's goal is not just to fit a single embodiment. The goal is transferable adaptation: a compact model core that keeps functioning when anonymous channels are remapped, environments shift, and embodiment changes (`hexapod`, `car`, `drone`) are introduced. `polymorph120` is a high-DOF stress embodiment used to test scaling behavior.

## What Good Looks Like

As of `2026-02-17`, using current artifacts:

- Standard profile best (`mild,gust,force`): `overall_transfer_score ~= 0.4107` (`artifacts/model-core-champion-v02.pt`)
- Hardy profile best (`gust,force,storm,blackout,crosswind`): `overall_transfer_score ~= 0.3899` (`artifacts/model-core-champion-v02.pt`)
- Hardy `car` mismatch improved from `~0.919` to `~0.713` vs prior champion, but `car` under `storm/crosswind` remains the hardest transfer case.

Practical near-term targets:

- Standard transfer score: `>= 0.410`
- Hardy transfer score: `>= 0.390`
- Hardy `car` transfer score: `>= 0.355`
- Embodiment gap (best - worst transfer): `<= 0.090`

Stretch targets:

- Standard transfer score: `>= 0.420`
- Hardy transfer score: `>= 0.400`
- Hardy `car` transfer score: `>= 0.365`

## Artifact Types and How to Read Them

1. Checkpoints and run metrics
- `*.pt`: model weights + config + run flags.
- `*.metrics.json`: epoch/generation records (`fitness`, timing, flags).
- Use these to answer: "what exact option combination produced this model?"

2. Cross-eval summaries
- `artifacts/cross-eval-*.json`
- Primary ranking fields:
  - `overall_transfer_score`: ranking score (weighted average if `--embodiment-weights` is used).
  - `overall_transfer_score_unweighted`: plain average across embodiments.
  - `overall_mean_mismatch`: lower is better.
  - `overall_mean_vitality`: higher is better.
  - `overall_recovery`: higher is better (post-remap recovery quality).
- `by_embodiment` exposes specialization vs generalization tradeoffs.

3. Cross-eval reports
- `artifacts/cross-eval-*.md` and `.csv` from `add-cross-report`.
- Fastest way to compare top checkpoints and quantify deltas vs the best.

4. Visual artifacts
- `artifacts/*compare*.gif`, `artifacts/*adaptation*.gif`
- Read these for temporal behavior:
  - remap shock amplitude,
  - recovery speed,
  - life-field stability under force/wind/light disturbances.

5. Primitive principle verification
- `tests/test_principles.py` verifies:
  - memory updates from zero state,
  - remap changes I/O behavior,
  - interconnected memory influences outputs,
  - top-k paging sparsity,
  - world controls affect stress/object motion.
- `tests/test_cross_eval.py` verifies:
  - recovery score behavior,
  - rollout metric schema,
  - hardy scenario profile resolution,
  - embodiment weighting behavior.

## Metric Semantics

`transfer_score` in cross-eval is:

- `0.55 * (1 / (1 + mean_mismatch))`
- `+ 0.30 * mean_vitality`
- `+ 0.15 * recovery`

Interpretation:

- `mean_mismatch`: anonymous control alignment error. If this rises, adaptation is failing.
- `mean_vitality`: world life-field persistence proxy. If this collapses, adaptation may be unstable in the environment.
- `recovery`: improvement after remap events. Low values indicate brittle remap response.

## Typical Current Behavior vs Desired Behavior

Typical current behavior:

- `hexapod` and `drone` transfer are consistently stronger than `car`.
- `car` in `storm` and `crosswind` dominates hardy failure cases.
- After remap, mismatch spikes and partially recovers; some cases remain elevated.
- `blackout` causes low vitality across embodiments (expected stress condition).

Behavior we want to overcome:

- Large embodiment transfer gap (`~0.10`+ between best and worst).
- Car-specific mismatch inflation under directional disturbances.
- Incomplete post-remap recovery in hardest scenarios.

## Quick Evaluation Workflow

1. Run candidate training and persist metrics/flags.
2. Evaluate standard transfer:

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/model-a.pt,artifacts/model-b.pt" --profile pi5 --embodiments "hexapod,car,drone" --scenario-profile standard --runs-per-combo 2 --steps 90 --remap-every 15 --output artifacts/cross-eval-standard.json
```

3. Evaluate hardy transfer:

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/model-a.pt,artifacts/model-b.pt" --profile pi5 --embodiments "hexapod,car,drone" --scenario-profile hardy --runs-per-combo 2 --steps 110 --remap-every 12 --output artifacts/cross-eval-hardy.json
```

4. Optional car-priority ranking:

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/model-a.pt,artifacts/model-b.pt" --profile pi5 --embodiments "hexapod,car,drone" --embodiment-weights "hexapod=1,car=2.5,drone=1" --scenario-profile hardy --runs-per-combo 2 --steps 110 --remap-every 12 --output artifacts/cross-eval-hardy-car-priority.json
```

5. Optional high-DOF stress inclusion:

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/model-a.pt,artifacts/model-b.pt" --profile pi5 --embodiments "hexapod,car,drone,polymorph120" --scenario-profile hardy --runs-per-combo 2 --steps 110 --remap-every 12 --output artifacts/cross-eval-hardy-poly4.json
```

6. Generate report tables:

```bash
~/.local/bin/uv run add-cross-report --input-path artifacts/cross-eval-hardy.json --markdown-out artifacts/cross-eval-hardy.md --csv-out artifacts/cross-eval-hardy.csv
```

7. Visualize top-2 under the same disturbance schedule:

```bash
~/.local/bin/uv run add-viz compare artifacts/top-1.pt artifacts/top-2.pt --profile pi5 --embodiment car --steps 160 --remap-every 20 --force-mode thrust --wind-x 0.4 --output artifacts/top1-vs-top2-car.gif
```

## Decision Rules After Each Experiment

- If hardy score improves but embodiment gap worsens, favor robustness regularization or weighted selection depending on deployment priority.
- If standard improves but hardy regresses, increase disturbance curriculum or remap volatility schedule.
- If only one embodiment improves, keep the checkpoint but do not promote as global best.
- Promote to "candidate champion" only when both standard and hardy scores improve or remain within noise while reducing the weakest-embodiment deficit.

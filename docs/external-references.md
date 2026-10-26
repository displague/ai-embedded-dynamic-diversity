# External Reference Repositories

This project supports external, gitignored reference repos for research calibration.

## Policy

- External repos are for analysis and benchmarking guidance.
- Do not add runtime coupling to external references unless explicitly planned.
- Keep references outside tracked source under `external_refs/`.

## Current Reference

- `https://github.com/leggedrobotics/robotic_world_model`
- Local path: `external_refs/robotic_world_model` (gitignored)

## Sync

Use the helper script:

```powershell
.\scripts\sync_external_refs.ps1
.\scripts\sync_external_refs.ps1 -PullLatest
```

The script writes `external_refs/manifest.json` with commit/branch/timestamp.

## Experiment Traceability

When using external references to guide simulator/eval design:

1. Record the external commit hash from `external_refs/manifest.json`.
2. Record the impacted internal artifacts (cross-eval/report/viz outputs).
3. Keep adaptation logic implemented inside this repo only.

## Sibling embodiment: displague/robot

- `https://github.com/displague/robot` (private at the time of writing)
- Not an external reference in the sense above: it is imported at runtime by
  `src/ai_embedded_dynamic_diversity/sim/robot_tendon.py` (opt-in, `ROBOT_REPO` or `../robot`), which
  registers the embodiment `humanoid_tendon32` from its joint table and tendon simulation.
- Trace: the robot repository logs the prompt that produced each change in `docs/prompt-log.md`; the bridge
  was created under its entry P-011 (2026-10-26).

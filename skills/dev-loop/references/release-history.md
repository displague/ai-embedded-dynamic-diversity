# Release History

## v0.2.0 (2026-04-26) — Rich Environment, DOF Coupling, GDI, JEPA

**Key changes:**
- Toroidal 40×40×20 world (`new_env_v1`): occlusion, physics objects, hazard zones
- DOF spatial coupling: `dof_spatial_map()` closes the causal loop
- GDI external population diversity metrics
- JEPA latent world predictor
- Articulation differentiation losses (io_diff, dof_coverage)
- Per-DOF named profiling in sim profiler
- IO channel activity heatmap in viz

**Champion:** `champion-new-env-v1.pt` — peak fitness +0.0506 at gen 7, 82% signal reliability at gen 40.

**Key finding:** champion-v09 (−0.097 on legacy) ranked #1 (+0.034) on new_env_v1 — the structured environment exposes previously-latent strategies.

**Issues opened:** #6–#13 (staged curriculum, diversity budget, learned DOF maps, multi-channel action field, per-embodiment GDI, hazard-aware signals, CI, per-embodiment predictor)

**PR:** #5 — squash merged; 9 Copilot review items addressed.

## v0.1.0 and earlier

See `IMPLEMENTED.md` in the repo root for full pre-release feature chronology.

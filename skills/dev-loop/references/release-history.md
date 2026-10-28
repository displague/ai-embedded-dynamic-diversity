# Release History

## v0.3.0 (2026-10-28) — Grounded bodies: PPO on the tendon humanoid and RPO, one core across both

**Key changes:**
- `humanoid_tendon32` (#33): the displague/robot tendon humanoid as the first embodiment with contact physics and a drive train that can be overloaded, registered statically.
- `rl/` (#36): PPO on fast stand-ins of grounded bodies with the champion core grown into a recurrent actor; behaviour cloning from the robot's hand-written controller; damage gates instead of fall terminations; Roboparty RPO as a second body; the tendon-model gate; cut-back for cross-eval; GIF rendering.
- Repo health: the 4 failing tests fixed and CI running pytest (#38); reproducible CUDA torch under uv (#45); dependency updates and all open security alerts (#40, #47); hyphenated storyboard scenarios (#25).

**Findings (one seed):** cloning the hand-written controller then PPO stands 90 % under pushes (grown champion 63 %, random 45 %); RPO started from the tendon policy stands 79 % from its first iterations (27 % from scratch); body training erodes cross-eval transfer (0.567 to 0.435); get-up not learned; no policy passes the tendon gate yet.

**Champion:** unchanged (v09). No body-trained core is promoted (cross-eval regression).

**Issues opened:** #48 (stand-in gap), #49 (get-up curriculum), #50 (rehearsal and regression gate), #51 (seeds).

**PRs:** #33, #36, #38, #40, #45, #47, #25 squash merged; Dependabot #24, #26, #28-#32, #41-#44 superseded.

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

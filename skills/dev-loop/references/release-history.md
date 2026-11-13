# Release History

## v0.5.0 (2026-11-13) — A coach for the get-up: a voice through two ears, and a helper's hands that fade

**Key changes:**
- `rl/coach.py` (#67; closes #66): `Voice` (a coach's words as syllables over 8 frequency bands, heard at two ears on
  the robot's stand-in: spreading, head shadow, ambient and own-drive noise, the time difference, an echo of the
  last second; 25 observation channels) and `Hands` (a lift on the trunk that takes only what the soles leave, a
  level per start that fades, episodes never helped, a helper that may quit part-way).
- `getup_eval --guide/--wait/--mute`, the unassisted test; `render` shows what the coach says.
- `skills/grounded-rl`: how to train and judge on grounded bodies.

**Findings:** three seeds: the early-fade helper is withdrawn in training every time and leaves 46 % unassisted
rise against 52 % with no support. One seed: a late fade leaves 22 % (dependence: 53 % with the hands); spoken cues
are unused (38 %, 42 % muted); a spoken command is not obeyed (0 %, staged or not).

**Champion:** unchanged (v09). Nothing here beats training alone on the half-kneel.

**Issues opened:** #68 (obey a spoken command), #69 (the helper on fallen starts), #70 (the tendon gate with ears).

**PRs:** #67, #71.

## v0.4.0 (2026-11-04) — A plain baseline, rests and locks, and the tendon gate as the target

**Key changes:**
- Plain baselines `--arch mlp|gru` and a five-seed study (#58, #59; closes #51).
- Rests and locks (#58, #56): the drive's self-locking drums and electrical power in the stand-in, a power cost,
  held loads and quiet scenes; `tendon_env` + `rl.cli finetune` (PPO on the full tendon model); the gate's power,
  locked fraction and rest scenes.
- Per-joint action ranges from the teacher, wind range (#60); ankles in torque mode (#62); the hybrid gate and
  fine-tune (#63); get-up as a curriculum from a half-kneel (#61) and `rl.getup_eval`; review fixes (#64).

**Findings (five seeds):** the plain MLP beats the grown core on the tendon gate (22 against 17 of 30 named scenes
cloned, 18 against 0 uncloned); a GRU matches the MLP; the core's head start on RPO holds on every seed. A held
load in training cuts median idle power 175 to 38 W (best 10.6 W, hand-written 5.4 W). Only the hybrid holds the
8 m/s wind (2 of 5 plain seeds pass all six named scenes). Get-up rises 61 % from a half-kneel, 0 % from a fall.

**Champion:** unchanged (v09). Nothing body-trained is promoted; the robot uses the plain network (robot D-052).

**PRs:** #55, #54 (before the tag, after v0.3.0), #58-#64.

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

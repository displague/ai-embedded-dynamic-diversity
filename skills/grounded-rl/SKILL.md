---
name: grounded-rl
description: Train and judge policies by reward on bodies with contact physics (rl/): the tendon humanoid's stand-in, the tendon gate, the get-up stages, and a coach's voice and helping hands.
---

# Grounded RL Skill

## Use This Skill When

1. Training or evaluating anything under `src/ai_embedded_dynamic_diversity/rl/`.
2. Changing the stand-in's observation, reward, curriculum or supports (`rl/standin_env.py`, `rl/coach.py`).
3. Reading a result that will be quoted in `docs/rl-standin.md`, the README or a release.

## Setup

- The body comes from a displague/robot checkout: `export ROBOT_REPO=../robot` (its `sim/robot_standin.xml` and
  `.json`). The stand-in is a contract: joints and their order are fixed; sensors and spec keys may be added there.
- On Windows use the main checkout's venv from any worktree:
  `PYTHONPATH=src ROBOT_REPO=../robot ../ai-embedded-dynamic-diversity/.venv/Scripts/python.exe -m ...`
- A stand-in run uses every core for physics. Two runs side by side each keep most of their speed; three do not.
- Policies and logs go under `artifacts/rl/<study>/` (local, not committed).

## The loop

```bash
python -m ai_embedded_dynamic_diversity.rl.cli train --task getup --getup-start curriculum --arch mlp --iterations 800 --label <name>
python -m ai_embedded_dynamic_diversity.rl.getup_eval artifacts/rl/<name>.pt          # 64 starts, mean action, no helper
python -m ai_embedded_dynamic_diversity.rl.tendon_eval artifacts/rl/<stand policy>.pt  # the gate on the full tendon model
pytest tests/test_rl_*.py                                                              # seconds; needs ROBOT_REPO
```

- 800 iterations of 2,048 environments are 39 M steps, about 21 minutes alone.
- The plain MLP (`--arch mlp`) is the robot's baseline (it beats the grown core on the tendon gate); use it unless
  the question is about the core.

## Supports for the get-up stages (rl/coach.py, issue #66)

| Flag | What it adds |
| --- | --- |
| `--coach voice` | A coach's words heard through two ears: 25 observation channels (8 bands per ear, the time difference, an 8-band echo of the last second) |
| `--coach-guide 0.5` | The share of episodes with cues after the command; the rest hear the command repeated |
| `--coach-wait 0.5,12 --coach-wait-after 0.3` | "Stay" for that long before the command (beyond the episode: never told); the wait begins once the first start's success passes 0.3 |
| `--help-cap 0.6` | A helper lifts the trunk by up to that share of the weight, with a level per start that fades |
| `--help-free 0.5 --help-quit 0.2 --help-fade-at 0.35` | The share never helped, the share the helper walks away from, the helped success at which the level steps down |

`getup_eval` never uses the hands. `--guide 0` leaves the command only, `--wait 0.5,12` adds the wait, `--mute`
keeps the coach's script but silences it (the control for "does the policy listen?").

## Rules learned the hard way

1. **Judge on the unassisted test, never on training success.** With the helper, helped episodes rose 53 % while
   the same policy rose 22 % alone (`hands-s0`).
2. **A support that is not withdrawn early becomes the skill.** The helper stayed at full strength for 610 of 800
   iterations because its fade waited for 60 % helped success. Fade early, keep many episodes unhelped, and watch
   `free` next to `helped` in the log.
3. **One level per start.** When fallen starts unlocked, their failures brought the help back on the half-kneel
   until each start had its own level.
4. **A new demand goes in after the skill, not with it, and even then check it was learned.** Asked to wait for
   the command from the first step, the policy learned never to rise (0 of 64). Staged (`--coach-wait-after`), it
   rose first and then unlearned rising: still 0. Obeying a spoken command is open; do not claim it.
5. **Early fade is the setting that worked:** `--help-cap 0.6 --help-free 0.5 --help-fade-at 0.35` (47 % unassisted,
   11 of 64 ended by an overloaded line, against 48 % and 34 with no support).
6. **A memoryless policy needs the past in its inputs.** Words last 0.3 s and the coach pauses; the echo channels
   carry what was said through the silence. That was not enough here: try `--arch gru`.
7. **Extra inputs are not free.** 17 hearing channels with nothing to say cost rise rate at equal steps (38 %
   against 48 %, one seed).
8. **One seed is an anecdote.** The half-kneel's two seeds differ by 6 points at 1,500 iterations. Quote a
   difference only when it is larger than that, and say how many seeds.
9. **Changing the observation orphans old policies.** A policy saved with 17 hearing channels does not load into a
   25-channel environment; re-train, and record which code a number came from.

## Reference Loading (On Demand)

- `docs/rl-standin.md`: method, every published table.
- displague/robot `docs/32`, `docs/46`, `docs/48`, `docs/49`: the same skills on the full tendon model.

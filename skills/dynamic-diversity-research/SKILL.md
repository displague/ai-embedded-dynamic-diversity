---
name: dynamic-diversity-research
description: Continue and extend the AI Embedded Dynamic Diversity project using its PyTorch core, world simulator, training loop, and export pipeline. Use when asked to add experiments, improve robustness/efficiency, adapt anonymous I/O remapping, prepare Raspberry Pi 5 deployment, or maintain autonomous research workflows for this repository.
---

# Dynamic Diversity Research Skill

Follow this workflow.

1. Read `README.md` and map user request to one or more components:
- `models/core.py`
- `sim/world.py`
- `sim/embodiments.py`
- `sim/viz_cli.py`
- `train/cli.py`
- `deploy/cli.py`
- `deploy/bench_cli.py`
- `deploy/hil_cli.py`

2. Preserve these core invariants unless user explicitly requests a change:
- Keep sensor/actuator channels anonymous (no modality-specific hardcoding in the model core).
- Keep both passive interpretation and active memory pathways.
- Keep remap adaptation capability for channel reassignment.
- Keep efficiency and robustness in objective tradeoffs.

3. For model changes, update all impacted places in one pass:
- `config.py` dataclasses
- model forward signature and outputs
- loss terms in `train/losses.py`
- export inputs in `deploy/cli.py`

4. For simulator changes:
- Maintain 3D state evolution over time.
- Model at least one resource constraint and one stress/pressure field.
- Keep observation encoding anonymous and fixed-size (`signal_dim`).

5. For Raspberry Pi 5 deployment work:
- Prioritize smaller hidden dimensions, reduced memory slots, and stable ops supported by TorchScript/ONNX.
- Prefer deterministic paths and avoid exotic custom ops.
- Add or update a benchmark script before claiming improvements.

6. Validate each non-trivial change:
- Run a short train smoke test (1-2 epochs).
- Run simulator rollout.
- Run visualization smoke (short GIF generation) if remap/adaptation behavior changed.
- Run at least one export command (TorchScript preferred first).
- Run `add-bench` for edge-profile changes.

7. Record experiment intent and outcomes in `references/experiment-log.md` as compact entries:
- date
- hypothesis
- change
- result
- next action

Load `references/research-backlog.md` when user asks for roadmap prioritization.
Update root `TODO.md` for goals and root `IMPLEMENTED.md` for delivered work and lessons learned.

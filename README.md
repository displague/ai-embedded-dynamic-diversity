# AI Embedded Dynamic Diversity

PyTorch research scaffold for a portable "model core" that adapts to anonymous long-lived I/O channels across changing embodiments (robotic, vehicular, biologic).

## Research intent

- Learn passive tensor fields that continuously interpret anonymous multi-modal overlays into edge-node readiness.
- Learn active memory tensors that capture reusable latent/genetic memory.
- Train in a 3D+time constrained world (Conway-inspired dynamics + resources + stress) for robust efficiency.
- Preserve adaptation under remapped channels (e.g., actuator/function reassignment).

## Embodiments (DOF)

- `hexapod`: `10` DOF controls, `5` sensor channels
- `car`: `6` DOF controls, `5` sensor channels
- `drone`: `8` DOF controls, `5` sensor channels
- `polymorph120`: `120` DOF controls, `12` sensor channels (complex stress-test; multiple of 10/6/8)

List from CLI:

```bash
~/.local/bin/uv run add-sim embodiments
```

## Environment

- Training target: NVIDIA RTX 5080 16GB
- Deployment target: Raspberry Pi 5
- Python package manager: `uv` at `~/.local/bin/uv`

## Setup

```bash
~/.local/bin/uv sync
```

Windows/PowerShell fallback:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

CUDA wheel setup for this project venv (PowerShell):

```powershell
~/.local/bin/uv pip install --python .venv\Scripts\python.exe --torch-backend cu130 --reinstall torch torchvision torchaudio
@'
import torch
print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)
'@ | .\.venv\Scripts\python -
```

If `uv run` re-syncs CPU torch from lockfile, run training/eval via `.\.venv\Scripts\python -m ...` (or `uv run --no-sync` if your environment is already correct). Training now defaults to strict device selection, so `--device cuda` fails fast if CUDA is unavailable.

## Run training

```bash
~/.local/bin/uv run add-train --epochs 20 --batch-size 16 --unroll-steps 12 --device cuda
```

Pi-profile training from laptop (smaller deploy-aligned core):

```bash
~/.local/bin/uv run add-train --profile pi5 --epochs 30 --batch-size 24 --unroll-steps 16 --device cuda --save-path artifacts/model-core-pi5.pt
```

Co-evolution mode:

```bash
~/.local/bin/uv run add-train --coevolution --population-size 6 --elite-fraction 0.5 --mutation-std 0.01 --epochs 12 --profile pi5 --device cuda --save-path artifacts/model-core-coevo.pt
```

Parallel diverse-fitness sweep:

```bash
~/.local/bin/uv run add-train-parallel --variants 6 --max-workers 3 --profile pi5 --epochs 10 --device cuda --out-dir artifacts/parallel
```

Aggressive transfer-focused sweep (embodiment-aware loss + mixed device pool):

```bash
~/.local/bin/uv run add-train-parallel --variants 8 --max-workers 4 --profile pi5 --epochs 30 --batch-size 20 --unroll-steps 16 --device cuda --device-pool "cuda,cuda,cpu,cpu" --coevolution --population-size 6 --embodiments "hexapod,car,drone,polymorph120" --enable-embodiment-transfer-loss --transfer-loss-weight 0.4 --transfer-fitness-weight 0.1 --transfer-samples-per-step 3 --use-amp --allow-tf32 --out-dir artifacts/parallel-aggressive
```

Long CUDA-heavy warm-start sweep (recommended to run via project venv Python to avoid lockfile resync):

```powershell
.\.venv\Scripts\python -m ai_embedded_dynamic_diversity.train.parallel_cli --variants 8 --max-workers 3 --profile pi5 --epochs 48 --batch-size 24 --unroll-steps 20 --device cuda --device-pool "cuda,cuda,cpu" --coevolution --population-size 6 --embodiments "hexapod,car,drone,polymorph120" --enable-embodiment-transfer-loss --transfer-loss-weight 0.5 --transfer-fitness-weight 0.16 --transfer-samples-per-step 4 --use-amp --allow-tf32 --init-weights artifacts/parallel-cuda-mixed-sweep/variant-01.pt --out-dir artifacts/parallel-cuda-long-v02
```

Convergence-focused follow-up sweep (current champion path):

```powershell
.\.venv\Scripts\python -m ai_embedded_dynamic_diversity.train.parallel_cli --variants 10 --max-workers 3 --profile pi5 --epochs 56 --batch-size 24 --unroll-steps 20 --device cuda --device-pool "cuda,cuda,cpu" --coevolution --population-size 6 --embodiments "hexapod,car,drone,polymorph120" --enable-embodiment-transfer-loss --transfer-loss-weight 0.55 --transfer-fitness-weight 0.18 --transfer-samples-per-step 4 --use-amp --allow-tf32 --init-weights artifacts/model-core-champion-v02.pt --out-dir artifacts/parallel-cuda-converge-v03
```

Focused warm-start fine-tune from current champion:

```bash
~/.local/bin/uv run add-train --profile pi5 --epochs 36 --batch-size 24 --unroll-steps 18 --device cuda --gating-mode symplectic --topk-gating 4 --enable-dmd-gating --enable-phase-gating --enable-curriculum --enable-genetic-memory --embodiments "hexapod,car,drone,polymorph120" --enable-embodiment-transfer-loss --transfer-loss-weight 0.45 --transfer-fitness-weight 0.12 --transfer-samples-per-step 3 --init-weights artifacts/focused-variant03-long.pt --save-path artifacts/focused-variant03-long-poly-ft.pt
```

Latest champion checkpoint: `artifacts/model-core-champion-v03.pt` (from `artifacts/parallel-cuda-converge-v03/variant-07.pt`).
Latest capability sweep candidates (pending high-repeat promotion):
- transfer-only hardy poly4 best: `artifacts/parallel-cuda-capability-v01/variant-01.pt`
- transfer+capability hardy poly4 best: `artifacts/parallel-cuda-capability-v01/variant-00.pt`

Curriculum + latent genetic memory:

```bash
~/.local/bin/uv run add-train --profile pi5 --epochs 40 --device cuda --remap-probability-start 0.08 --remap-probability-end 0.45 --env-volatility-start 0.05 --env-volatility-end 0.7 --genetic-memory-decay 0.92 --save-path artifacts/model-core-curriculum.pt
```

Experimental flags are optional and combinable: `--gating-mode symplectic`, `--topk-gating 4`, `--enable-dmd-gating`, `--enable-phase-gating`, `--enable-curriculum`, `--enable-genetic-memory`.
Each run stores flags+performance+fitness in `*.metrics.json` (or set `--metrics-path`).
Use `--strict-device` (default) to prevent accidental CPU fallback when CUDA was requested.

Embodiment-aware transfer optimization (optional):

- `--embodiments "hexapod,car,drone,polymorph120"`
- `--enable-embodiment-transfer-loss`
- `--transfer-loss-weight 0.45`
- `--transfer-fitness-weight 0.12`
- `--transfer-samples-per-step 3`
- `--init-weights artifacts/focused-variant03-long.pt` for warm-start fine-tuning

## Run simulator rollout

```bash
~/.local/bin/uv run add-sim --steps 10 --batch-size 2 --device cpu
```

Visualization is independent of training loop execution and can use any checkpoint:

```bash
~/.local/bin/uv run add-viz run --weights artifacts/model-core-coevo.pt --profile pi5 --embodiment hexapod --steps 160 --remap-every 20 --force-mode push --wind-x 0.4 --light-intensity 0.8 --output artifacts/adaptation.gif
~/.local/bin/uv run add-viz compare artifacts/model-core-pi5.pt artifacts/model-core-coevo.pt --profile pi5 --embodiment drone --steps 160 --remap-every 20 --force-mode thrust --output artifacts/adaptation-compare.gif
```

`force-mode` options: `none`, `poke`, `press`, `push`, `continuous-blow`, `thrust`, `move`.

Memory gating benchmark:

```bash
~/.local/bin/uv run add-gating-bench --profile pi5 --device cpu --output artifacts/gating-bench.json
```

Cross-embodiment transfer evaluation (`hexapod`, `car`, `drone`):

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-dir artifacts/parallel-long --profile pi5 --embodiments 'hexapod,car,drone' --scenarios 'mild,gust,force' --runs-per-combo 2 --steps 90 --remap-every 15 --output artifacts/cross-eval-summary.json
```

Hardy-line transfer evaluation profile (stronger disturbances):

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/focused-variant03-curriculum.pt,artifacts/focused-variant03-long.pt,artifacts/parallel-long/variant-03.pt,artifacts/parallel-long/variant-02.pt" --profile pi5 --embodiments "hexapod,car,drone" --scenario-profile hardy --runs-per-combo 2 --steps 110 --remap-every 12 --output artifacts/cross-eval-hardy-focused-vs-top.json
~/.local/bin/uv run add-cross-report --input-path artifacts/cross-eval-hardy-focused-vs-top.json --markdown-out artifacts/cross-eval-hardy-focused-vs-top.md --csv-out artifacts/cross-eval-hardy-focused-vs-top.csv
```

Car-priority ranking for hardy-line selection (optional weighting):

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/focused-variant03-long.pt,artifacts/focused-variant03-curriculum.pt,artifacts/parallel-long/variant-03.pt" --profile pi5 --embodiments "hexapod,car,drone" --embodiment-weights "hexapod=1,car=2.5,drone=1" --scenario-profile hardy --runs-per-combo 2 --steps 110 --remap-every 12 --output artifacts/cross-eval-hardy-car-priority.json
```

Capability-aware cross-eval (biological/technological proxy harness):

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list "artifacts/model-core-champion-v03.pt,artifacts/parallel-cuda-capability-v01/variant-01.pt,artifacts/parallel-cuda-capability-v01/variant-00.pt" --profile pi5 --embodiments "hexapod,car,drone,polymorph120" --scenario-profile hardy --runs-per-combo 3 --steps 110 --remap-every 12 --capability-profile bio-tech-v1 --capability-score-weight 0.25 --output artifacts/cross-eval-capability-vs-champion.json
~/.local/bin/uv run add-cross-report --input-path artifacts/cross-eval-capability-vs-champion.json --markdown-out artifacts/cross-eval-capability-vs-champion.md --csv-out artifacts/cross-eval-capability-vs-champion.csv
```

Focused comparison over an explicit checkpoint list:

```bash
~/.local/bin/uv run add-cross-eval --checkpoints-list \"artifacts/focused-variant03-long.pt,artifacts/parallel-long/variant-03.pt,artifacts/parallel-long/variant-02.pt\" --profile pi5 --embodiments 'hexapod,car,drone' --scenarios 'mild,gust,force' --runs-per-combo 2 --steps 90 --remap-every 15 --output artifacts/cross-eval-focused-vs-top.json
~/.local/bin/uv run add-cross-report --input-path artifacts/cross-eval-focused-vs-top.json --markdown-out artifacts/cross-eval-focused-vs-top.md --csv-out artifacts/cross-eval-focused-vs-top.csv
```

## Export for edge

```bash
~/.local/bin/uv run add-export torchscript --weights artifacts/model-core.pt --output artifacts/model-core.ts
~/.local/bin/uv run add-export onnx --weights artifacts/model-core.pt --output artifacts/model-core.onnx --opset 17
~/.local/bin/uv run add-export quantized-torchscript --weights artifacts/model-core-pi5.pt --output artifacts/model-core-pi5-int8.ts
~/.local/bin/uv run add-bench --weights artifacts/model-core-pi5.pt --profile pi5 --device cpu --steps 300
~/.local/bin/uv run add-hil udp-bridge --weights artifacts/model-core-pi5.pt --listen-port 45454 --send-port 45455
```

## Project layout

- `src/ai_embedded_dynamic_diversity/models/core.py`: passive/active tensor core and anonymous edge router.
- `src/ai_embedded_dynamic_diversity/sim/world.py`: 3D+time world with life/resource/stress fields.
- `src/ai_embedded_dynamic_diversity/sim/embodiments.py`: reproducible embodiment schemas and mapping generation.
- `src/ai_embedded_dynamic_diversity/train/cli.py`: unrolled training with remap perturbations.
- `src/ai_embedded_dynamic_diversity/train/cross_eval_cli.py`: cross-embodiment transfer evaluation and ranking.
- `src/ai_embedded_dynamic_diversity/train/cross_report_cli.py`: delta/summary report generation for cross-eval outputs.
- `src/ai_embedded_dynamic_diversity/deploy/cli.py`: TorchScript and ONNX export path.
- `src/ai_embedded_dynamic_diversity/sim/viz_cli.py`: remap adaptation visualization (GIF/MP4 output).
- `src/ai_embedded_dynamic_diversity/deploy/bench_cli.py`: inference latency benchmarking for edge validation.
- `src/ai_embedded_dynamic_diversity/deploy/hil_cli.py`: UDP hardware-in-the-loop adapter.
- `skills/dynamic-diversity-research/SKILL.md`: workflow for future AI agents continuing research.
- `docs/ARTIFACTS.md`: artifact interpretation guide (goals, success criteria, expected behavior, failure patterns).

## Immediate next experiments

1. Add real sensor/actuator drivers behind the UDP HIL adapter.
2. Implement quantization-aware training and compare with dynamic quantization.
3. Add policy-level tasks on top of the current core adaptation objective.

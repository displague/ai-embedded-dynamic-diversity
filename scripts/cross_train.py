"""Cross-training between the cellular world and the grounded bodies (MuJoCo), and scoring what comes back.

    python scripts/cross_train.py evolve --seeds artifacts/model-core-champion-v09.pt --seed 1 --out artifacts/cross/evo-a-s1.pt
    python scripts/cross_train.py evolve --seeds artifacts/model-core-champion-v09.pt,artifacts/rl/cut-stand-grown.pt --seed 1 --out ...
    python scripts/cross_train.py evolve --eyes --seeds ... --out ...          # the world's eyes on (sim/world.py)
    python scripts/cross_train.py score artifacts/model-core-champion-v09.pt artifacts/cross/*.pt --out artifacts/cross/score.json

`evolve` is scripts/eval_evolve_new_env.py's phase 2 (coevolution in new_env_v1, 40 epochs, population 6) from
any seed checkpoints: champions, or cores a body trained in MuJoCo and cut back (rl/cut_back.py).
`score` is its phase 1 over several evaluation seeds: fitness on the legacy world and on new_env_v1, each with the
world's eyes off and on. The fitness target is the first observation slots, which the eyes do not touch, so the
four columns are comparable for one checkpoint and between checkpoints.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

EVOLVE_ARGS = [
    "--batch-size=12", "--unroll-steps=16", "--lr=1.5e-4", "--device=cuda", "--profile=pi5",
    "--world-profile=new_env_v1", "--coevolution", "--population-size=6", "--elite-fraction=0.4",
    "--mutation-std=0.008", "--enable-curriculum", "--curriculum-power=1.2", "--remap-probability-start=0.12",
    "--remap-probability-end=0.45", "--env-volatility-start=0.08", "--env-volatility-end=0.65",
    "--enable-embodiment-transfer-loss", "--embodiments=hexapod,car,drone,polymorph120",
    "--transfer-loss-weight=0.40", "--transfer-fitness-weight=0.12", "--transfer-samples-per-step=3",
    "--enable-autopoietic-objective", "--autopoietic-loss-weight=0.14", "--autopoietic-fitness-gain=0.16",
    "--enable-capability-guardrail", "--signal-reliability-floor=0.50", "--conjoining-gain-floor=0.25",
    "--capability-guardrail-penalty-weight=0.20", "--noise-profile=dropout-quant-v2", "--enable-noise-curriculum",
    "--noise-strength-start=0.2", "--noise-strength-end=0.9", "--force-curriculum-mode=continuous-blow",
    "--force-curriculum-strength-start=0.1", "--force-curriculum-strength-end=0.8", "--enable-world-predictor",
    "--world-pred-loss-weight=0.04", "--diversity-selection-bonus=0.15", "--enable-multi-scale-gating", "--use-amp",
    "--allow-tf32",
]


def evolve(seeds: str, out: str, seed: int, epochs: int, eyes: bool) -> None:
    os.environ["ADD_WORLD_EYES"] = "1" if eyes else "0"
    import typer
    from typer.testing import CliRunner

    from ai_embedded_dynamic_diversity.train.cli import run
    app = typer.Typer()
    app.command()(run)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    res = CliRunner().invoke(app, EVOLVE_ARGS + ["--epochs=%d" % epochs, "--init-weights-cycle=" + seeds,
                                                 "--seed=%d" % seed, "--save-path=" + out,
                                                 "--metrics-path=" + out.replace(".pt", ".metrics.json")])
    Path(out.replace(".pt", ".log")).write_text(res.output, encoding="utf-8")
    if res.exception:
        raise res.exception
    ck = torch.load(out, map_location="cpu", weights_only=False)
    print(json.dumps(dict(out=out, seeds=seeds.split(","), seed=seed, eyes=eyes, epochs=epochs,
                          training_fitness=ck.get("fitness"))))


def score(paths: list[str], out: str, evals: int, no_eyes: bool = False, seed0: int = 1000) -> None:
    from eval_evolve_new_env import EVAL_BATCH, EVAL_ENV_VOLATILITY, EVAL_STEPS

    from ai_embedded_dynamic_diversity.config import ModelConfig, WorldConfig, world_config_for_profile
    from ai_embedded_dynamic_diversity.models import ModelCore
    from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
    from ai_embedded_dynamic_diversity.train.cli import _build_transfer_states, evaluate_fitness
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wcfgs = {"legacy": WorldConfig(x=20, y=20, z=10), "new_env_v1": world_config_for_profile("new_env_v1")}

    def world(w, eyes):
        os.environ["ADD_WORLD_EYES"] = "1" if eyes else "0"
        keys = ("actuation_delay_steps", "actuation_noise_std", "sensor_latency_steps", "sensor_dropout_burst_prob",
                "surface_friction_scale", "disturbance_correlation_horizon", "num_occlusion_objects",
                "occlusion_seed", "num_physics_objects", "phys_mass", "phys_friction", "hazard_zones")
        return SignalingWorld(w.x, w.y, w.z, w.resource_channels, w.decay, device=str(dev),
                              **{k: getattr(w, k) for k in keys})

    rows = []
    for path in paths:
        ck = torch.load(path, map_location=dev, weights_only=False)
        mcfg = ModelConfig(**ck["model_config"])
        model = ModelCore(**asdict(mcfg)).to(dev)
        model.load_state_dict(ck["model"])
        states = _build_transfer_states(["hexapod", "car", "drone", "polymorph120"], mcfg, dev, seed=9001)
        row = dict(checkpoint=path)
        for name, wcfg in wcfgs.items():
            for eyes in ((False,) if no_eyes else (False, True)):
                vals = []
                for k in range(evals):
                    torch.manual_seed(seed0 + k)
                    vals.append(evaluate_fitness(
                        model, world(wcfg, eyes), mcfg, wcfg, dev, steps=EVAL_STEPS, batch=EVAL_BATCH,
                        env_volatility=EVAL_ENV_VOLATILITY, transfer_states=states, transfer_fitness_weight=0.10,
                        transfer_samples_per_step=2, noise_profile="dropout-quant-v2", noise_strength=0.8,
                        noise_seed=seed0 + 11345 + k, autopoietic_fitness_weight=0.15, genetic_memory_persistence_weight=0.05,
                        force_curriculum_mode="continuous-blow", force_curriculum_strength=0.6))
                t = torch.tensor(vals)
                row["%s%s" % (name, "+eyes" if eyes else "")] = dict(mean=float(t.mean()), sd=float(t.std()), vals=vals)
        rows.append(row)
        print("%-52s " % os.path.basename(path) + "  ".join(
            "%s %+.4f (%.4f)" % (k, v["mean"], v["sd"]) for k, v in row.items() if k != "checkpoint"), flush=True)
    # every checkpoint saw the same worlds, controls and noise, so differences from the first are paired
    print("paired difference from %s (mean, standard error over %d evaluations):" % (os.path.basename(paths[0]), evals))
    for row in rows[1:]:
        cells = []
        for k, v in row.items():
            if k == "checkpoint":
                continue
            d = torch.tensor(v["vals"]) - torch.tensor(rows[0][k]["vals"])
            v["paired"] = dict(mean=float(d.mean()), se=float(d.std() / max(1, len(d)) ** 0.5))
            cells.append("%s %+.4f (%.4f)" % (k, v["paired"]["mean"], v["paired"]["se"]))
        print("%-52s " % os.path.basename(row["checkpoint"]) + "  ".join(cells), flush=True)
    if out:
        Path(out).write_text(json.dumps(dict(evals=evals, seed0=seed0, rows=rows), indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evolve")
    e.add_argument("--seeds", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--epochs", type=int, default=40)
    e.add_argument("--eyes", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("paths", nargs="+")
    s.add_argument("--out", default="")
    s.add_argument("--evals", type=int, default=8)
    s.add_argument("--seed0", type=int, default=1000, help="the first evaluation seed: change it for a fresh set")
    s.add_argument("--no-eyes", action="store_true", help="skip the columns with the world's eyes on")
    a = ap.parse_args()
    if a.cmd == "evolve":
        evolve(a.seeds, a.out, a.seed, a.epochs, a.eyes)
    else:
        score([p for x in a.paths for p in (sorted(glob.glob(x)) or [x])], a.out, a.evals, a.no_eyes, a.seed0)

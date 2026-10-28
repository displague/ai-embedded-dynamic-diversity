"""PPO on the tendon humanoid's stand-in, with the model core as the actor.

    python -m ai_embedded_dynamic_diversity.rl.cli train --task getup --init artifacts/model-core-champion-v09.pt
    python -m ai_embedded_dynamic_diversity.rl.cli compare --task getup --checkpoint artifacts/model-core-champion-v09.pt

`compare` trains the same task twice with the same seed, once from the grown checkpoint and once from random
weights of the same architecture, and writes a report (docs/rl-standin.md explains the measures).
"""
from __future__ import annotations

import json
import os

import typer

from ai_embedded_dynamic_diversity.rl.ppo import PPOConfig, train as ppo_train
from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig

app = typer.Typer(add_completion=False)


def _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches, body="tendon32", pad_dofs=32):
    return (EnvConfig(task=task, num_envs=num_envs, seed=seed, body=body, pad_dofs=pad_dofs),
            PPOConfig(iterations=iterations, seed=seed, device=device, horizon=horizon, epochs=epochs,
                      minibatches=minibatches))


@app.command()
def train(task: str = "getup", init: str = "", body: str = "tendon32", pad_dofs: int = 32, teacher: str = "",
          num_envs: int = 2048, iterations: int = 300, seed: int = 0, device: str = "cuda", horizon: int = 24,
          epochs: int = 4, minibatches: int = 2, out: str = "artifacts/rl", label: str = ""):
    """`init`: a champion checkpoint to grow, or a policy from this trainer (any body, same padding) to start
    from; `teacher`: demonstrations to clone first (displague/robot tools/record_teacher.py)."""
    env_cfg, ppo = _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches, body, pad_dofs)
    label = label or "%s-%s-%s-s%d" % (body, task, "init" if init else "random", seed)
    s = ppo_train(env_cfg, ppo, init or None, out, label, teacher=teacher or None)
    print(json.dumps(s, indent=1, default=str))


def _first(log, key, threshold):
    for r in log:
        v = r.get(key)
        if v is not None and v >= threshold:
            return r["steps"]
    return None


def _tail_mean(log, key, n=10):
    vals = [r[key] for r in log[-n:] if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


@app.command()
def compare(task: str = "getup", checkpoint: str = "artifacts/model-core-champion-v09.pt", num_envs: int = 2048,
            iterations: int = 300, seed: int = 0, device: str = "cuda", horizon: int = 24, epochs: int = 4,
            minibatches: int = 2, out: str = "artifacts/rl"):
    """Same task, same seed: grown from `checkpoint` versus random weights. Writes <out>/compare-<task>.md."""
    rows = []
    for init, tag in ((checkpoint, "grown"), (None, "random")):
        env_cfg, ppo = _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches)
        label = "%s-%s-s%d" % (task, tag, seed)
        ppo_train(env_cfg, ppo, init, out, label)
        with open(os.path.join(out, label + ".log.json")) as f:
            log = json.load(f)["log"]
        rows.append(dict(tag=tag, label=label, init=init or "random", steps=log[-1]["steps"],
                         wall_s=log[-1]["wall_s"],
                         first_success_10=_first(log, "success_rate", 0.10),
                         first_success_50=_first(log, "success_rate", 0.50),
                         final_success=_tail_mean(log, "success_rate"),
                         final_damage=_tail_mean(log, "damage_rate"),
                         final_standing=_tail_mean(log, "standing_fraction"),
                         final_reward=_tail_mean(log, "reward")))
    path = os.path.join(out, "compare-%s-s%d.md" % (task, seed))
    with open(path, "w") as f:
        f.write("# Grown champion versus random weights: %s (seed %d)\n\n" % (task, seed))
        f.write("Checkpoint: `%s`. %d environments, %d iterations of %d steps.\n\n" % (checkpoint, num_envs,
                                                                                       iterations, horizon))
        f.write("| Start | Steps | Wall s | Steps to 10 %% success | Steps to 50 %% success | Success (last 10 it) | "
                "Damage (last 10) | Standing fraction (last 10) | Reward (last 10) |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        fmt = lambda v, p="%.3f": "-" if v is None else (p % v)
        for r in rows:
            f.write("| %s | %d | %.0f | %s | %s | %s | %s | %s | %s |\n" % (
                r["tag"], r["steps"], r["wall_s"], fmt(r["first_success_10"], "%d"), fmt(r["first_success_50"], "%d"),
                fmt(r["final_success"]), fmt(r["final_damage"]), fmt(r["final_standing"]), fmt(r["final_reward"])))
    with open(path.replace(".md", ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(open(path).read())


if __name__ == "__main__":
    app()

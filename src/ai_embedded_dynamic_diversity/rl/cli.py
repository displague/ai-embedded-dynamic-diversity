"""PPO on the tendon humanoid's stand-in, with the model core as the actor.

    python -m ai_embedded_dynamic_diversity.rl.cli train --task getup --init artifacts/model-core-champion-v09.pt
    python -m ai_embedded_dynamic_diversity.rl.cli compare --task getup --checkpoint artifacts/model-core-champion-v09.pt
    python -m ai_embedded_dynamic_diversity.rl.cli train --task stand --wind-mean 0,8 --wind-gust 0,2

`compare` trains the same task twice with the same seed, once from the grown checkpoint and once from random
weights of the same architecture, and writes a report (docs/rl-standin.md explains the measures).
"""
from __future__ import annotations

import json
import os

import typer

from ai_embedded_dynamic_diversity.rl.ppo import PPOConfig, train as ppo_train
from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, teacher_action_half

app = typer.Typer(add_completion=False)


def _range(text: str) -> tuple:
    """'lo,hi' (or one value for both) in m/s."""
    if not text:
        return (0.0, 0.0)
    v = [float(x) for x in text.split(",")]
    if len(v) == 1:
        v = v * 2
    if len(v) != 2 or v[0] < 0 or v[1] < v[0]:
        raise typer.BadParameter("expected lo,hi with 0 <= lo <= hi, got %r" % text)
    return tuple(v)


def _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches, body="tendon32", pad_dofs=32,
          wind_mean="", wind_gust="", wind_tau=2.0, arch="core", locks=False, power_cost=0.0, quiet=0.0,
          hand_load="", episode_s=10.0, init_log_std=-2.0, quiet_noise=1.0, action_half=None):
    return (EnvConfig(task=task, num_envs=num_envs, seed=seed, body=body, pad_dofs=pad_dofs,
                      wind_mean=_range(wind_mean), wind_gust=_range(wind_gust), wind_tau=wind_tau,
                      locks=locks, power_cost=power_cost, quiet_fraction=quiet, hand_load=_range(hand_load),
                      episode_s=episode_s, quiet_noise=quiet_noise, action_half=action_half),
            PPOConfig(iterations=iterations, seed=seed, device=device, horizon=horizon, epochs=epochs,
                      minibatches=minibatches, arch=arch, init_log_std=init_log_std))


@app.command()
def train(task: str = "getup", init: str = "", body: str = "tendon32", pad_dofs: int = 32, teacher: str = "",
          num_envs: int = 2048, iterations: int = 300, seed: int = 0, device: str = "cuda", horizon: int = 24,
          epochs: int = 4, minibatches: int = 2, out: str = "artifacts/rl", label: str = "",
          wind_mean: str = "", wind_gust: str = "", wind_tau: float = 2.0, arch: str = "core",
          locks: bool = False, power_cost: float = 0.0, quiet: float = 0.0, hand_load: str = "",
          episode_s: float = 10.0, init_log_std: float = -2.0, quiet_noise: float = 1.0,
          teacher_range: bool = False):
    """`init_log_std`: the exploration noise's starting log std (rests: small enough that drums can lock, -4.5);
    `arch`: "core" (the model core) or "mlp" (the plain baseline, random weights only); `init`: a champion checkpoint to grow, or a policy from this trainer (any body, same padding) to start
    from; `teacher`: demonstrations to clone first (displague/robot tools/record_teacher.py); `wind_mean`,
    `wind_gust`: 'lo,hi' m/s ranges of each model variant's mean wind and gust sigma (default: no wind);
    rests and locks (issue #56): `locks` the drums self-lock, `power_cost` c in c log(1 + W / 10 W), `quiet` the fraction of
    envs pushed only every 8 to 16 s, `hand_load` 'lo,hi' kg in one hand, `episode_s` the episode length."""
    env_cfg, ppo = _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches, body, pad_dofs,
                         wind_mean, wind_gust, wind_tau, arch, locks, power_cost, quiet, hand_load, episode_s,
                         init_log_std, quiet_noise,
                         teacher_action_half(teacher) if teacher_range and teacher else None)
    label = label or "%s-%s-%s-s%d" % (body, task, arch if arch in ("mlp", "gru") else "init" if init else "random", seed)
    s = ppo_train(env_cfg, ppo, init or None, out, label, teacher=teacher or None)
    print(json.dumps(s, indent=1, default=str))


@app.command()
def finetune(policy: str, workers: int = 20, per_worker: int = 8, iterations: int = 150, seed: int = 0,
             device: str = "cuda", horizon: int = 48, epochs: int = 4, minibatches: int = 4, out: str = "artifacts/rl",
             label: str = "", power_cost: float = 0.5, quiet: float = 0.5, wind_workers: float = 0.3,
             episode_s: float = 20.0, critic_warmup: int = 10, lr: float = 3e-4, init_log_std: float = -3.0,
             desired_kl: float = 0.01, wind_max: float = 8.0):
    """PPO on the full tendon model (rl/tendon_env.py), continuing a stand policy saved by `train`: the drums'
    locks and the electrical power as the robot's model has them (issue #56)."""
    from ai_embedded_dynamic_diversity.rl.tendon_env import TendonEnvConfig, TendonVecEnv

    import torch

    pe = torch.load(policy, map_location="cpu", weights_only=False)["env"]
    cfg = TendonEnvConfig(workers=workers, per_worker=per_worker, power_cost=power_cost, quiet_fraction=quiet,
                          wind_workers=wind_workers, wind_max=wind_max, episode_s=episode_s, seed=seed,
                          action_fraction=pe.get("action_fraction") or 0.2, action_half=pe.get("action_half"))
    env = TendonVecEnv(cfg)
    try:
        ppo = PPOConfig(iterations=iterations, seed=seed, device=device, horizon=horizon, epochs=epochs,
                        minibatches=minibatches, critic_warmup=critic_warmup, lr=lr, save_every=10,
                        init_log_std=init_log_std, desired_kl=desired_kl)
        label = label or "tendon-ft-s%d" % seed
        s = ppo_train(cfg, ppo, None, out, label, env=env, finetune=policy)
    finally:
        env.close()
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
            minibatches: int = 2, out: str = "artifacts/rl", wind_mean: str = "", wind_gust: str = "",
            wind_tau: float = 2.0):
    """Same task, same seed: grown from `checkpoint` versus random weights. Writes <out>/compare-<task>.md."""
    rows = []
    for init, tag in ((checkpoint, "grown"), (None, "random")):
        env_cfg, ppo = _cfgs(task, num_envs, iterations, seed, device, horizon, epochs, minibatches,
                             wind_mean=wind_mean, wind_gust=wind_gust, wind_tau=wind_tau)
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

"""The get-up test (issue #49; displague/robot docs/32): a get-up policy run deterministically (its mean action) from
the curriculum's starts on the stand-in. Reports the rise rate (stood 1 s, the curriculum's `ever_stood`), the
fraction still standing at the end, how many the damage gates ended, the worst lines' force against their
continuous rating, the longest time over it and the peak shell impact.

    python -m ai_embedded_dynamic_diversity.rl.getup_eval artifacts/rl/getup/gu-mlp-long-s0.pt
    python -m ai_embedded_dynamic_diversity.rl.getup_eval <policy> --start fallen --n 64 --seed 7
"""
from __future__ import annotations

import numpy as np
import torch
import typer

from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
from ai_embedded_dynamic_diversity.rl.tendon_eval import load_policy

app = typer.Typer(add_completion=False)


def evaluate_getup(policy: str, start: str = "curriculum", n: int = 64, seed: int = 7) -> dict:
    """`start`: "curriculum" begins every episode at the first start (the half-kneel), "fallen" from a fall."""
    actor, norm, saved = load_policy(policy)
    pe = saved["env"]
    env = StandinEnv(EnvConfig(task="getup", num_envs=n, variants=16, nthread=16, seed=seed, getup_start=start,
                               action_fraction=pe.get("action_fraction"), action_half=pe.get("action_half"),
                               ankle_torque=bool(pe.get("ankle_torque"))))
    env.reset()
    obs, _ = env.observe()
    mem = actor.initial_state(n, "cpu")
    alive = np.ones(n, bool)
    stood = np.zeros(n, bool)
    line_max = np.zeros(env.nu)
    over_any = 0.0
    impact = np.zeros(n)
    z_end = np.zeros(n)
    with torch.no_grad():
        for _ in range(env.max_steps - 1):
            m, mem = actor(norm(torch.as_tensor(obs)), mem)
            obs, _, _, done, info = env.step(m.clamp(-1, 1).numpy())
            line = env._line_force(env.sens[:, env.s_tau]) / env.f_cont
            line_max = np.maximum(line_max, (line * alive[:, None]).max(axis=0))
            over_any = max(over_any, float(info.line_over_max[alive].max()) if alive.any() else 0.0)
            impact = np.maximum(impact, np.where(alive, info.impact, 0.0))
            stood |= alive & info.extras["ever_stood"]
            z_end = np.where(alive, info.extras["z"], z_end)
            alive &= ~done
    k = np.argsort(-line_max)[:5]
    return dict(n=n, start=start, rise=float(stood.mean()), standing_end=float((z_end > 0.85 * env.stand_z).mean()),
                ended_by_damage=int((~alive).sum()),
                worst_lines={env.dof_names[i]: float(line_max[i]) for i in k},
                longest_over_rating_s=over_any, peak_impact_n=float(impact.max()),
                median_impact_n=float(np.median(impact)))


@app.command()
def main(policy: str, start: str = "curriculum", n: int = 64, seed: int = 7):
    import json
    print(json.dumps(evaluate_getup(policy, start, n, seed), indent=1))


if __name__ == "__main__":
    app()

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


def evaluate_getup(policy: str, start: str = "curriculum", n: int = 64, seed: int = 7, guide: float | None = None,
                   wait: str = "", mute: bool = False) -> dict:
    """`start`: "curriculum" begins every episode at the first start (the half-kneel), "fallen" from a fall.
    Always without the helper's hands (the unassisted test). A policy trained with the coach's voice hears it here
    too: `guide` overrides the share of episodes with cues (0: the command only), `wait` 'lo,hi' s of "stay"
    before the command, `mute` silences the coach (its script runs, so the command still starts the clock, but
    the ears hear noise only)."""
    actor, norm, saved = load_policy(policy)
    pe = saved["env"]
    env = StandinEnv(EnvConfig(task="getup", num_envs=n, variants=16, nthread=16, seed=seed, getup_start=start,
                               action_fraction=pe.get("action_fraction"), action_half=pe.get("action_half"),
                               ankle_torque=bool(pe.get("ankle_torque")), coach=pe.get("coach", "off"),
                               coach_guide=pe.get("coach_guide", 1.0) if guide is None else guide,
                               coach_wait=tuple(float(x) for x in wait.split(",")) if wait else (0.0, 0.0)))
    if mute and env.voice is not None:
        env.voice.mute = True
    env.reset()
    obs, _ = env.observe()
    mem = actor.initial_state(n, "cpu")
    alive = np.ones(n, bool)
    stood = np.zeros(n, bool)
    false_start = np.zeros(n, bool)
    told = np.zeros(n, bool)
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
            false_start |= alive & info.extras["false_start"]
            if env.voice is not None:
                told |= alive & env.voice.told
            z_end = np.where(alive, info.extras["z"], z_end)
            alive &= ~done
    k = np.argsort(-line_max)[:5]
    heard = {}
    if env.voice is not None:      # of those told to rise, how many did; of those never told, how many stayed
        heard = dict(told=int(told.sum()), rise_when_told=float(stood[told].mean()) if told.any() else None,
                     false_start=float(false_start.mean()),
                     stayed_when_not_told=float((~false_start[~told]).mean()) if (~told).any() else None)
    return dict(n=n, start=start, **heard, rise=float(stood.mean()), standing_end=float((z_end > 0.85 * env.stand_z).mean()),
                ended_by_damage=int((~alive).sum()),
                worst_lines={env.dof_names[i]: float(line_max[i]) for i in k},
                longest_over_rating_s=over_any, peak_impact_n=float(impact.max()),
                median_impact_n=float(np.median(impact)))


@app.command()
def main(policy: str, start: str = "curriculum", n: int = 64, seed: int = 7, guide: float = -1.0, wait: str = "",
         mute: bool = False):
    import json
    print(json.dumps(evaluate_getup(policy, start, n, seed, None if guide < 0 else guide, wait, mute), indent=1))


if __name__ == "__main__":
    app()

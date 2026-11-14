"""The mirror test: a policy trained on the `mirror` task (rl/standin_env.py) follows a trainer's squat by sight.

    python -m ai_embedded_dynamic_diversity.rl.mirror_eval artifacts/rl/vision/mirror-s0.pt
    python -m ai_embedded_dynamic_diversity.rl.mirror_eval <policy> --hide      # the trainer gone: the control
    python -m ai_embedded_dynamic_diversity.rl.mirror_eval <policy> --light 0.02,0.02

Mean action, 64 robots, 20 s. Reports how closely the pelvis follows the height shown (the mean error and the
correlation between the two), against a robot that just stands (the error of ignoring the trainer), and how many
fell. With `--hide` the trainer still leads, unseen: a policy that follows then is not following by sight.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import typer

from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
from ai_embedded_dynamic_diversity.rl.tendon_eval import load_policy

app = typer.Typer(add_completion=False)


def evaluate_mirror(policy: str, n: int = 64, seed: int = 7, seconds: float = 20.0, hide: bool = False,
                    shut: bool = False, light: str = "") -> dict:
    actor, norm, saved = load_policy(policy)
    pe = saved["env"]
    env = StandinEnv(EnvConfig(task="mirror", num_envs=n, variants=16, nthread=16, seed=seed, episode_s=seconds,
                               action_fraction=pe.get("action_fraction"), eyes=pe.get("eyes", "three"),
                               trainer=pe.get("trainer", "squat"), trainer_arc=pe.get("trainer_arc", 40.0),
                               mirror_depth=pe.get("mirror_depth", 0.07), mirror_tol=pe.get("mirror_tol", 0.02),
                               light=tuple(float(x) for x in light.split(",")) if light else tuple(
                                   pe.get("light", (0.08, 1.0)))))
    env.figure.hidden, env.eyes.shut = hide, shut
    env.reset()
    obs, _ = env.observe()
    mem = actor.initial_state(n, "cpu")
    alive = np.ones(n, bool)
    hs, ts = [], []
    with torch.no_grad():
        for _ in range(env.max_steps - 1):
            m, mem = actor(norm(torch.as_tensor(obs)), mem)
            obs, _, _, done, info = env.step(m.clamp(-1, 1).numpy())
            hs.append(np.where(alive, info.extras["z"] / env.stand_z, np.nan))
            ts.append(np.where(alive, info.extras["mirror_target"], np.nan))
            alive &= ~done
    h, t = np.array(hs)[50:], np.array(ts)[50:]              # after the first second
    ok = ~np.isnan(h)
    env.close()
    return dict(n=n, seconds=seconds, hide=hide, shut=shut, fell=int((~alive).sum()),
                height_error=float(np.abs(h[ok] - t[ok]).mean()),
                error_if_it_just_stood=float(np.abs(np.nanmean(h) - t[ok]).mean()),
                correlation=float(np.corrcoef(h[ok], t[ok])[0, 1]),
                height_when_shown_low=float(h[ok][t[ok] < t[ok].min() + 0.02].mean()),
                height_when_shown_standing=float(h[ok][t[ok] > 0.995].mean()))


@app.command()
def main(policy: str, n: int = 64, seed: int = 7, seconds: float = 20.0, hide: bool = False, shut: bool = False,
         light: str = ""):
    print(json.dumps(evaluate_mirror(policy, n, seed, seconds, hide, shut, light), indent=1))


if __name__ == "__main__":
    app()

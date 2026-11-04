"""Rests and locks in the stand-in (issue #56): the drive's electrical power as the robot's tendon model counts it,
and its drums' self-lock gated on whole-body stillness (displague/robot docs/30)."""
from __future__ import annotations

import numpy as np
import pytest


def _env_available() -> bool:
    try:
        import json

        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.rl.standin_env import body_paths
        _, spec = body_paths("tendon32")
        with open(spec) as f:
            return "locks" in json.load(f)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _env_available(),
                                reason="needs mujoco and a displague/robot checkout whose stand-in has locks")


def _env(**kw):
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    return StandinEnv(EnvConfig(task="stand", num_envs=2, variants=1, nthread=1, seed=0, max_delay_steps=0,
                                push_interval_s=(100.0, 100.0), **kw))


def _run(env, actions):
    env.reset()
    p, lk = [], []
    for a in actions:
        _, _, r, done, info = env.step(a)
        p.append(info.extras["power_w"].copy())
        lk.append(info.extras["locked"].copy())
    return np.array(p), np.array(lk), done


def test_held_pose_locks_and_rests():
    env = _env(locks=True, episode_s=20.0)
    zero = np.zeros((2, env.act_dim))
    p, lk, done = _run(env, [zero] * 400)               # 8 s holding the standing pose (it sways ~5 s first)
    assert not done.any()
    assert lk[-1].min() > 0.5                            # most drums locked once the body is still
    assert p[-25:].mean() < 0.1 * p[:50].mean()          # and the power falls to near zero


def test_dither_costs_power_and_never_locks():
    env = _env(locks=True)
    rng = np.random.default_rng(0)
    acts = [rng.normal(0.0, 0.3, (2, env.act_dim)) for _ in range(100)]
    p, lk, _ = _run(env, acts)
    held = _env(locks=True)
    ph, _, _ = _run(held, [np.zeros((2, held.act_dim))] * 100)
    assert p[-50:].mean() > 5 * ph[-50:].mean()
    assert lk[-50:].mean() < 0.2


def test_power_cost_lowers_reward_and_defaults_off():
    off, on = _env(), _env(power_cost=0.5)
    assert off.priv_dim + 3 == on.priv_dim
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.3, (2, off.act_dim))
    off.reset(), on.reset()
    r_off = sum(off.step(a)[2].sum() for _ in range(10))
    r_on = sum(on.step(a)[2].sum() for _ in range(10))
    assert r_on < r_off

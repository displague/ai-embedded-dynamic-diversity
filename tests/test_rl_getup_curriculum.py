"""Get-up as a curriculum from a half-kneel (issue #49): the start rests upright where it is placed, and a start is
unlocked only once the one before it succeeds on enough episodes."""
from __future__ import annotations

import numpy as np
import pytest


def _env_available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.rl.standin_env import body_paths
        body_paths("tendon32")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _env_available(), reason="needs mujoco and a displague/robot checkout")


def _env(n=4):
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    return StandinEnv(EnvConfig(task="getup", num_envs=n, variants=1, nthread=1, seed=0, getup_start="curriculum"))


def test_halfkneel_start_rests_upright():
    env = _env()
    env.reset()
    q, v, R, up, grav, lin = env._pelvis()
    assert (env.level == 0).all()                       # the first start: half-kneeling
    assert (up > 0.9).all()
    assert (np.abs(q[:, 2] - 0.46) < 0.08).all()        # pelvis about 0.46 m up, resting on a foot and a knee
    pose = env.target.copy()
    for _ in range(50):                                 # held for 1 s by the servos, it stays there
        env._roll(np.arange(env.N), pose, env.substeps)
    q, v, R, up, grav, lin = env._pelvis()
    assert (up > 0.9).all()


def test_unlock_needs_sustained_success():
    env = _env(2)
    env.reset()
    idx = np.array([0, 1])
    env.ever_stood[:] = True
    for _ in range(50):                                 # 100 successes: not yet enough episodes to unlock
        env._curriculum_record(idx)
    assert env.frontier == 0
    for _ in range(60):
        env._curriculum_record(idx)
    assert env.frontier == 1                            # past 200 episodes at success 1.0: fallen starts unlock
    assert set(env._curriculum_pick(1000)) == {0, 1}    # and the half-kneel is still rehearsed

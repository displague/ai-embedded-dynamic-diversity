"""PPO on the full tendon model (rl/tendon_env.py): the worker processes step the robot's rigs one control tick at a
time and return the gate's observation, a finite reward and the electrical power (issue #56)."""
from __future__ import annotations

import numpy as np
import pytest


def _available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.sim.robot_tendon import find_robot_repo
        import os
        repo = find_robot_repo(None)
        with open(os.path.join(repo, "tools", "sim_tendon.py")) as f:
            return "still_gate" in f.read()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="needs mujoco and a displague/robot checkout with still_gate")


def test_steps_and_reports_power():
    from ai_embedded_dynamic_diversity.rl.tendon_env import TendonEnvConfig, TendonVecEnv

    env = TendonVecEnv(TendonEnvConfig(workers=1, per_worker=2, wind_workers=0.0))
    try:
        obs, priv = env.observe()
        assert obs.shape == (2, env.obs_dim) and priv.shape == (2, env.priv_dim)
        assert np.isfinite(obs).all() and np.isfinite(priv).all()
        for _ in range(10):
            obs, priv, r, done, info = env.step(np.zeros((2, env.act_dim), dtype=np.float32))
        assert np.isfinite(r).all() and np.isfinite(obs).all()
        assert (info.extras["power_w"] >= 0).all()
        assert not done.any()
    finally:
        env.close()

from __future__ import annotations

import numpy as np
import pytest


def _env_available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.rl.standin_env import body_paths
        body_paths("tendon32")
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _env_available(),
                                reason="needs mujoco and a checkout of displague/robot (ROBOT_REPO)")


def _run(seconds=1.5, **wind):
    """Two envs (env 0 on the nominal variant, env 1 on a randomised one), no pushes, zero actions."""
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    env = StandinEnv(EnvConfig(task="stand", num_envs=2, variants=2, nthread=2, seed=5,
                               push_interval_s=(100.0, 100.0), max_delay_steps=0, **wind))
    env.reset()
    start = env.state[:, env.i_qpos:env.i_qpos + 2].copy()
    obs = priv = None
    for _ in range(int(round(seconds / env.dt_ctrl))):
        obs, priv, r, done, info = env.step(np.zeros((2, env.act_dim)))
        assert not done.any()
    moved = env.state[:, env.i_qpos:env.i_qpos + 2] - start
    out = dict(env=env, obs=obs, priv=priv, moved=moved, info=info)
    env.close()
    return out


def test_wind_off_is_calm_and_keeps_the_layout() -> None:
    a, b = _run(), _run(wind_mean=(0.0, 0.0), wind_gust=(0.0, 0.0), wind_tau=0.5, air_density=2.0)
    env = a["env"]
    assert not env.wind_on
    for m in env.models:
        assert m.opt.density == 0.0 and not np.any(m.opt.wind)
    assert env.priv_dim == env.obs_dim + 1 + 1 + 3 + env.n_impact_slots + env.P + 1
    assert "wind" not in a["info"].extras
    # bit-identical with or without the (inactive) wind settings
    assert np.array_equal(a["obs"], b["obs"]) and np.array_equal(a["priv"], b["priv"])
    assert np.array_equal(a["moved"], b["moved"])


def test_wind_pushes_a_standing_standin_downwind() -> None:
    calm = _run()
    windy = _run(wind_mean=(9.0, 9.0), wind_ramp_s=0.0)
    env = windy["env"]
    assert env.wind_on and env.priv_dim == calm["env"].priv_dim + 2
    assert windy["priv"].shape == (2, env.priv_dim)
    # the nominal variant stays calm and unchanged
    assert not np.any(env.models[0].opt.wind) and env.models[0].opt.density == 0.0
    assert np.array_equal(windy["moved"][0], calm["moved"][0])
    # the randomised variant: 9 m/s along its direction, the same randomisation otherwise
    u = env.wind_u[1]
    assert np.allclose(env.models[1].opt.wind[0:2], 9.0 * u)
    assert np.allclose(windy["info"].extras["wind"][1], 9.0 * u)
    downwind = float(windy["moved"][1] @ u) - float(calm["moved"][1] @ u)
    assert downwind > 0.005, downwind


def test_gusts_have_the_robot_statistics() -> None:
    """Ornstein-Uhlenbeck gusts: sigma along the wind, 0.75 sigma across, decorrelating over tau."""
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    env = StandinEnv(EnvConfig(task="stand", num_envs=2, variants=2, nthread=1, wind_mean=(5.0, 5.0),
                               wind_gust=(2.0, 2.0), wind_ramp_s=0.0))
    g = []
    for _ in range(20000):
        env._blow(env.dt_ctrl)
        g.append(env.wind_g[1].copy())
    env.close()
    g = np.array(g)
    sd = g.std(axis=0)
    assert abs(sd[0] - 2.0) < 0.3 and abs(sd[1] - 1.5) < 0.25, sd
    lag = int(round(2.0 / 0.02))
    rho = np.corrcoef(g[:-lag, 0], g[lag:, 0])[0, 1]
    assert abs(rho - np.exp(-1.0)) < 0.15, rho

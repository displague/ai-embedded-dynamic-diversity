"""The stand-in's drive cascade (displague/robot tools/export_standin.py `outer_loop`): a lagged servo target
and the outer joint loop's integral trim, so a joint held under load reaches its target as on the tendon model."""
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
            return "outer_loop" in json.load(f)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _env_available(),
                                reason="needs mujoco and a displague/robot checkout whose stand-in has outer_loop")


def _knee_step(outer_loop: bool, seconds: float = 1.0, delta: float = 0.03) -> float:
    """Fraction of a knee target step reached after `seconds`, standing, nominal model."""
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    env = StandinEnv(EnvConfig(task="stand", num_envs=1, variants=1, nthread=1, seed=0, max_delay_steps=0,
                               push_interval_s=(100.0, 100.0), outer_loop=outer_loop))
    env.reset()
    k = env.dof_names.index("shank.L_pitch")
    a = np.zeros((1, env.act_dim))
    for _ in range(25):
        env.step(a)
    q0 = env.state[0, env.i_qpos:env.i_qpos + env.nq][env.qadr[k]]
    a[0, k] = delta / env.half[k]
    for _ in range(int(round(seconds / env.dt_ctrl))):
        env.step(a)
    q1 = env.state[0, env.i_qpos:env.i_qpos + env.nq][env.qadr[k]]
    env.close()
    return float((q1 - q0) / delta)


def test_outer_loop_holds_a_loaded_knee_at_its_target() -> None:
    on, off = _knee_step(True), _knee_step(False)
    assert on > 0.8, on                 # the tendon model reaches about 0.9 in 1 s (robot docs/29)
    assert on > off + 0.4, (on, off)    # without the trim the servo sags under the body's weight


def test_servo_starts_at_its_target() -> None:
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    env = StandinEnv(EnvConfig(task="stand", num_envs=2, variants=2, nthread=1, seed=0))
    env.reset()
    assert env.na == env.nu
    act = env.state[:, env.i_act:env.i_act + env.na]
    assert np.allclose(act, env.default, atol=0.05)
    assert np.all(env.models[1].actuator_dynprm[:, 0] != env.models[0].actuator_dynprm[:, 0])
    env.close()

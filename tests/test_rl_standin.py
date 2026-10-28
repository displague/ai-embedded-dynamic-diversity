from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_embedded_dynamic_diversity.models.core import ModelCore
from ai_embedded_dynamic_diversity.rl.actor import DEFAULT_CORE, CoreActor, RunningNorm, grow_core


def _checkpoint(tmp_path):
    torch.manual_seed(3)
    core = ModelCore(**DEFAULT_CORE)
    path = tmp_path / "champ.pt"
    torch.save({"model": core.state_dict(), "model_config": dict(DEFAULT_CORE)}, path)
    return core, str(path)


def test_grown_core_matches_the_original_on_its_old_inputs(tmp_path) -> None:
    old, path = _checkpoint(tmp_path)
    grown, report = grow_core(104, 32, path)
    assert report["grown"] == ["passive.encoder.0.weight", "router.base_map"]
    assert not report["fresh"]
    x = torch.randn(5, 48)
    x_big = torch.zeros(5, 104)
    x_big[:, :48] = x                                   # the new input channels silent
    mem = old.init_memory(5, old.memory_slots, old.memory_dim, "cpu")
    a, b = old(x, mem), grown(x_big, mem)
    assert torch.allclose(a["latent"], b["latent"], atol=1e-6)
    assert torch.allclose(a["memory"], b["memory"], atol=1e-6)
    assert b["io"].shape == (5, 32)


def test_random_core_has_the_same_architecture(tmp_path) -> None:
    _, path = _checkpoint(tmp_path)
    grown, _ = grow_core(104, 32, path)
    fresh, report = grow_core(104, 32, None)
    assert report["checkpoint"] is None
    assert {k: v.shape for k, v in grown.state_dict().items()} == {k: v.shape for k, v in fresh.state_dict().items()}


def test_actor_starts_near_the_standing_pose() -> None:
    core, _ = grow_core(104, 32, None)
    actor = CoreActor(core)
    mean, mem = actor(torch.zeros(8, 104), actor.initial_state(8, "cpu"))
    assert mean.shape == (8, 32) and float(mean.abs().max()) < 1.0
    assert mem.shape == (8, core.memory_slots, core.memory_dim)


def test_running_norm() -> None:
    n = RunningNorm(3)
    x = torch.randn(1000, 3) * 4 + 2
    n.update(x)
    y = n(x)
    assert abs(float(y.mean())) < 0.05 and abs(float(y.std()) - 1.0) < 0.05


def _env_available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.rl.standin_env import body_paths
        body_paths("tendon32")
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _env_available(), reason="needs mujoco and a checkout of displague/robot (ROBOT_REPO)")
def test_standin_env_steps_and_holds_the_standing_pose() -> None:
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

    env = StandinEnv(EnvConfig(task="stand", num_envs=8, variants=2, nthread=2, push_interval_s=(100.0, 100.0)))
    env.reset()
    obs, priv = env.observe()
    assert obs.shape == (8, env.obs_dim) and priv.shape == (8, env.priv_dim)
    for _ in range(50):
        obs, priv, r, done, info = env.step(np.zeros((8, env.act_dim)))
    assert not done.any() and info.standing.all()
    assert np.isfinite(obs).all() and np.isfinite(r).all()
    env.close()

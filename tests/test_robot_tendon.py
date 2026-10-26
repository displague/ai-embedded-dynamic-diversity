from __future__ import annotations

import pytest

from ai_embedded_dynamic_diversity.sim import robot_tendon
from ai_embedded_dynamic_diversity.sim.embodiments import discover_from_spec, get_embodiment


def _available() -> bool:
    try:
        robot_tendon.find_robot_repo()
        import mujoco  # noqa: F401
    except (FileNotFoundError, ImportError):
        return False
    return True


needs_robot = pytest.mark.skipif(not _available(), reason="needs mujoco and a checkout of displague/robot (ROBOT_REPO)")


@needs_robot
def test_tendon_humanoid_registers_and_observes() -> None:
    env = robot_tendon.TendonHumanoid(control_hz=50.0)
    emb = get_embodiment(robot_tendon.EMBODIMENT_NAME)
    assert len(emb.controls) == 32
    assert {"strain", "pressure", "imu"} <= set(emb.sensors)
    obs = env.reset()
    assert obs.shape == (len(env.layout),)
    assert float(obs.min()) >= 0.0 and float(obs.max()) <= 1.0
    spec = env.spec()
    rediscovered = discover_from_spec({"name": spec["name"], "controls": spec["controls"], "sensors": spec["sensors"]})
    assert rediscovered.controls == emb.controls


@needs_robot
def test_tendon_humanoid_stands_under_hold_policy() -> None:
    result = robot_tendon.smoke_rollout(steps=50, policy="hold")
    assert result["standing_at_end"]
    assert result["standing_fraction"] == 1.0
    assert result["worst_line_over_continuous"] < 3.0

"""The gate runs a policy as it was trained (pre-release review of v0.4.0): a policy saved with `hand_ankles` runs as
the hybrid without the command-line flag; in the hybrid the hand-written ankles keep the model's own ankle gain
even for a torque-mode policy; the hand-written controller restores that gain after a torque-mode policy."""
from __future__ import annotations

import os

import pytest
import torch


def _available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.sim.robot_tendon import find_robot_repo
        repo = find_robot_repo(None)
        with open(os.path.join(repo, "tools", "sim_tendon.py")) as f:
            return "still_gate" in f.read()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _available(), reason="needs mujoco and a displague/robot checkout with still_gate")


def _policy(tmp_path, **env):
    from ai_embedded_dynamic_diversity.rl.actor import PlainMLP, RunningNorm
    from ai_embedded_dynamic_diversity.rl.tendon_eval import _load_sim_tendon, find_robot_repo

    st = _load_sim_tendon(find_robot_repo(None))
    names = [w["name"] for w in st.Rig().w]
    n = len(names)
    obs_dim = 3 * n + 3 + 3 + 2           # q, qd, previous action, gravity, gyro, foot loads
    net = PlainMLP(obs_dim, n, 16)
    path = os.path.join(tmp_path, "p.pt")
    torch.save(dict(core=net.state_dict(), log_std=torch.zeros(n), arch="mlp",
                    core_config=dict(signal_dim=obs_dim, io_channels=n, hidden=16),
                    obs_norm=RunningNorm(obs_dim).state_dict(), dof_names=names,
                    env=dict(task="stand", action_fraction=0.2, **env)), path)
    return path


def test_saved_hand_ankles_and_ankle_gain(tmp_path):
    from ai_embedded_dynamic_diversity.rl.tendon_eval import TendonPolicyRunner, _TeacherRunner

    hy = TendonPolicyRunner(_policy(tmp_path, hand_ankles=True, ankle_torque=True))
    assert hy.hand_ankles                                  # from the saved policy, no flag
    hy.reset()
    assert hy.rig.balance
    assert hy.st.OUTER_GAIN["foot"] == hy.st._OUTER_GAIN_FOOT   # the hand-written ankles keep their gain

    alone = TendonPolicyRunner(_policy(tmp_path, ankle_torque=True))
    assert not alone.hand_ankles
    alone.reset()
    assert alone.st.OUTER_GAIN["foot"] == 0.0              # torque mode for the policy's own ankles

    t = _TeacherRunner()
    t.reset()
    assert t.st.OUTER_GAIN["foot"] == t.st._OUTER_GAIN_FOOT     # restored for the hand-written controller

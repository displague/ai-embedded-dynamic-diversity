"""A champion's core with a body's own roots (rl/actor.RootedCore), cut off and put back (rl/cut_back.py)."""
from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.models.core import ModelCore
from ai_embedded_dynamic_diversity.rl import cut_back
from ai_embedded_dynamic_diversity.rl.actor import DEFAULT_CORE, CoreActor, RootedCore, build_policy_net, grow_core

OBS, ACT = 20, 6


def _champion(tmp_path):
    torch.manual_seed(0)
    core = ModelCore(**DEFAULT_CORE)
    path = str(tmp_path / "champion.pt")
    torch.save({"model": core.state_dict(), "model_config": dict(DEFAULT_CORE), "profile": "pi5"}, path)
    return path, core


def _policy(tmp_path, champ, name="policy.pt", frozen=False):
    core, report = grow_core(OBS, ACT, champ, roots=True)
    with torch.no_grad():                       # a body that has learned something
        if frozen:                              # in its roots only
            core.inner.passive.encoder[0].weight[:, 48:].add_(0.1)
            core.inner.router.base_map[16:].add_(0.1)
        else:
            for p in core.parameters():
                p.add_(0.01 * torch.randn_like(p))
    inner = core.inner
    path = str(tmp_path / name)
    torch.save(dict(core=core.state_dict(), log_std=torch.zeros(ACT), arch="core", obs_norm={"mean": torch.zeros(OBS),
                    "var": torch.ones(OBS), "count": torch.tensor(1.0)}, env=dict(body="tendon32"), ppo={},
                    grow=report, dof_names=["j%d" % i for i in range(ACT)],
                    trunk_config={k: DEFAULT_CORE[k] for k in DEFAULT_CORE if k not in ("signal_dim", "io_channels")}),
               path)
    return path, core


def test_a_rooted_core_takes_the_bodys_sizes_and_starts_as_the_champion(tmp_path):
    champ, base = _champion(tmp_path)
    core, report = grow_core(OBS, ACT, champ, roots=True)
    assert isinstance(core, RootedCore) and (core.signal_dim, core.io_channels) == (OBS, ACT)
    assert report["roots"] and report["old_signal_dim"] == 48 and report["old_io_channels"] == 16
    out = core(torch.randn(3, OBS), core.init_memory(3, core.memory_slots, core.memory_dim, "cpu"))
    assert out["io"].shape == (3, ACT)
    assert torch.equal(core.inner.passive.encoder[0].weight[:, :48], base.passive.encoder[0].weight)
    assert torch.equal(core.inner.router.base_map[:16], base.router.base_map)


def test_learning_the_body_sends_no_gradient_to_the_champions_own_inputs_and_outputs(tmp_path):
    champ, _ = _champion(tmp_path)
    core, _ = grow_core(OBS, ACT, champ, roots=True)
    actor = CoreActor(core)
    mean, _ = actor(torch.randn(5, OBS), actor.initial_state(5, "cpu"))
    mean.pow(2).sum().backward()
    g_in, g_out = core.inner.passive.encoder[0].weight.grad, core.inner.router.base_map.grad
    assert g_in[:, :48].abs().max() == 0 and g_in[:, 48:].abs().max() > 0
    assert g_out[:16].abs().max() == 0 and g_out[16:].abs().max() > 0


def test_cutting_the_roots_keeps_them_and_embodying_gives_the_same_policy(tmp_path):
    champ, base = _champion(tmp_path)
    policy, core = _policy(tmp_path, champ)
    cut = str(tmp_path / "core.pt")
    cut_back.cut(champ, policy, cut)
    saved = torch.load(cut, weights_only=False)
    assert saved["model"]["passive.encoder.0.weight"].shape == base.passive.encoder[0].weight.shape
    assert saved["bodies"]["tendon32"]["encoder_cols"].shape == (64, OBS)
    assert saved["bodies"]["tendon32"]["router_rows"].shape == (ACT, 64)
    ModelCore(**saved["model_config"]).load_state_dict(saved["model"])        # a champion-sized core again
    back = str(tmp_path / "back.pt")
    cut_back.embody(cut, "tendon32", back)
    again = build_policy_net(torch.load(back, weights_only=False))
    x, m = torch.randn(4, OBS), core.init_memory(4, core.memory_slots, core.memory_dim, "cpu")
    assert torch.allclose(core(x, m)["io"], again(x, m)["io"], atol=1e-6)


def test_another_cores_roots_can_be_tried_on_a_new_trunk(tmp_path):
    champ, _ = _champion(tmp_path)
    policy, _ = _policy(tmp_path, champ)
    cut = str(tmp_path / "core.pt")
    cut_back.cut(champ, policy, cut)
    out = str(tmp_path / "child.pt")
    cut_back.embody(champ, "tendon32", out, cut)                             # the champion's trunk, the cut core's roots
    child = build_policy_net(torch.load(out, weights_only=False))
    roots = torch.load(cut, weights_only=False)["bodies"]["tendon32"]
    assert torch.equal(child.inner.passive.encoder[0].weight[:, 48:], roots["encoder_cols"])
    core2, report = grow_core(OBS, ACT, champ, roots=True, roots_from=policy)
    assert torch.equal(core2.inner.router.base_map[16:], roots["router_rows"]) and report["roots_from"] == policy


def test_pack_puts_a_body_into_a_champion_file_only_if_its_trunk_was_frozen(tmp_path):
    import pytest
    champ, _ = _champion(tmp_path)
    moved, _ = _policy(tmp_path, champ, "moved.pt")
    cut_back.cut(champ, moved, str(tmp_path / "moved-core.pt"))
    with pytest.raises(ValueError):
        cut_back.pack(champ, str(tmp_path / "moved-core.pt"), str(tmp_path / "no.pt"))
    policy, _ = _policy(tmp_path, champ, frozen=True)
    cut = str(tmp_path / "core.pt")
    cut_back.cut(champ, policy, cut)
    out = str(tmp_path / "v.pt")
    cut_back.pack(champ, cut, out, note="test")
    v = torch.load(out, weights_only=False)
    assert "tendon32" in v["bodies"] and v["profile"] == "pi5" and v["body_note"] == "test"
    assert all(torch.equal(v["model"][k], t) for k, t in torch.load(champ, weights_only=False)["model"].items())

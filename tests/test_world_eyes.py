"""Eyes in the cellular world's observation (sim/world.py `eyes`)."""
from __future__ import annotations

import torch

from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld


def _world(eyes, **kw):
    torch.manual_seed(0)
    w = DynamicDiversityWorld(12, 12, 6, 5, eyes=eyes, **kw)
    st = w.init(2)
    return w, st


def test_off_by_default_and_the_first_slots_are_unchanged_with_eyes_on(monkeypatch):
    monkeypatch.delenv("ADD_WORLD_EYES", raising=False)
    assert DynamicDiversityWorld(12, 12, 6, 5).eyes is False
    monkeypatch.setenv("ADD_WORLD_EYES", "1")
    assert DynamicDiversityWorld(12, 12, 6, 5).eyes is True
    a, sa = _world(False)
    b, sb = _world(True)
    oa, ob = a.encode_observation(sa, 48), b.encode_observation(sb, 48)
    assert torch.equal(oa[:, :28], ob[:, :28])
    assert oa[:, 28:].abs().sum() == 0 and ob[:, 28:].abs().sum() > 0


def test_a_small_signal_dim_has_no_room_and_is_left_alone():
    w, st = _world(True)
    assert w.encode_observation(st, 24).shape == (2, 24)


def test_the_periphery_sees_where_the_life_is_and_only_where_it_is_lit():
    w, st = _world(True)
    st.life.zero_()
    st.life[:, :, :, :6, :6] = 1.0                    # life in one quadrant
    c = w.default_controls(2)
    c.light_position[:] = torch.tensor([-0.5, -0.5, 0.0])      # the light over that quadrant
    c.light_intensity[:] = 1.0
    w._prev_controls = c
    eye = w.encode_observation(st, 48)[:, 28:]
    assert eye[0, 0] > 5 * eye[0, 1:4].max()
    c.light_position[:] = torch.tensor([0.9, 0.9, 0.0])        # the light moved away: the same life is dim
    assert w.encode_observation(st, 48)[0, 28] < 0.5 * eye[0, 0]


def test_the_third_eye_reads_stress_whatever_the_light():
    w, st = _world(True)
    st.stress.zero_()
    st.stress[:, :, :, 6:, 6:] = 1.0
    c = w.default_controls(2)
    w._prev_controls = c
    a = w.encode_observation(st, 48)[:, 44:48].clone()
    c.light_intensity[:] = 0.01
    b = w.encode_observation(st, 48)[:, 44:48]
    assert torch.allclose(a, b) and a[0, 3] > 0.9 and a[0, 0] < 0.1


def test_colour_is_the_difference_between_resource_channels_at_the_centre():
    w, st = _world(True)
    st.resources.zero_()
    st.resources[:, 0] = 1.0                           # only the first: "red"
    eye = w.encode_observation(st, 48)[:, 28:]
    assert eye[0, 8:12].min() > 0.9                    # red against green
    st.resources[:, 0], st.resources[:, 1] = 0.0, 1.0
    assert w.encode_observation(st, 48)[0, 36:40].max() < -0.9

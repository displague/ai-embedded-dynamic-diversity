"""Eyes and a visible trainer for the stand-in's tasks (rl/vision.py, rl/coach.Figure)."""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from ai_embedded_dynamic_diversity.rl import vision
from ai_embedded_dynamic_diversity.rl.coach import Figure

AXES = ((1, 0, 0), (0, 1, 0), (0, 0, 1))          # forward x, left y, up z
I3 = np.eye(3)[None]


def _eyes(mode="three", n=1, noise=0.0, light=(1.0, 1.0)):
    e = vision.Eyes(n, np.random.default_rng(0), mode, AXES, noise=noise, light=light)
    e.reset(np.arange(n))
    return e


def _pos(n=1, z=1.5):
    mid = np.tile([0.0, 0.0, z], (n, 1))
    return {"L": mid + [0.0, 0.03, 0.0], "R": mid - [0.0, 0.03, 0.0], "C": mid + [0.0, 0.0, 0.04]}


def _ball(e, at, r=0.15, rgb=(0.85, 0.15, 0.12)):
    sc = e.scene
    sc.ball_pos, sc.ball_r, sc.ball_rgb = np.array([[at]], dtype=float), np.array([r]), np.array([rgb])


def test_modes_give_the_overlaid_pair_the_separate_third_eye_or_both():
    per_eye = 5 * 3 + 3 * (3 * 3)
    assert _eyes("pair").OBS == 2 * per_eye
    assert _eyes("third").OBS == 4 * 3
    assert _eyes("three").OBS == 2 * per_eye + 12
    with pytest.raises(ValueError):
        vision.Eyes(1, np.random.default_rng(0), "one", AXES)


def test_the_periphery_is_wide_and_coarse_and_the_fovea_narrow_and_fine():
    (dp, sp), (df, sf) = vision.grid(*vision.PERIPHERY), vision.grid(*vision.FOVEA)
    wide = lambda d: math.degrees(np.arctan2(d[:, 1], d[:, 0]).max() - np.arctan2(d[:, 1], d[:, 0]).min())
    assert wide(dp) > 4 * wide(df)
    assert sp[0] > 2.5 * sf[0]


def test_the_floor_is_below_and_the_sky_above_and_tilting_the_head_moves_the_horizon():
    e = _eyes("pair")
    lum = e.see(_pos(), I3)[0, :15].reshape(3, 5)            # the left eye's periphery, rows from the top
    assert lum[0].mean() > lum[2].mean() + 0.2               # sky brighter than floor
    th = math.radians(40)                                    # pitch the head down
    down = np.array([[[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]]])
    lum_down = e.see(_pos(), down)[0, :15].reshape(3, 5)
    assert lum_down[0].mean() < lum[0].mean() - 0.1          # the top row now sees floor


def test_a_red_ball_ahead_is_red_in_the_fovea_and_colour_fails_in_the_dark():
    e = _eyes("pair")
    _ball(e, (2.0, 0.0, 1.5))
    k = 15
    rg = e.see(_pos(), I3)[0, k + 9:k + 18].reshape(3, 3)
    assert rg[1, 1] > 0.5 and rg[1, 1] > rg[0, 0] + 0.3      # red at the centre of gaze
    night = vision.Eyes(64, np.random.default_rng(1), "pair", AXES, light=(0.02, 0.02))
    night.reset(np.arange(64))
    night.scene.ball_pos = np.tile([[[2.0, 0.0, 1.5]]], (64, 1, 1))
    night.scene.ball_r, night.scene.ball_rgb = np.array([0.15]), np.array([[0.85, 0.15, 0.12]])
    day = vision.Eyes(64, np.random.default_rng(1), "pair", AXES, light=(1.0, 1.0))
    day.reset(np.arange(64))
    day.scene.ball_pos, day.scene.ball_r, day.scene.ball_rgb = night.scene.ball_pos, night.scene.ball_r, night.scene.ball_rgb
    c = k + 9 + 4
    assert night.see(_pos(64), np.tile(I3, (64, 1, 1)))[:, c].mean() < 0.6 * day.see(_pos(64), np.tile(I3, (64, 1, 1)))[:, c].mean()


def test_the_two_eyes_differ_more_for_a_near_ball_than_a_far_one():
    def gap(dist):
        e = _eyes("pair")
        _ball(e, (dist, 0.0, 1.5), r=0.08)
        o = e.see(_pos(), I3)[0]
        return np.abs(o[:42] - o[42:]).sum()
    assert gap(0.5) > 3 * gap(3.0)


def test_the_third_eye_reads_range_whatever_the_light():
    near, far = [], []
    for light in (0.02, 1.0):
        e = _eyes("third", light=(light, light))
        _ball(e, (1.0, 0.0, 1.54), r=0.2)
        near.append(e.see(_pos(), I3)[0].reshape(3, 4)[1:2, 1:3].mean())
        _ball(e, (4.0, 0.0, 1.54), r=0.2)
        far.append(e.see(_pos(), I3)[0].reshape(3, 4)[1:2, 1:3].mean())
    assert near[0] > far[0] + 0.3
    assert abs(near[0] - near[1]) < 0.05 and abs(far[0] - far[1]) < 0.05


def test_shut_eyes_see_nothing_of_the_ball():
    e = _eyes("three")
    _ball(e, (1.0, 0.0, 1.5))
    e.shut = True
    a = e.see(_pos(), I3).copy()
    _ball(e, (1.0, 0.5, 1.5))
    assert np.abs(e.see(_pos(), I3)[0, -12:] - a[0, -12:]).max() < 1e-9


def _figure(n=3):
    z = np.linspace(0.5, 1.0, 6)
    frames = np.stack([np.stack([np.zeros(6), np.zeros(6), z * h], axis=1) for h in np.linspace(0.5, 1.0, 5)])
    return Figure(n, frames, ["pelvis", "head", "hand.L", "hand.R", "foot.L", "foot.R"], rate=0.5)


def test_the_figure_waits_low_rises_at_the_command_and_paces_the_learner():
    f = _figure()
    f.reset(np.arange(3), np.tile([2.0, 0.0], (3, 1)), np.zeros((3, 2)))
    told = np.array([False, True, True])
    guide = np.array([False, False, True])
    low = np.full(3, 0.5)
    for _ in range(200):
        f.step(told, guide, low, np.zeros(3, bool), 0.02)
    assert f.phase[0] == 0.0                         # not told: it waits
    assert f.phase[1] == pytest.approx(1.0)          # told, unpaced: it stands up
    assert f.phase[2] == pytest.approx(f.lead)       # paced: one lead ahead of a learner still low
    assert f.arms[0] == 0.0 and f.arms[1] == pytest.approx(1.0)
    top = f.balls()[:, 1, 2]
    assert top[1] > top[2] > top[0]
    for _ in range(100):
        f.step(told, guide, np.ones(3), np.ones(3, bool), 0.02)
    assert f.phase[2] == pytest.approx(1.0) and f.arms[1] == 0.0      # the learner stands: arms down


def test_the_figure_faces_the_learner():
    f = _figure(1)
    f.frames[:, 2, 0] = 0.3                          # a hand held forward
    f.reset(np.arange(1), np.array([[0.0, 2.0]]), np.zeros((1, 2)))
    b = f.balls()[0]
    assert b[2, 1] < b[0, 1] - 0.2                   # forward is towards the learner (-y from where it stands)


needs_robot = pytest.mark.skipif(not os.path.isfile(os.path.join(os.environ.get("ROBOT_REPO", "../robot"), "sim",
                                                                 "getup_demo.json")),
                                 reason="needs a displague/robot checkout with sim/getup_demo.json")


@needs_robot
def test_env_without_eyes_is_unchanged_and_with_them_gains_channels():
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
    kw = dict(task="getup", num_envs=8, variants=2, nthread=2, getup_start="curriculum", seed=3)
    a, b = StandinEnv(EnvConfig(**kw)), StandinEnv(EnvConfig(eyes="three", trainer="demo", **kw))
    a.reset(), b.reset()
    oa, pa = a.observe()
    ob, pb = b.observe()
    assert ob.shape[1] == oa.shape[1] + b.eyes.OBS and pb.shape[1] == pa.shape[1] + b.eyes.OBS + 6
    assert np.allclose(oa, ob[:, :oa.shape[1]])      # the body's randomisation is the same with eyes on
    assert b.hearing is False and b.voice is not None     # the script runs; nothing is heard
    for _ in range(3):
        b.step(np.zeros((8, b.act_dim)))
    assert np.isfinite(b.observe()[0]).all()
    a.close(), b.close()


@needs_robot
def test_the_trainer_stands_in_view_and_hiding_it_changes_what_is_seen():
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
    e = StandinEnv(EnvConfig(task="getup", num_envs=32, variants=2, nthread=2, getup_start="curriculum", seed=5,
                             eyes="three", trainer="demo", light=(1.0, 1.0)))
    e.reset()
    q = e.state[:, e.i_qpos:e.i_qpos + e.nq]
    pos, R = e._eye_places(q, None)
    to = e.voice.pos[:, :2] - pos["C"][:, :2]
    fwd = (R @ e.head_axes[0])[:, :2]
    cos = (to * fwd).sum(axis=1) / np.linalg.norm(to, axis=1) / np.maximum(np.linalg.norm(fwd, axis=1), 1e-6)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).max() < 75.0     # within the periphery's field
    seen = e.observe()[0][:, -e.eyes.OBS:]
    e.figure.hidden = True
    assert np.abs(e.observe()[0][:, -12:] - seen[:, -12:]).max() > 0.05
    e.close()


@needs_robot
@pytest.mark.parametrize("who", ["tendon32", "rpo23"])
def test_other_bodies_can_be_the_instructor(who):
    from ai_embedded_dynamic_diversity.rl.coach import figure_frames
    try:
        frames, names = figure_frames(who)
    except (FileNotFoundError, ValueError) as exc:                      # rpo23's meshes are fetched, not committed
        pytest.skip(str(exc))
    top = frames[:, :, 2].max(axis=1)
    assert top[-1] > top[0] + 0.2 and frames[:, :, 2].min() > -0.02


@needs_robot
def test_the_mirror_task_pays_for_following_the_trainers_squat():
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
    e = StandinEnv(EnvConfig(task="mirror", num_envs=8, variants=2, nthread=2, seed=2, eyes="three", trainer="squat"))
    e.reset()
    for _ in range(150):
        _, _, r, _, info = e.step(np.zeros((8, e.act_dim)))
    depth = 1.0 - e.figure.phase
    assert depth.max() > 0.2                                           # the trainer has gone down somewhere
    assert np.allclose(info.extras["mirror_target"], 1.0 - e.cfg.mirror_depth * depth)
    assert not info.extras["false_start"].any()                        # no command to wait for in this task
    with pytest.raises(ValueError):
        StandinEnv(EnvConfig(task="mirror", num_envs=2, variants=1, nthread=1)).step(np.zeros((2, 32)))
    e.close()

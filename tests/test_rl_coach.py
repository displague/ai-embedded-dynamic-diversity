"""The coach (rl/coach.py): a voice heard through two ears, and a helper's hands that fade."""
from __future__ import annotations

import numpy as np
import pytest

from ai_embedded_dynamic_diversity.rl import coach as C


def _voice(n=4, **kw):
    v = C.Voice(n, np.random.default_rng(0), **kw)
    v.reset(np.arange(n), np.zeros((n, 2)))
    return v


def _say(v, word):
    v.word[:] = v.wi[word]
    v.word_t[:] = 3.0
    v.pitch[:] = 0.0
    v.loud[:] = 1.0


def test_words_differ_and_synonyms_share_a_syllable():
    flat = {w: C.word_template(w).mean(axis=0) for w in C.WORD_NAMES}
    for a in C.WORD_NAMES:
        for b in C.WORD_NAMES:
            if a < b:
                assert len(C.word_template(a)) != len(C.word_template(b)) or not np.allclose(flat[a], flat[b], atol=0.05)
    n = int(round(C.SYLLABLE_S / C.TICK_S))
    assert np.allclose(C.word_template("get_up")[-n - 1:], C.word_template("stand_up")[-n - 1:])   # "up"


def test_a_voice_at_the_left_is_louder_and_earlier_in_the_left_ear():
    v = _voice(self_noise=0.0)
    _say(v, "get_up")
    ear_l, ear_r = np.tile([0.075, 0.0, 1.6], (4, 1)), np.tile([-0.075, 0.0, 1.6], (4, 1))
    v.pos[:] = [2.0, 0.0, 1.6]                         # at the robot's left (+x)
    h = v.hear(ear_l, ear_r, np.zeros(4))
    left, right, itd = h[:, :C.BANDS], h[:, C.BANDS:2 * C.BANDS], h[:, 2 * C.BANDS]
    assert (left.sum(axis=1) > right.sum(axis=1) + 0.5).all()
    assert (itd > 0.5).all()
    # the head shadows the high bands more than the low ones
    assert ((left - right)[:, 5].mean() > (left - right)[:, 1].mean())
    v.pos[:] = [0.0, -2.0, 1.6]                        # in front: both ears alike
    front = v.hear(ear_l, ear_r, np.zeros(4))
    assert np.abs(front[:, 2 * C.BANDS]).max() < 0.3
    v.pos[:] = [0.0, 2.0, 1.6]                         # behind: two bare ears cannot tell it from in front
    back = v.hear(ear_l, ear_r, np.zeros(4))
    assert np.abs(front[:, :2 * C.BANDS].mean(axis=0) - back[:, :2 * C.BANDS].mean(axis=0)).max() < 0.6


def test_the_robots_own_noise_masks_the_voice():
    v = _voice()
    _say(v, "lean")
    v.pos[:] = [0.0, -2.5, 1.6]
    ear_l, ear_r = np.tile([0.075, 0.0, 1.6], (4, 1)), np.tile([-0.075, 0.0, 1.6], (4, 1))
    quiet = v.hear(ear_l, ear_r, np.zeros(4))
    v2 = _voice()
    v2.pos[:] = v.pos
    silent = v2.hear(ear_l, ear_r, np.zeros(4))            # nobody speaking
    loud = v.hear(ear_l, ear_r, np.full(4, 6.0))           # the joints thrashing at 6 rad/s
    gap = lambda a: np.abs(a[:, :C.BANDS] - silent[:, :C.BANDS]).max(axis=1).mean()
    assert gap(quiet) > 1.0
    assert (loud[:, 2] > quiet[:, 2] + 1.0).all()          # the drive band fills up


def test_script_stay_then_command_then_cues_then_good():
    v = _voice(n=2, wait=(1.0, 1.0), guide=1.0, gap_s=(0.2, 0.2))
    said = []
    for k in range(400):
        t = np.full(2, k * 0.02)
        stood = np.full(2, k > 250)
        v.speak(t, np.full(2, 0.5 if k < 150 else 0.7 if k < 200 else 0.9), stood, np.zeros(2, bool))
        w = C.WORD_NAMES[v.word[0]] if v.word[0] >= 0 else None
        if w and (not said or said[-1] != w):
            said.append(w)
        if k * 0.02 < 1.0:
            assert not v.told.any()
    assert said[0] == "stay"
    assert said[1] in C.RISE_COMMANDS
    assert said[2:] == ["lean", "push", "tall", "good"]
    assert v.told.all() and v.praised.all()


def test_script_warns_of_an_overloaded_line_and_repeats_the_command_without_guidance():
    v = _voice(n=2, guide=1.0, gap_s=(0.2, 0.2))
    v.guide[:] = [True, False]
    words = [set(), set()]
    for k in range(300):
        v.speak(np.full(2, k * 0.02), np.full(2, 0.5), np.zeros(2, bool), np.full(2, k > 100))
        for i in range(2):
            if v.word[i] >= 0:
                words[i].add(C.WORD_NAMES[v.word[i]])
    assert "easy" in words[0] and "lean" in words[0]
    assert words[1] <= set(C.RISE_COMMANDS)


def test_hands_lift_only_what_the_feet_leave_and_ease_off():
    h = C.Hands(3, np.random.default_rng(0), weight=200.0, cap=0.5, free=0.0, quit_p=0.0)
    h.reset(np.arange(3), 10.0)
    h.cap_env[:] = 100.0
    f = h.force(np.zeros(3), np.array([0.4, 0.4, 0.92]), 0.92, np.array([0.0, 170.0, 0.0]), np.zeros((3, 2)), 0.02)
    assert f[0, 2] == pytest.approx(100.0)      # nothing on the feet: the cap
    assert f[1, 2] == pytest.approx(30.0)       # the feet carry 170 of 200 N: the rest
    assert f[2, 2] == 0.0                       # standing: nothing
    assert h.impulse[0] == pytest.approx(2.0)
    h.quit_at[:] = 1.0                          # the helper walks away after 1 s
    assert (h.force(np.full(3, 2.0), np.full(3, 0.4), 0.92, np.zeros(3), np.zeros((3, 2)), 0.02) == 0.0).all()


def test_hands_fade_on_success_and_come_back_on_failure():
    h = C.Hands(4, np.random.default_rng(0), weight=200.0, cap=0.5, free=0.0, quit_p=0.0, min_episodes=20)
    h.reset(np.arange(4), 10.0)
    for _ in range(5):
        h.record(np.arange(4), np.ones(4, bool))
    assert h.level[0] == pytest.approx(0.8)     # 20 helped episodes, all risen: one step down
    for _ in range(5):
        h.record(np.arange(4), np.zeros(4, bool))
    assert h.level[0] == pytest.approx(0.8 / 0.8 ** 0.5)   # they stopped rising at that level: part of the way back
    h.level[0] = 0.035
    h.ema["helped"][0], h.count["at_level"][0] = 1.0, 100
    h.record(np.arange(4), np.ones(4, bool))
    assert h.level[0] == 0.0                    # withdrawn


def test_each_start_has_its_own_level():
    h = C.Hands(4, np.random.default_rng(0), weight=200.0, cap=0.5, free=0.0, quit_p=0.0, min_episodes=20, starts=2)
    h.reset(np.arange(4), 10.0, np.array([0, 0, 1, 1]))
    for _ in range(10):                         # the first start's episodes rise, the second's do not
        h.record(np.arange(4), np.array([True, True, False, False]))
    assert h.level[0] == pytest.approx(0.8) and h.level[1] == 1.0
    h.reset(np.arange(4), 10.0, np.array([0, 0, 1, 1]))
    assert (h.cap_env[:2] <= 0.8 * 0.5 * 200.0 + 1e-9).all() and (h.cap_env[2:] > 0.5 * 0.6 * 200.0 - 1e-9).all()


def _env_available() -> bool:
    try:
        import mujoco  # noqa: F401
        from ai_embedded_dynamic_diversity.rl.standin_env import body_paths
        body_paths("tendon32")
        return True
    except Exception:
        return False


needs_env = pytest.mark.skipif(not _env_available(), reason="needs mujoco and a displague/robot checkout")


def _env(**kw):
    from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv
    return StandinEnv(EnvConfig(task="getup", num_envs=4, variants=1, nthread=1, seed=0, getup_start="curriculum", **kw))


@needs_env
def test_env_without_a_coach_is_unchanged_and_with_one_gains_channels():
    plain, coached = _env(), _env(coach="voice", help_cap=0.6)
    assert coached.obs_dim == plain.obs_dim + C.Voice.OBS
    assert coached.priv_dim == plain.priv_dim + C.Voice.OBS + 3
    plain.reset()
    coached.reset()
    assert np.allclose(plain.state, coached.state)      # the same starts: the coach draws from its own streams
    o, p = coached.observe()
    assert o.shape == (4, coached.obs_dim) and p.shape == (4, coached.priv_dim)
    assert np.isfinite(o).all()


@needs_env
def test_helper_lifts_a_kneeling_robot_and_rising_before_the_command_is_a_false_start():
    env = _env(coach="voice", coach_wait=(5.0, 5.0), help_cap=0.6, help_free=0.0, help_quit=0.0)
    env.reset()
    lifted = 0.0
    for _ in range(100):                               # the standing pose as the target: the robot lunges up
        _, _, r, done, info = env.step(np.zeros((4, env.act_dim)))
        lifted = max(lifted, float(env.hands.lift.max()))
    assert lifted > 0.2 * env.weight
    assert not env.voice.told.any()
    assert not info.extras["ever_stood"].any()         # nothing before the command counts as a rise
    assert info.extras["false_start"].any()


def test_a_muted_coach_runs_its_script_in_silence():
    v = _voice(self_noise=0.0)
    _say(v, "get_up")
    v.mute = True
    assert (v.source() == 0.0).all()
    v.speak(np.full(4, 5.0), np.full(4, 0.5), np.zeros(4, bool), np.zeros(4, bool))
    v.word[:] = -1
    v.speak(np.full(4, 6.0), np.full(4, 0.5), np.zeros(4, bool), np.zeros(4, bool))
    assert v.told.all()


def test_the_echo_remembers_a_word_through_the_pause_after_it():
    ear_l, ear_r = np.tile([0.075, 0.0, 1.6], (4, 1)), np.tile([-0.075, 0.0, 1.6], (4, 1))
    echoes = {}
    for word in ("stay", "get_up"):
        v = _voice(self_noise=0.0)
        v.pos[:] = [0.0, -2.0, 1.6]
        v.pitch[:], v.loud[:], v.ambient[:] = 0.0, 1.0, C.E_REF
        for k in range(40):                              # the word, then 0.4 s of silence
            v.word[:] = v.wi[word] if k < C.TEMPLATE_LEN[v.wi[word]] else -1
            v.word_t[:] = k
            h = v.hear(ear_l, ear_r, np.zeros(4))
        echoes[word] = h[:, -C.BANDS:].mean(axis=0)
        assert (h[:, :C.BANDS].mean(axis=0) < 1.5).all()                # the ears hear only noise now
    assert np.abs(echoes["stay"] - echoes["get_up"]).max() > 0.1        # but the echo still tells the words apart


def test_the_wait_begins_only_when_armed():
    v = C.Voice(4, np.random.default_rng(0), wait=(2.0, 3.0))
    v.armed = False
    v.reset(np.arange(4), np.zeros((4, 2)))
    assert (v.t_cmd == 0.0).all()
    v.armed = True
    v.reset(np.arange(4), np.zeros((4, 2)))
    assert (v.t_cmd >= 2.0).all()

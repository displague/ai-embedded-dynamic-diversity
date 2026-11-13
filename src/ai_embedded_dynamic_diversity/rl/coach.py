"""A coach for the stand-in's tasks: guidance by voice (hands off), and a helper's hands that fade.

Two kinds of support a physical trainer gives, both outside the policy and both meant to be withdrawn:

`Voice`: spoken instruction, heard through two ears.
  - The coach stands somewhere around the robot and speaks words: a command ("get up" or "stand up", "stay",
    "come here"), then, watching the robot, cues ("lean", "push", "tall", "easy" when a line is over its rating)
    and "good" once it stands.
  - A word is a short sequence of syllables; a syllable is a pattern of energy over `BANDS` frequency bands (two
    formant peaks, or a hiss). Nothing here is a waveform: the model is what a cochlea gives at the control rate,
    band energies per ear. Each episode's voice has its own pitch (a shift along the bands), tempo and loudness.
  - To each ear: spreading (1 / r^2), the head's shadow on the far ear (more in the high bands), ambient noise, and
    the robot's own drive noise (it grows with the joints' speed, so a thrashing robot hears less).
  - The observation: log band energies for the left and the right ear, and the interaural time difference
    (the path difference over the speed of sound, noisy when the voice is faint): 2 BANDS + 1 channels.
  - Two bare microphones cannot tell front from back, here as on a real head without shaped ears.

`Hands`: a helper that lifts under the arms (a force on the trunk), as displague/robot's get-up helper does
(its D-073): it lifts only what the feet leave uncarried, up to a cap, steadies the trunk's sideways speed, and
eases off near standing height. Its cap fades as the helped episodes succeed, and comes back if they stop
succeeding. Some episodes are never helped (the unassisted gate), and the helper may quit in the middle of an
episode without warning (displague/robot hub record marques-withdrawal-laws-response-20261111).

Both are numpy over the batch; the environment (rl/standin_env.py) owns the physics.
"""
from __future__ import annotations

import math

import numpy as np

BANDS = 8                      # frequency bands per ear
SOUND_C = 343.0                # m/s
TICK_S = 0.02                  # the word templates' frame (the control tick at 50 Hz)
SYLLABLE_S = 0.14              # a syllable's length at tempo 1
E_REF = 0.01                   # the energy the log observation is scaled by (about the ambient floor)
ITD_SCALE = 0.5e-3             # s: the time difference channel is ITD / this (0.44 ms across a 0.15 m head)
SHADOW_DB = np.linspace(0.0, 12.0, BANDS)       # the far ear's loss, low band to high, for a source at one side
SELF_PROFILE = np.exp(-0.5 * ((np.arange(BANDS) - 2.0) / 1.5) ** 2)   # the drives' noise sits in the low-mid bands

HISS = "hiss"                  # a syllable without formants: energy in the top bands
# word -> syllables; a voiced syllable is its two formant peaks' places along the bands
WORDS = {
    "get_up": ((1.5, 5.0), (2.5, 3.5)),
    "stand_up": (HISS, (2.8, 4.5), (2.5, 3.5)),         # shares "up" with get_up
    "come_here": ((2.2, 3.2), (1.0, 6.0)),
    "stay": (HISS, (1.8, 5.5)),
    "lean": ((1.0, 6.2), (1.0, 6.2)),
    "push": ((1.4, 2.6), HISS),
    "tall": ((2.9, 3.6), (2.0, 2.8)),
    "easy": ((1.0, 6.0), HISS, (1.0, 6.0)),
    "good": ((1.4, 2.4), (1.4, 2.4)),
}
WORD_NAMES = tuple(WORDS)
RISE_COMMANDS = ("get_up", "stand_up")       # either means: rise


def _syllable(s):
    k = np.arange(BANDS, dtype=float)
    if s == HISS:
        return 0.6 / (1.0 + np.exp(-(k - 5.0) * 2.0))
    return sum(np.exp(-0.5 * ((k - f) / 0.7) ** 2) for f in s)


def word_template(name):
    """(frames, BANDS) the word's band energies at 1 m, tempo 1 and no pitch shift, one row per TICK_S."""
    n = int(round(SYLLABLE_S / TICK_S))
    env = np.sin(np.linspace(0.0, math.pi, n + 2)[1:-1]) ** 0.5       # each syllable swells and dies
    rows = []
    for s in WORDS[name]:
        rows.extend(env[:, None] * _syllable(s)[None, :])
        rows.append(np.zeros(BANDS))                                   # a short stop between syllables
    return np.array(rows)


TEMPLATES = [word_template(w) for w in WORD_NAMES]
MAX_FRAMES = max(len(t) for t in TEMPLATES)
TEMPLATE_LEN = np.array([len(t) for t in TEMPLATES])
TEMPLATE = np.zeros((len(WORD_NAMES), MAX_FRAMES + 1, BANDS))
for _i, _t in enumerate(TEMPLATES):
    TEMPLATE[_i, :len(_t)] = _t


def _shift(e, s):
    """Band energies (n, BANDS) moved by `s` (n,) bands along the frequency axis (linear interpolation): pitch."""
    k = np.arange(BANDS)[None, :] - s[:, None]
    k0 = np.floor(k).astype(int)
    f = k - k0
    a = np.take_along_axis(e, np.clip(k0, 0, BANDS - 1), axis=1) * ((k0 >= 0) & (k0 < BANDS))
    b = np.take_along_axis(e, np.clip(k0 + 1, 0, BANDS - 1), axis=1) * ((k0 + 1 >= 0) & (k0 + 1 < BANDS))
    return (1.0 - f) * a + f * b


class Voice:
    """The coach's voice for N environments. `guide`: the share of episodes with cues after the command (the
    others hear only the command, repeated); `wait`: (lo, hi) s before the command, during which the coach says
    "stay" (hi beyond the episode: some episodes are never told to rise)."""

    OBS = 2 * BANDS + 1

    def __init__(self, n, rng, guide=1.0, wait=(0.0, 0.0), dist=(1.2, 3.0), height=(1.2, 1.7),
                 loud_db=(-6.0, 6.0), pitch=(-0.5, 0.5), tempo=(0.8, 1.25), ambient=(0.5, 2.0),
                 self_noise=0.02, gap_s=(0.5, 1.2), lean_h=0.6, push_h=0.85):
        self.n, self.rng = n, rng
        self.guide_share, self.wait = guide, wait
        self.dist, self.height, self.loud_db, self.pitch_r, self.tempo_r = dist, height, loud_db, pitch, tempo
        self.ambient_r, self.self_noise, self.gap_s = ambient, self_noise, gap_s
        self.lean_h, self.push_h = lean_h, push_h
        self.wi = {w: i for i, w in enumerate(WORD_NAMES)}
        z = np.zeros
        self.word = np.full(n, -1)                 # the word being spoken, -1: silence
        self.word_t = z(n)                         # frames into it
        self.next_at = z(n)                        # s into the episode: the coach speaks again from then
        self.t_cmd = z(n)                          # s into the episode: the command comes then
        self.cmd = z(n, dtype=int)                 # this episode's command word
        self.told = z(n, dtype=bool)               # the command has begun
        self.praised = z(n, dtype=bool)
        self.guide = z(n, dtype=bool)
        self.pos = z((n, 3))                       # where the coach stands (world)
        self.pitch, self.tempo, self.loud, self.ambient = z(n), np.ones(n), np.ones(n), np.ones(n)
        self.heard = z((n, self.OBS))
        self.mute = False                          # a test: the script runs, but the coach makes no sound

    def reset(self, idx, root_xy):
        """New episodes for envs `idx`, whose roots are at `root_xy` (len(idx), 2)."""
        k, rng = len(idx), self.rng
        u = lambda r: rng.uniform(r[0], r[1], k)
        a, d = rng.uniform(-math.pi, math.pi, k), u(self.dist)
        self.pos[idx, 0] = root_xy[:, 0] + d * np.cos(a)
        self.pos[idx, 1] = root_xy[:, 1] + d * np.sin(a)
        self.pos[idx, 2] = u(self.height)
        self.pitch[idx], self.tempo[idx] = u(self.pitch_r), u(self.tempo_r)
        self.loud[idx] = 10.0 ** (u(self.loud_db) / 10.0)
        self.ambient[idx] = E_REF * u(self.ambient_r)
        self.word[idx], self.word_t[idx] = -1, 0.0
        self.t_cmd[idx] = u(self.wait)
        self.next_at[idx] = rng.uniform(0.0, 0.3, k)
        self.cmd[idx] = [self.wi[RISE_COMMANDS[i]] for i in rng.integers(0, len(RISE_COMMANDS), k)]
        self.told[idx] = False
        self.praised[idx] = False
        self.guide[idx] = rng.random(k) < self.guide_share
        self.heard[idx] = 0.0

    def speak(self, t, height, stood, over):
        """Advance the script one tick. `t` (n,) s into each episode; `height` the root's height over its standing
        height; `stood` has stood for a second; `over` a gated line is above its rating."""
        w = self.wi
        speaking = self.word >= 0
        self.word_t[speaking] += self.tempo[speaking]
        ended = speaking & (self.word_t >= TEMPLATE_LEN[np.maximum(self.word, 0)])
        self.word[ended] = -1
        due = (self.word < 0) & (t >= self.next_at)
        if not due.any():
            return
        before = t < self.t_cmd
        cue = np.where(over, w["easy"], np.where(height < self.lean_h, w["lean"],
                                                 np.where(height < self.push_h, w["push"], w["tall"])))
        after = np.where(stood, np.where(self.praised, -1, w["good"]), np.where(self.guide, cue, self.cmd))
        say = np.where(before, w["stay"] if self.wait[1] > 0.0 else -1, np.where(self.told, after, self.cmd))
        i = np.nonzero(due)[0]
        self.word[i], self.word_t[i] = say[i], 0.0
        self.praised[i] |= (say[i] == w["good"])
        self.told[i] |= ~before[i]
        dur = TEMPLATE_LEN[np.maximum(say[i], 0)] * TICK_S / self.tempo[i]
        gap = self.rng.uniform(self.gap_s[0], self.gap_s[1], len(i))
        # the wait ends on time: a "stay" never runs past the command by more than its own length
        self.next_at[i] = np.where(before[i], np.minimum(t[i] + dur + gap, np.maximum(self.t_cmd[i], t[i] + dur)),
                                   t[i] + dur + gap)

    def source(self):
        """(n, BANDS) the voice's band energies now, at 1 m."""
        wd = np.maximum(self.word, 0)
        f = np.clip(self.word_t, 0.0, MAX_FRAMES - 1e-6)
        i0 = np.floor(f).astype(int)
        fr = (f - i0)[:, None]
        e = (1.0 - fr) * TEMPLATE[wd, i0] + fr * TEMPLATE[wd, i0 + 1]
        return _shift(e, self.pitch) * (self.loud * ((self.word >= 0) & (not self.mute)))[:, None]

    def hear(self, ear_l, ear_r, joint_speed):
        """The observation (n, 2 BANDS + 1) at ears `ear_l`, `ear_r` (n, 3; world), with the joints' rms speed
        `joint_speed` (n,) rad/s for the robot's own noise."""
        src = self.source()
        mid = 0.5 * (ear_l + ear_r)
        axis = ear_l - ear_r
        axis /= np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-6)
        to = self.pos - mid
        lateral = np.einsum("ni,ni->n", to, axis) / np.maximum(np.linalg.norm(to, axis=1), 1e-6)   # +1: at the left
        r_l = np.maximum(np.linalg.norm(self.pos - ear_l, axis=1), 0.2)
        r_r = np.maximum(np.linalg.norm(self.pos - ear_r, axis=1), 0.2)
        shade = lambda far: 10.0 ** (-SHADOW_DB[None, :] * np.clip(far, 0.0, 1.0)[:, None] / 10.0)
        noise = self.ambient[:, None] + self.self_noise * (joint_speed ** 2)[:, None] * SELF_PROFILE[None, :]
        jit = lambda: np.exp(0.2 * self.rng.standard_normal((self.n, BANDS)))
        e_l = src * shade(-lateral) / (r_l ** 2)[:, None] + noise * jit()
        e_r = src * shade(lateral) / (r_r ** 2)[:, None] + noise * jit()
        snr = src.sum(axis=1) / (np.minimum(r_l, r_r) ** 2) / noise.sum(axis=1)
        present = snr / (1.0 + snr)                 # how far the voice stands out of the noise
        itd = (r_r - r_l) / SOUND_C / ITD_SCALE     # positive: the left ear hears it first
        itd = present * itd + (1.0 - present) * 0.3 * self.rng.standard_normal(self.n)
        self.heard = np.concatenate([np.log1p(e_l / E_REF), np.log1p(e_r / E_REF), itd[:, None]], axis=1)
        return self.heard


class Hands:
    """A helper lifting the trunk of each of N robots of weight `weight` (N). `cap` (of the weight) is the most it
    lifts at level 1. Each of the task's `starts` (the get-up curriculum's half-kneel, fallen, ...) has its own
    level: it fades by `fade` whenever that start's helped episodes succeed more often than `fade_at` (an average
    over at least `min_episodes` of them), and is raised again when they succeed less often than `back_at`. So a
    harder start being unlocked does not bring the help back on an easier one."""

    def __init__(self, n, rng, weight, cap=0.6, level=1.0, free=0.25, quit_p=0.2, fade=0.8, fade_at=0.6,
                 back_at=0.25, min_episodes=300, damp=60.0, ease_m=0.10, floor=0.03, starts=1):
        self.n, self.rng, self.weight = n, rng, weight
        self.cap, self.free, self.quit_p = cap, free, quit_p
        self.level = np.full(starts, float(level))
        self.fade, self.fade_at, self.back_at, self.min_episodes = fade, fade_at, back_at, min_episodes
        self.damp, self.ease_m, self.floor = damp, ease_m, floor
        self.start = np.zeros(n, dtype=int)        # the start each env's episode began from
        self.helped = np.zeros(n, dtype=bool)
        self.cap_env = np.zeros(n)                 # N: this episode's cap
        self.level_env = np.zeros(n)               # the level this episode was given at
        self.quit_at = np.full(n, np.inf)          # s: the helper lets go then
        self.impulse = np.zeros(n)                 # N s given this episode
        self.lift = np.zeros(n)
        self.ema = {k: np.zeros(starts) for k in ("helped", "free", "impulse")}
        self.count = {k: np.zeros(starts, dtype=int) for k in ("helped", "free", "at_level")}

    def record(self, idx, success):
        """Fold finished episodes into their start's averages, and move its level."""
        for k, ok in zip(idx, success):
            s = self.start[k]
            key = "helped" if self.helped[k] else "free"
            self.count[key][s] += 1
            a = max(0.005, 1.0 / self.count[key][s])
            self.ema[key][s] += a * (float(ok) - self.ema[key][s])
            if self.helped[k]:
                self.count["at_level"][s] += 1
                b = max(0.005, 1.0 / self.count["at_level"][s])
                self.ema["impulse"][s] += b * (self.impulse[k] / self.weight - self.ema["impulse"][s])
        for s in range(len(self.level)):
            if self.level[s] <= 0.0 or self.count["at_level"][s] < self.min_episodes:
                continue
            if self.ema["helped"][s] > self.fade_at:
                self.level[s] = self.level[s] * self.fade if self.level[s] * self.fade >= self.floor else 0.0
                self.count["at_level"][s] = self.count["helped"][s] = 0      # a new level is judged afresh
            elif self.ema["helped"][s] < self.back_at and self.level[s] < 1.0:
                self.level[s] = min(1.0, self.level[s] / math.sqrt(self.fade))
                self.count["at_level"][s] = self.count["helped"][s] = 0

    def reset(self, idx, episode_s, start=None):
        """New episodes for envs `idx`, beginning from `start` (len(idx),; None: the only start)."""
        k = len(idx)
        self.start[idx] = 0 if start is None else start
        lv = self.level[self.start[idx]]
        self.level_env[idx] = lv
        self.helped[idx] = (self.rng.random(k) >= self.free) & (lv > 0.0)
        # the helper's strength differs from one episode to the next, and it may walk away part-way
        self.cap_env[idx] = self.cap * lv * self.weight * self.rng.uniform(0.6, 1.0, k)
        quits = self.rng.random(k) < self.quit_p
        self.quit_at[idx] = np.where(quits, self.rng.uniform(0.1, 1.0, k) * episode_s, np.inf)
        self.impulse[idx] = 0.0

    def force(self, t, z, z_stand, feet_load, v_xy, dt):
        """(n, 3) N on the trunk: `t` s into the episode, `z` the root's height, `z_stand` its standing height,
        `feet_load` (n,) N on the soles, `v_xy` (n, 2) the trunk's horizontal velocity."""
        on = self.helped & (t < self.quit_at)
        ease = np.clip((0.98 * z_stand - z) / self.ease_m, 0.0, 1.0)
        lift = np.where(on, np.minimum(self.cap_env, np.maximum(self.weight - feet_load, 0.0)) * ease, 0.0)
        steady = -(self.damp * self.level_env * (on & (ease > 0.0)))[:, None] * v_xy
        self.lift = lift
        self.impulse += (lift + np.linalg.norm(steady, axis=1)) * dt
        return np.concatenate([steady, lift[:, None]], axis=1)

    def summary(self):
        """The first start's figures as scalars (the log's columns), and every start's under `by_start`."""
        f = lambda a: [float(x) for x in a]
        return dict(level=float(self.level[0]), helped_success=float(self.ema["helped"][0]),
                    free_success=float(self.ema["free"][0]), impulse_weight_s=float(self.ema["impulse"][0]),
                    by_start=dict(level=f(self.level), helped_success=f(self.ema["helped"]),
                                  free_success=f(self.ema["free"]), impulse_weight_s=f(self.ema["impulse"]),
                                  helped_episodes=[int(x) for x in self.count["helped"]],
                                  free_episodes=[int(x) for x in self.count["free"]]))

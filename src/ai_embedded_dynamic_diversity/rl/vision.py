"""Eyes for the stand-in's tasks: retinas at several scales and wavelengths, overlaid or apart.

Nothing here is an image. As rl/coach.py's voice is what a cochlea gives at the control rate, this is what a
retina gives: the responses of a few receptors, each looking along one direction with an angular size, computed
from the scene's geometry (a floor, a sky, a light level, and the trainer's body as a set of coloured balls).

Three eyes on the head (displague/robot tools/export_mjcf.EYES; its docs/50):

- `L` and `R`, a pair 60 mm apart that see the same light, each with two receptor sheets overlaid on one retina, as
  in a human eye:
    - a periphery: wide and coarse (`PERIPHERY`), brightness only, as rods give;
    - a fovea: narrow and fine at the centre of gaze (`FOVEA`), brightness and two colour-opponent channels
      (red against green, blue against yellow), as cones give. Colour needs light: its noise grows as the light falls.
  Each eye adapts to the light (its brightness channels are relative to its own mean). The two eyes differ only by
  their baseline, so depth is in the difference between them (disparity).
- `C`, a third eye above them that measures a different property: range (`RANGE`), as an active time-of-flight
  sensor gives, in the near infrared. It does not depend on the scene's light, and it has no colour.

`mode`: "pair" (the two overlaid eyes), "third" (the range eye alone: the property apart from the others), or
"three" (all of them).

The environment (rl/standin_env.py) owns the physics and tells the eyes where they are, in numpy; the render
itself runs in torch, on the GPU when there is one: at 2,048 robots it took twice as long as the physics in numpy.
"""
from __future__ import annotations

import math

import numpy as np
import torch

MODES = ("pair", "third", "three")
# (columns, rows, horizontal field, vertical field) in degrees
PERIPHERY = (5, 3, 120.0, 72.0)
FOVEA = (3, 3, 24.0, 24.0)
RANGE = (4, 3, 80.0, 60.0)
RANGE_MAX = 6.0                # m: beyond this the range eye reads "far"
FLOOR_RGB = np.array([0.45, 0.40, 0.34])
SKY_RGB = np.array([0.74, 0.82, 0.95])
TILE_M = 1.0                   # the floor's pattern repeats every this far
TILE_CONTRAST = 0.35
EYE_OFFSETS = {"L": (0.03, 0.0, 0.0), "R": (-0.03, 0.0, 0.0), "C": (0.0, 0.0, 0.04)}   # left, forward, up: a body
                                                                                     # without eye sensors


def grid(cols, rows, fov_h, fov_v):
    """Receptor directions (K, 3) as (forward, left, up) components, and each one's angular size (K,) rad: half
    the spacing between neighbours."""
    az = np.radians(np.linspace(-0.5, 0.5, cols + 1)[:-1] * fov_h + 0.5 * fov_h / cols)
    el = np.radians(np.linspace(-0.5, 0.5, rows + 1)[:-1] * fov_v + 0.5 * fov_v / rows)
    a, e = np.meshgrid(-az, -el)           # rows from the top, columns from the left, as an image reads
    d = np.stack([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)], axis=-1).reshape(-1, 3)
    size = 0.5 * math.radians(0.5 * (fov_h / cols + fov_v / rows))
    return d, np.full(len(d), size)


class Scene:
    """What there is to see, per environment: the light, the floor's pattern, and balls (the trainer's body)."""

    def __init__(self, n, rng, light=(0.08, 1.0), balls=0):
        self.n, self.rng, self.light_r = n, rng, light
        self.light = np.ones(n)
        self.tile_phase = np.zeros((n, 2))
        self.tile_turn = np.zeros(n)
        self.ball_pos = np.zeros((n, balls, 3))
        self.ball_r = np.zeros(balls)
        self.ball_rgb = np.zeros((balls, 3))
        self.shown = np.ones(n, dtype=bool)        # False: the balls are not there

    def reset(self, idx):
        k = len(idx)
        lo, hi = self.light_r
        self.light[idx] = np.exp(self.rng.uniform(math.log(lo), math.log(hi), k))    # dusk to daylight
        self.tile_phase[idx] = self.rng.uniform(0.0, TILE_M, (k, 2))
        self.tile_turn[idx] = self.rng.uniform(0.0, math.pi, k)


def render(scene, eye, R, dirs, size, device="cpu"):
    """What receptors `dirs` (K, 3; head frame) of angular size `size` (K,) see, each from its own place `eye`
    (n, K, 3; world), with the head's orientation `R` (n, 3, 3; head to world): (rgb (n, K, 3) before the light,
    range (n, K) m), torch tensors on `device`."""
    t32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
    eye, R, dirs, size = t32(eye), t32(R), t32(dirs), t32(size)
    w = torch.einsum("nij,kj->nki", R, dirs)                                 # (n, K, 3) world
    wz = w[:, :, 2]
    below = torch.sigmoid(-wz / (0.6 * size[None, :]))                       # the share looking at the floor
    t = torch.clamp(torch.clamp(eye[:, :, 2], min=0.02) / torch.clamp(-wz, min=0.02), max=40.0)   # m to the floor
    hit = eye[:, :, 0:2] + w[:, :, 0:2] * t[:, :, None]
    turn, phase = t32(scene.tile_turn), t32(scene.tile_phase)
    c, s = torch.cos(turn)[:, None], torch.sin(turn)[:, None]
    hx = c * hit[:, :, 0] + s * hit[:, :, 1] + phase[:, 0:1]
    hy = -s * hit[:, :, 0] + c * hit[:, :, 1] + phase[:, 1:2]
    k = 2.0 * math.pi / TILE_M
    blur = torch.exp(-0.5 * (t * size[None, :] * k) ** 2)                    # a far tile is smaller than the receptor
    tile = 1.0 + TILE_CONTRAST * torch.cos(k * hx) * torch.cos(k * hy) * blur
    rgb = below[:, :, None] * tile[:, :, None] * t32(FLOOR_RGB) + (1.0 - below[:, :, None]) * t32(SKY_RGB)
    rng_m = below * torch.clamp(t, max=RANGE_MAX) + (1.0 - below) * RANGE_MAX
    if scene.ball_pos.shape[1]:
        v = t32(scene.ball_pos)[:, None, :, :] - eye[:, :, None, :]          # (n, K, J, 3)
        dist = torch.clamp(torch.linalg.norm(v, dim=3), min=0.05)
        cos = (v * w[:, :, None, :]).sum(dim=3) / dist
        th2 = 2.0 * (1.0 - cos)                                              # the angle off the receptor's axis, squared
        rho2 = (t32(scene.ball_r)[None, None, :] / dist) ** 2                # the ball's angular radius, squared
        s2 = (size ** 2)[None, :, None]
        cover = torch.clamp(rho2 / s2, max=1.0) * torch.exp(-0.5 * th2 / (s2 + rho2))
        cover = cover * t32(scene.shown)[:, None, None]
        total = 1.0 - torch.prod(1.0 - cover, dim=2)                         # the share of the receptor the balls fill
        wsum = cover.sum(dim=2) + 1e-9
        ball_rgb = torch.einsum("nkj,jc->nkc", cover, t32(scene.ball_rgb)) / wsum[:, :, None]
        ball_rng = (cover * dist).sum(dim=2) / wsum
        rgb = (1.0 - total[:, :, None]) * rgb + total[:, :, None] * ball_rgb
        rng_m = (1.0 - total) * rng_m + total * torch.clamp(ball_rng, max=RANGE_MAX)
    return rgb, rng_m


class Eyes:
    """The eyes of N robots. `axes`: the head's (forward, left, up) in its own frame (the robot's spec `eyes`)."""

    def __init__(self, n, rng, mode="three", axes=((0, -1, 0), (1, 0, 0), (0, 0, 1)), noise=0.04, light=(0.08, 1.0),
                 device=None):
        if mode not in MODES:
            raise ValueError("eyes must be one of %s, got %r" % (", ".join(MODES), mode))
        self.n, self.rng, self.mode, self.noise = n, rng, mode, noise
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(rng.integers(0, 2 ** 31)))
        self.axes = np.array(axes, dtype=float).T                            # columns: forward, left, up
        self.scene = Scene(n, rng, light)
        head = lambda g: (g[0] @ self.axes.T, g[1])
        self.periphery, self.fovea, self.range = head(grid(*PERIPHERY)), head(grid(*FOVEA)), head(grid(*RANGE))
        self.kp, self.kf, self.kr = len(self.periphery[0]), len(self.fovea[0]), len(self.range[0])
        self.pair = mode in ("pair", "three")
        self.third = mode in ("third", "three")
        self.per_eye = self.kp + 3 * self.kf
        self.OBS = (2 * self.per_eye if self.pair else 0) + (self.kr if self.third else 0)
        # every receptor of every eye in one render: which eye each belongs to, its direction and its size
        one = (np.concatenate([self.periphery[0], self.fovea[0]]), np.concatenate([self.periphery[1], self.fovea[1]]))
        sets = ([("L", one), ("R", one)] if self.pair else []) + ([("C", self.range)] if self.third else [])
        self.owner = [e for e, (d, _) in sets for _ in d]
        self.dirs = np.concatenate([d for _, (d, _) in sets])
        self.size = np.concatenate([z for _, (_, z) in sets])
        self.seen = np.zeros((n, self.OBS))
        self.shut = False                          # a test: the eyes see nothing (noise only)

    def reset(self, idx):
        self.scene.reset(idx)

    def offsets(self):
        """The three eyes' places from the point between the pair, in the head's frame (a body without cameras)."""
        return {k: self.axes @ np.array([v[1], v[0], v[2]]) for k, v in EYE_OFFSETS.items()}

    def _pair(self, rgb, light):
        """One of the pair from its receptors' light (n, kp + kf, 3): (n, kp + 3 kf): the periphery's brightness,
        then the fovea's brightness, red against green and blue against yellow."""
        g = lambda shape: torch.randn(shape, generator=self.gen, device=self.device)
        rgb = rgb * light[:, :, None]
        # photon noise: relative noise grows as the light falls
        rgb = torch.clamp(rgb * (1.0 + self.noise / torch.sqrt(light)[:, :, None] * g(rgb.shape)), min=0.0)
        if self.shut:
            rgb = 0.01 * torch.abs(g(rgb.shape))
        lum = rgb.mean(dim=2)
        adapt = lum[:, :self.kp].mean(dim=1, keepdim=True) + 0.01            # the eye adapts to its own mean
        r, gr, b = (rgb[:, self.kp:, i] for i in range(3))
        dim = 0.02 / light                                                   # cones need light
        rg = (r - gr) / (r + gr + dim)
        by = (b - 0.5 * (r + gr)) / (b + 0.5 * (r + gr) + dim)
        return torch.cat([lum[:, :self.kp] / adapt - 1.0, lum[:, self.kp:] / adapt - 1.0, rg, by], dim=1)

    def _third(self, rng_m):
        """The range eye from its receptors' range (n, kr) m: nearness, 1 at the eye and 0 at RANGE_MAX or beyond."""
        rng_m = rng_m * (1.0 + 0.03 * torch.randn(rng_m.shape, generator=self.gen, device=self.device))
        if self.shut:
            rng_m = torch.full_like(rng_m, RANGE_MAX)
        return 1.0 - torch.clamp(rng_m, 0.0, RANGE_MAX) / RANGE_MAX

    def see(self, pos, R):
        """The observation (n, OBS). `pos`: {"L", "R", "C"} -> (n, 3) world; `R` (n, 3, 3) the head's orientation."""
        eye = np.stack([pos[e] for e in self.owner], axis=1)                 # (n, K, 3)
        rgb, rng_m = render(self.scene, eye, R, self.dirs, self.size, self.device)
        light = torch.as_tensor(self.scene.light, dtype=torch.float32, device=self.device)[:, None]
        parts, k = [], self.kp + self.kf
        if self.pair:
            parts += [self._pair(rgb[:, :k], light), self._pair(rgb[:, k:2 * k], light)]
        if self.third:
            parts.append(self._third(rng_m[:, -self.kr:]))
        self.seen = torch.cat(parts, dim=1).cpu().numpy().astype(np.float64)
        return self.seen

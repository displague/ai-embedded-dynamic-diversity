"""Behaviour cloning from the robot's algorithmic controller (displague/robot tools/record_teacher.py).

The teacher is the hand-written controller of the robot repository (capture-point balance, weight shift,
one-foot hold, the static step) running on the full tendon model. Its demonstrations hold, at 50 Hz, the
stand-in observation without the previous-action channels (72) and the joint target offsets it applied, in
radians (32). For a given environment the loader rebuilds the 104-channel observation and the action by
dividing the offsets by that environment's action span, so the cloned policy speaks the environment's units.

Cloning is sequence regression: windows of `window` steps per episode, the core unrolled from an empty memory
at the window start, mean squared error between the policy mean and the teacher's action. The observation
normaliser is fitted to the teacher data and handed to PPO, so the policy sees the same scaling after cloning.
"""
from __future__ import annotations

import numpy as np
import torch

from ai_embedded_dynamic_diversity.rl.actor import CoreActor, RunningNorm


def load_teacher(path: str, half: np.ndarray) -> dict:
    z = np.load(path, allow_pickle=False)
    off, obs72, ep = z["offset"], z["obs"], z["episode"]
    act = np.clip(off / half, -1.0, 1.0).astype(np.float32)
    prev = np.zeros_like(act)
    prev[1:] = act[:-1]
    starts = np.r_[True, ep[1:] != ep[:-1]]
    prev[starts] = 0.0
    obs = np.concatenate([obs72, prev], axis=1).astype(np.float32)
    return dict(obs=obs, act=act, episode=ep, keys=list(z["keys"]))


def clone(actor: CoreActor, obs_norm: RunningNorm, data: dict, device, epochs: int = 30, window: int = 64,
          batch: int = 64, lr: float = 1e-3, seed: int = 0) -> list[dict]:
    """Fit the actor's mean to the teacher's actions; returns one record per epoch."""
    rng = np.random.default_rng(seed)
    obs = torch.as_tensor(data["obs"], device=device)
    act = torch.as_tensor(data["act"], device=device)
    obs_norm.update(obs)
    on = obs_norm(obs)
    ep = data["episode"]
    # windows that stay inside one episode
    starts = []
    for e in np.unique(ep):
        idx = np.nonzero(ep == e)[0]
        for s in range(idx[0], idx[-1] - window + 2, window // 2):
            starts.append(s)
    starts = np.array(starts)
    opt = torch.optim.Adam(actor.core.parameters(), lr=lr)
    log = []
    for epoch in range(epochs):
        rng.shuffle(starts)
        losses = []
        for b in range(0, len(starts), batch):
            s = torch.as_tensor(starts[b:b + batch], device=device)
            ix = s[:, None] + torch.arange(window, device=device)[None, :]
            o, a = on[ix], act[ix]                               # (B, W, D)
            mem = actor.initial_state(len(s), device)
            preds = []
            for t in range(window):
                m, mem = actor(o[:, t], mem)
                preds.append(m)
            pred = torch.stack(preds, dim=1)
            loss = (pred - a).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.core.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        rec = dict(epoch=epoch, mse=float(np.mean(losses)), windows=len(starts))
        log.append(rec)
        if epoch % 5 == 0 or epoch == epochs - 1:
            print("[bc] epoch %3d mse %.5f (%d windows)" % (epoch, rec["mse"], len(starts)), flush=True)
    return log

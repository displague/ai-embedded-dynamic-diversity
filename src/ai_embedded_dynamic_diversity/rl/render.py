"""Render a trained policy on a stand-in body to a GIF (visual evidence for docs and releases).

    python -m ai_embedded_dynamic_diversity.rl.render artifacts/rl/stand-grown-bc.pt --body tendon32 \
        --seconds 8 --out docs/assets/rl-tendon32-stand.gif

One environment, the nominal model (no randomisation), the policy's mean action; pushes as in the stand task.
Frames from a camera tracking the root body; a caption with the time, pelvis height and push markers.
"""
from __future__ import annotations

import numpy as np
import torch
import typer

from ai_embedded_dynamic_diversity.models.core import ModelCore
from ai_embedded_dynamic_diversity.rl.actor import CoreActor, RunningNorm, load_checkpoint
from ai_embedded_dynamic_diversity.rl.standin_env import EnvConfig, StandinEnv

app = typer.Typer(add_completion=False)


def load(path: str):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg, state = load_checkpoint(path)
    core = ModelCore(**cfg)
    core.load_state_dict(state)
    actor = CoreActor(core).eval()
    norm = RunningNorm(core.signal_dim)
    norm.load_state_dict(saved["obs_norm"])
    return actor, norm, saved


@app.command()
def main(policy: str, body: str = "tendon32", task: str = "stand", seconds: float = 8.0, out: str = "policy.gif",
         fps: int = 25, width: int = 480, height: int = 360, seed: int = 1, label: str = ""):
    import mujoco
    from PIL import Image, ImageDraw

    actor, norm, saved = load(policy)
    env = EnvConfig(task=task, body=body, num_envs=1, variants=1, nthread=1, seed=seed,
                    pad_dofs=saved["env"].get("pad_dofs", 32) or 32, max_delay_steps=0,
                    action_fraction=saved["env"].get("action_fraction"))
    e = StandinEnv(env)
    e.reset()
    obs, _ = e.observe()
    m = e.base
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body(e.root).id
    cam.distance, cam.elevation, cam.azimuth = 2.6, -12.0, 135.0
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0            # the sensor sites (sole, impact) are not part of the body
    mem = actor.initial_state(1, "cpu")
    frames, every = [], max(1, int(round(1.0 / (fps * e.dt_ctrl))))
    pushed_at = -10.0
    for k in range(int(seconds / e.dt_ctrl)):
        with torch.no_grad():
            mean, mem = actor(norm(torch.as_tensor(obs)), mem)
        before = e.push_at.copy()
        obs, _, _, done, info = e.step(mean.numpy())
        if e.push_at[0] != before[0]:
            pushed_at = k * e.dt_ctrl
        if done[0]:
            mem = actor.initial_state(1, "cpu")
        if k % every:
            continue
        d.qpos[:] = e.state[0, e.i_qpos:e.i_qpos + e.nq]
        d.qvel[:] = e.state[0, e.i_qvel:e.i_qvel + e.nv]
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam, scene_option=opt)
        img = Image.fromarray(r.render())
        draw = ImageDraw.Draw(img)
        t = k * e.dt_ctrl
        text = "%s  t=%.1f s  pelvis %.2f m%s" % (label or body, t, float(info.extras["z"][0]),
                                                 "  PUSH" if t - pushed_at < 0.4 else "")
        draw.rectangle([0, 0, width, 22], fill=(0, 0, 0))
        draw.text((6, 5), text, fill=(255, 255, 255))
        frames.append(img)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    e.close()
    print({"gif": out, "frames": len(frames)})


if __name__ == "__main__":
    app()

from __future__ import annotations

import torch
import typer

from ai_embedded_dynamic_diversity.config import WorldConfig
from ai_embedded_dynamic_diversity.sim.embodiments import embodiment_dof_table
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld

app = typer.Typer(add_completion=False)


@app.command()
def rollout(steps: int = 10, batch_size: int = 2, device: str = "cpu") -> None:
    cfg = WorldConfig()
    world = DynamicDiversityWorld(cfg.x, cfg.y, cfg.z, cfg.resource_channels, cfg.decay, device=device)
    state = world.init(batch_size)
    action = torch.zeros(batch_size, cfg.x * cfg.y * cfg.z, device=device)
    for _ in range(steps):
        state = world.step(state, action)
    obs = world.encode_observation(state, signal_dim=32)
    print({"obs_shape": tuple(obs.shape), "life_mean": state.life.mean().item(), "stress_mean": state.stress.mean().item()})


@app.command()
def embodiments() -> None:
    rows = embodiment_dof_table()
    print({"embodiments": rows})


if __name__ == "__main__":
    app()

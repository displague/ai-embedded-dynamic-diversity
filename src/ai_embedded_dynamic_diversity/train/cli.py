from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import typer
from rich import print
from torch import nn

from ai_embedded_dynamic_diversity.config import TrainConfig, WorldConfig, model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld
from ai_embedded_dynamic_diversity.train.losses import loss_fn

app = typer.Typer(add_completion=False)


def choose_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_gradient_epoch(
    model: ModelCore,
    world: DynamicDiversityWorld,
    mcfg,
    tcfg: TrainConfig,
    opt: torch.optim.Optimizer,
    wcfg: WorldConfig,
    dev: torch.device,
    remap_probability: float,
    env_volatility: float,
    genetic_memory: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor, float]:
    state = world.init(tcfg.batch_size)
    if genetic_memory is None:
        memory = model.init_memory(tcfg.batch_size, mcfg.memory_slots, mcfg.memory_dim, dev)
    else:
        memory = genetic_memory.repeat(tcfg.batch_size, 1, 1).detach()
    epoch_loss = 0.0
    step_times = []
    for step_index in range(tcfg.unroll_steps):
        t0 = time.perf_counter()
        obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)

        remap_code = torch.zeros(tcfg.batch_size, mcfg.max_remap_groups, device=dev)
        if random.random() < remap_probability:
            group = random.randrange(mcfg.max_remap_groups)
            remap_code[:, group] = 1.0

        out = model(obs, memory, remap_code)
        memory = out["memory"].detach()

        target = torch.cat([obs[:, : mcfg.io_channels // 2], out["readiness"][:, : mcfg.io_channels - (mcfg.io_channels // 2)]], dim=1)
        loss, logs = loss_fn(
            out,
            target,
            entropy_weight=tcfg.entropy_weight,
            energy_weight=tcfg.energy_weight,
            memory_consistency_weight=tcfg.memory_consistency_weight,
        )
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss += logs["loss"]

        action_field = out["io"].detach().mean(dim=1, keepdim=True).repeat(1, wcfg.x * wcfg.y * wcfg.z)
        controls = world.random_controls(tcfg.batch_size, env_volatility, step_index=step_index)
        state = world.step(state, action_field, controls=controls)
        step_times.append((time.perf_counter() - t0) * 1000.0)
    final_memory = memory.mean(dim=0, keepdim=True).detach()
    mean_step_ms = sum(step_times) / max(1, len(step_times))
    return epoch_loss / max(1, tcfg.unroll_steps), final_memory, mean_step_ms


@torch.no_grad()
def evaluate_fitness(
    model: ModelCore,
    world: DynamicDiversityWorld,
    mcfg,
    wcfg: WorldConfig,
    dev: torch.device,
    steps: int,
    batch: int,
    env_volatility: float,
) -> float:
    model.eval()
    state = world.init(batch)
    memory = model.init_memory(batch, mcfg.memory_slots, mcfg.memory_dim, dev)
    energy_total = 0.0
    remap_events = 0.0
    for step_index in range(steps):
        obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)
        remap = torch.zeros(batch, mcfg.max_remap_groups, device=dev)
        if random.random() < 0.3:
            remap[:, random.randrange(mcfg.max_remap_groups)] = 1.0
            remap_events += 1.0
        out = model(obs, memory, remap)
        memory = out["memory"]
        action_field = out["io"].mean(dim=1, keepdim=True).repeat(1, wcfg.x * wcfg.y * wcfg.z)
        controls = world.random_controls(batch, env_volatility, step_index=step_index)
        state = world.step(state, action_field, controls=controls)
        energy_total += float(out["energy"].mean().item())
    vitality = float(state.life.mean().item())
    stress = float(state.stress.mean().item())
    energy = energy_total / max(1, steps)
    remap_bonus = remap_events / max(1.0, steps)
    # Higher is better: survive under stress using less energy.
    return vitality - 0.45 * stress - 0.1 * energy + 0.05 * remap_bonus


def mutate_from_parent(parent: ModelCore, mutation_std: float) -> ModelCore:
    child = copy.deepcopy(parent)
    if mutation_std > 0.0:
        with torch.no_grad():
            for param in child.parameters():
                noise = torch.randn_like(param) * mutation_std
                param.add_(noise)
    return child


@app.command()
def run(
    epochs: int = 20,
    batch_size: int = 16,
    unroll_steps: int = 12,
    lr: float = 2e-4,
    device: str = "cuda",
    profile: str = "base",
    gating_mode: str = "sigmoid",
    topk_gating: int = 0,
    enable_dmd_gating: bool = False,
    enable_phase_gating: bool = False,
    coevolution: bool = False,
    population_size: int = 4,
    elite_fraction: float = 0.5,
    mutation_std: float = 0.01,
    enable_curriculum: bool = False,
    remap_probability_start: float = 0.1,
    remap_probability_end: float = 0.35,
    env_volatility_start: float = 0.05,
    env_volatility_end: float = 0.55,
    genetic_memory_decay: float = 0.92,
    enable_genetic_memory: bool = False,
    seed: int = 7,
    metrics_path: str = "",
    save_path: str = "artifacts/model-core.pt",
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    mcfg = model_config_for_profile(profile)
    mcfg.gating_mode = gating_mode
    mcfg.topk_gating = topk_gating
    mcfg.enable_dmd_gating = enable_dmd_gating
    mcfg.enable_phase_gating = enable_phase_gating
    wcfg = WorldConfig()
    tcfg = TrainConfig(epochs=epochs, batch_size=batch_size, unroll_steps=unroll_steps, lr=lr, device=device)

    dev = choose_device(tcfg.device)
    world = DynamicDiversityWorld(wcfg.x, wcfg.y, wcfg.z, wcfg.resource_channels, wcfg.decay, device=str(dev))
    genetic_memory = torch.zeros(1, mcfg.memory_slots, mcfg.memory_dim, device=dev) if enable_genetic_memory else None

    def _lin_schedule(start: float, end: float, epoch_idx: int) -> float:
        if tcfg.epochs <= 1:
            return end
        alpha = epoch_idx / (tcfg.epochs - 1)
        return (1.0 - alpha) * start + alpha * end

    if not metrics_path:
        metrics_path = str(Path(save_path).with_suffix(".metrics.json"))
    metrics_records: list[dict] = []
    run_flags = {
        "profile": profile,
        "gating_mode": gating_mode,
        "topk_gating": topk_gating,
        "enable_dmd_gating": enable_dmd_gating,
        "enable_phase_gating": enable_phase_gating,
        "coevolution": coevolution,
        "enable_curriculum": enable_curriculum,
        "enable_genetic_memory": enable_genetic_memory,
        "genetic_memory_decay": genetic_memory_decay,
    }

    if not coevolution:
        model = ModelCore(**asdict(mcfg)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        for epoch in range(tcfg.epochs):
            if enable_curriculum:
                remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, epoch)
                env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, epoch)
            else:
                remap_probability = tcfg.remap_probability
                env_volatility = 0.0
            mean_loss, memory_snapshot, mean_step_ms = run_gradient_epoch(
                model,
                world,
                mcfg,
                tcfg,
                opt,
                wcfg,
                dev,
                remap_probability=remap_probability,
                env_volatility=env_volatility,
                genetic_memory=genetic_memory,
            )
            if genetic_memory is not None:
                genetic_memory = genetic_memory_decay * genetic_memory + (1.0 - genetic_memory_decay) * memory_snapshot
            fitness = evaluate_fitness(
                model,
                world,
                mcfg,
                wcfg,
                dev,
                steps=max(4, tcfg.unroll_steps // 2),
                batch=max(2, tcfg.batch_size // 4),
                env_volatility=env_volatility,
            )
            metrics_records.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "fitness": fitness,
                    "mean_step_ms": mean_step_ms,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                }
            )
            print(
                {
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "fitness": fitness,
                    "mean_step_ms": mean_step_ms,
                    "device": str(dev),
                    "profile": profile,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                }
            )
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": asdict(mcfg),
                "profile": profile,
                "genetic_memory": None if genetic_memory is None else genetic_memory.cpu(),
                "run_flags": run_flags,
            },
            save_path,
        )
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps({"flags": run_flags, "records": metrics_records}, indent=2), encoding="utf-8")
        print({"saved": save_path})
        return

    pop = [ModelCore(**asdict(mcfg)).to(dev) for _ in range(population_size)]
    opts = [torch.optim.AdamW(model.parameters(), lr=tcfg.lr) for model in pop]
    elite_count = max(1, int(population_size * elite_fraction))
    for generation in range(tcfg.epochs):
        if enable_curriculum:
            remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, generation)
            env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, generation)
        else:
            remap_probability = tcfg.remap_probability
            env_volatility = 0.0
        mean_step_acc = 0.0
        for idx, model in enumerate(pop):
            model.train()
            warmup_loss, _, step_ms = run_gradient_epoch(
                model,
                world,
                mcfg,
                tcfg,
                opts[idx],
                wcfg,
                dev,
                remap_probability=remap_probability,
                env_volatility=env_volatility,
            )
            mean_step_acc += step_ms
            print({"generation": generation + 1, "agent": idx, "warmup_loss": warmup_loss, "remap_probability": remap_probability, "env_volatility": env_volatility})

        scores = [
            evaluate_fitness(
                model,
                world,
                mcfg,
                wcfg,
                dev,
                steps=max(4, tcfg.unroll_steps // 2),
                batch=max(2, tcfg.batch_size // 4),
                env_volatility=env_volatility,
            )
            for model in pop
        ]
        rank = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        elites = rank[:elite_count]
        best_idx = elites[0]
        print({"generation": generation + 1, "best_agent": best_idx, "best_fitness": scores[best_idx], "mean_fitness": sum(scores) / len(scores)})
        metrics_records.append(
            {
                "generation": generation + 1,
                "best_agent": best_idx,
                "best_fitness": scores[best_idx],
                "mean_fitness": sum(scores) / len(scores),
                "mean_step_ms": mean_step_acc / max(1, len(pop)),
                "remap_probability": remap_probability,
                "env_volatility": env_volatility,
            }
        )

        next_pop = [copy.deepcopy(pop[i]).to(dev) for i in elites]
        while len(next_pop) < population_size:
            parent = pop[random.choice(elites)]
            child = mutate_from_parent(parent, mutation_std).to(dev)
            next_pop.append(child)
        pop = next_pop
        opts = [torch.optim.AdamW(model.parameters(), lr=tcfg.lr) for model in pop]

    final_scores = [
        evaluate_fitness(
            model,
            world,
            mcfg,
            wcfg,
            dev,
            steps=6,
            batch=max(2, tcfg.batch_size // 4),
            env_volatility=env_volatility_end,
        )
        for model in pop
    ]
    best_idx = max(range(len(pop)), key=lambda i: final_scores[i])
    best_model = pop[best_idx]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_model.state_dict(),
            "model_config": asdict(mcfg),
            "profile": profile,
            "coevolution": True,
            "population_size": population_size,
            "fitness": final_scores[best_idx],
            "run_flags": run_flags,
        },
        save_path,
    )
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps({"flags": run_flags, "records": metrics_records}, indent=2), encoding="utf-8")
    print({"saved": save_path, "best_agent": best_idx, "best_fitness": final_scores[best_idx]})


if __name__ == "__main__":
    app()

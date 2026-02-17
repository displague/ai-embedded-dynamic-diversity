from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import typer
from rich import print
from torch import nn

from ai_embedded_dynamic_diversity.config import TrainConfig, WorldConfig, model_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld
from ai_embedded_dynamic_diversity.train.losses import loss_fn

app = typer.Typer(add_completion=False)


def choose_device(preferred: str, strict: bool = True) -> torch.device:
    normalized = preferred.strip().lower()
    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if strict:
            raise typer.BadParameter(
                "CUDA was requested but is unavailable in this environment. "
                "Check that the active Python environment has a CUDA-enabled PyTorch build."
            )
        return torch.device("cpu")
    if normalized == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if strict:
            raise typer.BadParameter("MPS was requested but is unavailable in this environment.")
        return torch.device("cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if strict:
        raise typer.BadParameter(f"Unknown device '{preferred}'. Expected one of: cpu, cuda, mps.")
    return torch.device("cpu")


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


@dataclass
class TransferState:
    name: str
    projection: torch.Tensor
    mapping: torch.Tensor
    mapping_seed: int


def _resolve_embodiments(embodiments_csv: str) -> list[str]:
    names = []
    for token in embodiments_csv.replace(";", ",").split(","):
        name = token.strip().lower()
        if not name:
            continue
        get_embodiment(name)
        if name not in names:
            names.append(name)
    return names


def _build_transfer_states(
    embodiment_names: list[str],
    mcfg,
    dev: torch.device,
    seed: int,
) -> dict[str, TransferState]:
    states: dict[str, TransferState] = {}
    for idx, name in enumerate(embodiment_names):
        emb = get_embodiment(name)
        control_dim = len(emb.controls)
        gen = torch.Generator()
        gen.manual_seed(seed + idx * 7919)
        projection = (torch.randn(mcfg.signal_dim, control_dim, generator=gen) * 0.3).to(dev)
        mapping_seed = seed + idx * 409 + 17
        mapping = device_map_for_embodiment(mcfg.io_channels, emb, device=dev, permutation_seed=mapping_seed)
        states[name] = TransferState(
            name=name,
            projection=projection,
            mapping=mapping,
            mapping_seed=mapping_seed,
        )
    return states


def _transfer_mismatch_loss(
    out_io: torch.Tensor,
    obs: torch.Tensor,
    transfer_states: dict[str, TransferState] | None,
    mcfg,
    dev: torch.device,
    sample_count: int,
    remap_probability: float,
) -> tuple[torch.Tensor, float, int]:
    zero = out_io.new_zeros(())
    if not transfer_states:
        return zero, 0.0, 0

    names = list(transfer_states.keys())
    if sample_count <= 0:
        chosen = names
    else:
        chosen = random.sample(names, k=min(sample_count, len(names)))

    losses: list[torch.Tensor] = []
    remap_events = 0
    for name in chosen:
        state = transfer_states[name]
        if random.random() < remap_probability:
            state.mapping_seed += 97
            state.mapping = device_map_for_embodiment(
                mcfg.io_channels,
                get_embodiment(name),
                device=dev,
                permutation_seed=state.mapping_seed,
            )
            remap_events += 1
        desired = torch.tanh(obs @ state.projection)
        applied = out_io @ state.mapping
        losses.append(nn.functional.mse_loss(applied, desired))

    if not losses:
        return zero, 0.0, remap_events
    mismatch = torch.stack(losses).mean()
    return mismatch, float(mismatch.detach().item()), remap_events


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
    transfer_states: dict[str, TransferState] | None = None,
    transfer_loss_weight: float = 0.0,
    transfer_samples_per_step: int = 0,
    use_amp: bool = False,
    scaler=None,
) -> tuple[float, torch.Tensor, float, float]:
    state = world.init(tcfg.batch_size)
    if genetic_memory is None:
        memory = model.init_memory(tcfg.batch_size, mcfg.memory_slots, mcfg.memory_dim, dev)
    else:
        memory = genetic_memory.repeat(tcfg.batch_size, 1, 1).detach()
    epoch_loss = 0.0
    step_times = []
    transfer_mismatch_total = 0.0
    amp_enabled = use_amp and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for step_index in range(tcfg.unroll_steps):
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
            obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)

            remap_code = torch.zeros(tcfg.batch_size, mcfg.max_remap_groups, device=dev)
            if random.random() < remap_probability:
                group = random.randrange(mcfg.max_remap_groups)
                remap_code[:, group] = 1.0

            out = model(obs, memory, remap_code)
            memory = out["memory"].detach()

            target = torch.cat([obs[:, : mcfg.io_channels // 2], out["readiness"][:, : mcfg.io_channels - (mcfg.io_channels // 2)]], dim=1)
            base_loss, _ = loss_fn(
                out,
                target,
                entropy_weight=tcfg.entropy_weight,
                energy_weight=tcfg.energy_weight,
                memory_consistency_weight=tcfg.memory_consistency_weight,
            )
            transfer_mismatch = 0.0
            transfer_loss_tensor = out["io"].new_zeros(())
            if transfer_states and transfer_loss_weight > 0.0:
                transfer_loss_tensor, transfer_mismatch, _ = _transfer_mismatch_loss(
                    out_io=out["io"],
                    obs=obs,
                    transfer_states=transfer_states,
                    mcfg=mcfg,
                    dev=dev,
                    sample_count=transfer_samples_per_step,
                    remap_probability=remap_probability,
                )
            total_loss = base_loss + transfer_loss_weight * transfer_loss_tensor
            transfer_mismatch_total += transfer_mismatch

        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        epoch_loss += float(total_loss.detach().item())

        action_field = out["io"].detach().float().mean(dim=1, keepdim=True).repeat(1, wcfg.x * wcfg.y * wcfg.z)
        controls = world.random_controls(tcfg.batch_size, env_volatility, step_index=step_index)
        state = world.step(state, action_field, controls=controls)
        step_times.append((time.perf_counter() - t0) * 1000.0)
    final_memory = memory.mean(dim=0, keepdim=True).detach()
    mean_step_ms = sum(step_times) / max(1, len(step_times))
    mean_transfer_mismatch = transfer_mismatch_total / max(1, tcfg.unroll_steps)
    return epoch_loss / max(1, tcfg.unroll_steps), final_memory, mean_step_ms, mean_transfer_mismatch


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
    transfer_states: dict[str, TransferState] | None = None,
    transfer_fitness_weight: float = 0.0,
    transfer_samples_per_step: int = 0,
) -> float:
    model.eval()
    state = world.init(batch)
    memory = model.init_memory(batch, mcfg.memory_slots, mcfg.memory_dim, dev)
    energy_total = 0.0
    remap_events = 0.0
    transfer_mismatch_total = 0.0
    for step_index in range(steps):
        obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)
        remap = torch.zeros(batch, mcfg.max_remap_groups, device=dev)
        if random.random() < 0.3:
            remap[:, random.randrange(mcfg.max_remap_groups)] = 1.0
            remap_events += 1.0
        out = model(obs, memory, remap)
        memory = out["memory"]
        if transfer_states and transfer_fitness_weight > 0.0:
            _, transfer_mismatch, _ = _transfer_mismatch_loss(
                out_io=out["io"],
                obs=obs,
                transfer_states=transfer_states,
                mcfg=mcfg,
                dev=dev,
                sample_count=transfer_samples_per_step,
                remap_probability=0.3,
            )
            transfer_mismatch_total += transfer_mismatch
        action_field = out["io"].float().mean(dim=1, keepdim=True).repeat(1, wcfg.x * wcfg.y * wcfg.z)
        controls = world.random_controls(batch, env_volatility, step_index=step_index)
        state = world.step(state, action_field, controls=controls)
        energy_total += float(out["energy"].mean().item())
    vitality = float(state.life.mean().item())
    stress = float(state.stress.mean().item())
    energy = energy_total / max(1, steps)
    remap_bonus = remap_events / max(1.0, steps)
    transfer_mismatch = transfer_mismatch_total / max(1.0, steps)
    # Higher is better: survive under stress using less energy.
    return vitality - 0.45 * stress - 0.1 * energy + 0.05 * remap_bonus - transfer_fitness_weight * transfer_mismatch


def mutate_from_parent(parent: ModelCore, mutation_std: float) -> ModelCore:
    child = copy.deepcopy(parent)
    if mutation_std > 0.0:
        with torch.no_grad():
            for param in child.parameters():
                noise = torch.randn_like(param) * mutation_std
                param.add_(noise)
    return child


def _load_checkpoint_weights(model: ModelCore, init_weights: str, dev: torch.device) -> dict:
    ckpt = torch.load(init_weights, map_location=dev)
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint does not contain a 'model' state dict: {init_weights}")
    model.load_state_dict(ckpt["model"])
    return ckpt


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
    embodiments: str = "hexapod,car,drone",
    enable_embodiment_transfer_loss: bool = False,
    transfer_loss_weight: float = 0.35,
    transfer_samples_per_step: int = 2,
    transfer_fitness_weight: float = 0.08,
    use_amp: bool = True,
    allow_tf32: bool = True,
    compile_model: bool = False,
    strict_device: bool = True,
    init_weights: str = "",
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

    dev = choose_device(tcfg.device, strict=strict_device)
    if dev.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    embodiment_names = _resolve_embodiments(embodiments)
    if enable_embodiment_transfer_loss and not embodiment_names:
        raise typer.BadParameter("Embodiment transfer loss was enabled, but no embodiments were resolved.")
    if not enable_embodiment_transfer_loss:
        embodiment_names = []

    if compile_model and coevolution:
        print({"warning": "compile_model disabled in coevolution mode to avoid per-agent graph overhead"})
        compile_model = False
    if compile_model and not hasattr(torch, "compile"):
        print({"warning": "torch.compile is unavailable in this torch build; compile_model disabled"})
        compile_model = False

    world = DynamicDiversityWorld(wcfg.x, wcfg.y, wcfg.z, wcfg.resource_channels, wcfg.decay, device=str(dev))
    genetic_memory = torch.zeros(1, mcfg.memory_slots, mcfg.memory_dim, device=dev) if enable_genetic_memory else None
    amp_enabled = use_amp and dev.type == "cuda"

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
        "embodiments": embodiment_names,
        "enable_embodiment_transfer_loss": enable_embodiment_transfer_loss,
        "transfer_loss_weight": transfer_loss_weight,
        "transfer_samples_per_step": transfer_samples_per_step,
        "transfer_fitness_weight": transfer_fitness_weight,
        "use_amp": amp_enabled,
        "allow_tf32": allow_tf32 and dev.type == "cuda",
        "compile_model": compile_model,
        "strict_device": strict_device,
        "init_weights": init_weights,
    }

    if not coevolution:
        model = ModelCore(**asdict(mcfg)).to(dev)
        if init_weights:
            try:
                _load_checkpoint_weights(model, init_weights, dev)
            except (ValueError, RuntimeError) as exc:
                raise typer.BadParameter(str(exc)) from exc
        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        scaler = _make_grad_scaler(amp_enabled)
        transfer_states = (
            _build_transfer_states(embodiment_names, mcfg, dev, seed=seed + 1234)
            if embodiment_names
            else None
        )
        for epoch in range(tcfg.epochs):
            if enable_curriculum:
                remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, epoch)
                env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, epoch)
            else:
                remap_probability = tcfg.remap_probability
                env_volatility = 0.0
            mean_loss, memory_snapshot, mean_step_ms, mean_transfer_mismatch = run_gradient_epoch(
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
                transfer_states=transfer_states,
                transfer_loss_weight=transfer_loss_weight,
                transfer_samples_per_step=transfer_samples_per_step,
                use_amp=amp_enabled,
                scaler=scaler if amp_enabled else None,
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
                transfer_states=transfer_states,
                transfer_fitness_weight=transfer_fitness_weight,
                transfer_samples_per_step=transfer_samples_per_step,
            )
            metrics_records.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "fitness": fitness,
                    "mean_step_ms": mean_step_ms,
                    "mean_transfer_mismatch": mean_transfer_mismatch,
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
                    "mean_transfer_mismatch": mean_transfer_mismatch,
                    "device": str(dev),
                    "profile": profile,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                }
            )
        model_for_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_for_save.state_dict(),
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

    if init_weights:
        seed_model = ModelCore(**asdict(mcfg)).to(dev)
        try:
            _load_checkpoint_weights(seed_model, init_weights, dev)
        except (ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from exc
        pop = [copy.deepcopy(seed_model).to(dev) for _ in range(population_size)]
        for idx in range(1, population_size):
            pop[idx] = mutate_from_parent(pop[idx], mutation_std).to(dev)
    else:
        pop = [ModelCore(**asdict(mcfg)).to(dev) for _ in range(population_size)]
    opts = [torch.optim.AdamW(model.parameters(), lr=tcfg.lr) for model in pop]
    scalers = [_make_grad_scaler(amp_enabled) for _ in pop]
    transfer_states_by_agent = (
        [_build_transfer_states(embodiment_names, mcfg, dev, seed=seed + 5000 + idx * 97) for idx in range(population_size)]
        if embodiment_names
        else [None for _ in pop]
    )
    elite_count = max(1, int(population_size * elite_fraction))
    for generation in range(tcfg.epochs):
        if enable_curriculum:
            remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, generation)
            env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, generation)
        else:
            remap_probability = tcfg.remap_probability
            env_volatility = 0.0
        mean_step_acc = 0.0
        mean_transfer_mismatch_acc = 0.0
        for idx, model in enumerate(pop):
            model.train()
            warmup_loss, _, step_ms, mean_transfer_mismatch = run_gradient_epoch(
                model,
                world,
                mcfg,
                tcfg,
                opts[idx],
                wcfg,
                dev,
                remap_probability=remap_probability,
                env_volatility=env_volatility,
                transfer_states=transfer_states_by_agent[idx],
                transfer_loss_weight=transfer_loss_weight,
                transfer_samples_per_step=transfer_samples_per_step,
                use_amp=amp_enabled,
                scaler=scalers[idx],
            )
            mean_step_acc += step_ms
            mean_transfer_mismatch_acc += mean_transfer_mismatch
            print(
                {
                    "generation": generation + 1,
                    "agent": idx,
                    "warmup_loss": warmup_loss,
                    "mean_transfer_mismatch": mean_transfer_mismatch,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                }
            )

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
                transfer_states=transfer_states_by_agent[idx],
                transfer_fitness_weight=transfer_fitness_weight,
                transfer_samples_per_step=transfer_samples_per_step,
            )
            for idx, model in enumerate(pop)
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
                "mean_transfer_mismatch": mean_transfer_mismatch_acc / max(1, len(pop)),
                "remap_probability": remap_probability,
                "env_volatility": env_volatility,
            }
        )

        next_pop = [copy.deepcopy(pop[i]).to(dev) for i in elites]
        next_transfer_states = [copy.deepcopy(transfer_states_by_agent[i]) for i in elites]
        while len(next_pop) < population_size:
            parent_idx = random.choice(elites)
            parent = pop[parent_idx]
            child = mutate_from_parent(parent, mutation_std).to(dev)
            next_pop.append(child)
            next_transfer_states.append(copy.deepcopy(transfer_states_by_agent[parent_idx]))
        pop = next_pop
        transfer_states_by_agent = next_transfer_states
        opts = [torch.optim.AdamW(model.parameters(), lr=tcfg.lr) for model in pop]
        scalers = [_make_grad_scaler(amp_enabled) for _ in pop]

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
            transfer_states=transfer_states_by_agent[idx],
            transfer_fitness_weight=transfer_fitness_weight,
            transfer_samples_per_step=transfer_samples_per_step,
        )
        for idx, model in enumerate(pop)
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

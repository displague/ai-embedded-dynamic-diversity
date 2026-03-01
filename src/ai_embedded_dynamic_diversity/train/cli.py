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

from ai_embedded_dynamic_diversity.config import TrainConfig, WorldConfig, model_config_for_profile, world_config_for_profile
from ai_embedded_dynamic_diversity.models import ModelCore, UniversalConstructor, load_constructor_tape
from ai_embedded_dynamic_diversity.sim.embodiments import device_map_for_embodiment, get_embodiment
from ai_embedded_dynamic_diversity.sim.autopoiesis import autopoietic_metrics
from ai_embedded_dynamic_diversity.sim.world import DynamicDiversityWorld
from ai_embedded_dynamic_diversity.sim.signaling import SignalingWorld
from ai_embedded_dynamic_diversity.train.losses import loss_fn
from ai_embedded_dynamic_diversity.train.quantization import prepare_qat_model
from ai_embedded_dynamic_diversity.models.memory_bank import GeneticMemoryBank
from ai_embedded_dynamic_diversity.train.curriculum import AdaptiveLossController
from ai_embedded_dynamic_diversity.train.device import choose_device

app = typer.Typer(add_completion=False)


def _make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def _resolve_noise_profile(profile: str) -> str:
    normalized = profile.strip().lower()
    if not normalized:
        return "none"
    allowed = {"none", "dropout-quant-v1", "dropout-quant-v2"}
    if normalized not in allowed:
        raise ValueError(f"Unknown noise profile '{profile}'. Allowed: {', '.join(sorted(allowed))}")
    return normalized


def _resolve_force_curriculum_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    allowed = {"none", "continuous-blow"}
    if normalized not in allowed:
        raise ValueError(f"Unknown force curriculum mode '{mode}'. Allowed: {', '.join(sorted(allowed))}")
    return normalized


def _inject_force_curriculum_controls(
    controls,
    mode: str,
    env_volatility: float,
    strength: float,
    step_index: int,
) -> None:
    if mode == "none":
        return
    s = max(0.0, float(strength))
    if s <= 1e-8:
        return
    vol = max(0.0, float(env_volatility))
    if mode == "continuous-blow":
        phase = step_index / 6.0
        amp = min(1.0, max(0.0, (0.45 + 0.55 * vol) * s))
        w = 0.65 + 0.35 * float(torch.sin(torch.tensor(phase)).item())
        controls.force_active.fill_(1.0)
        controls.force_strength = torch.maximum(controls.force_strength, torch.full_like(controls.force_strength, amp * w))
        fx = 0.9 + 0.25 * float(torch.cos(torch.tensor(phase * 0.5)).item())
        fy = 0.15 * float(torch.sin(torch.tensor(phase * 0.8)).item())
        fz = 0.05 * float(torch.sin(torch.tensor(phase * 0.3)).item())
        vec = controls.force_vector.new_tensor([fx, fy, fz]).view(1, 3).repeat(controls.force_vector.size(0), 1)
        controls.force_vector = vec * (0.7 + 0.6 * vol)
        return
    raise ValueError(f"Unsupported force curriculum mode: {mode}")


def _apply_observation_noise(
    obs: torch.Tensor,
    profile: str,
    seed: int,
    step: int,
    strength: float = 1.0,
) -> torch.Tensor:
    s = max(0.0, float(strength))
    if profile == "none" or s <= 1e-8:
        return obs
    if profile == "dropout-quant-v1":
        gen = torch.Generator(device=obs.device)
        gen.manual_seed(int(seed * 10007 + step * 131))

        noisy = obs
        keep = (torch.rand(obs.shape, generator=gen, device=obs.device) > min(0.95, 0.12 * s)).float()
        noisy = noisy * keep
        noisy = noisy + (0.045 * s) * torch.randn(obs.shape, generator=gen, device=obs.device)

        min_v = torch.min(noisy, dim=1, keepdim=True).values
        max_v = torch.max(noisy, dim=1, keepdim=True).values
        span = (max_v - min_v).clamp_min(1e-6)
        normalized = (noisy - min_v) / span
        quantized = torch.round(normalized * 255.0) / 255.0
        return quantized * span + min_v
    if profile == "dropout-quant-v2":
        gen = torch.Generator(device=obs.device)
        gen.manual_seed(int(seed * 20011 + step * 313))

        noisy = obs
        keep = (torch.rand(obs.shape, generator=gen, device=obs.device) > min(0.98, 0.28 * s)).float()
        noisy = noisy * keep
        if step % 7 == 0:
            brownout = (torch.rand(obs.shape[0], obs.shape[1], generator=gen, device=obs.device) > min(0.95, 0.35 * s)).float()
            noisy = noisy * brownout
        noisy = noisy * (1.0 + (0.08 * s) * torch.randn(obs.shape, generator=gen, device=obs.device))
        noisy = noisy + (0.09 * s) * torch.randn(obs.shape, generator=gen, device=obs.device)

        min_v = torch.min(noisy, dim=1, keepdim=True).values
        max_v = torch.max(noisy, dim=1, keepdim=True).values
        span = (max_v - min_v).clamp_min(1e-6)
        normalized = (noisy - min_v) / span
        quantized = torch.round(normalized * 31.0) / 31.0
        return quantized * span + min_v
    return obs


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


def _autopoietic_step_score(
    out: dict[str, torch.Tensor],
    state,
    self_repair_weight: float,
    closure_weight: float,
    resource_cycle_weight: float,
) -> torch.Tensor:
    readiness_mean = torch.mean(out["readiness"])
    energy_mean = torch.mean(out["energy"])
    stress_mean = torch.mean(state.stress)
    resource_mean = torch.mean(state.resources[:, :1])
    vitality_mean = torch.mean(state.life)

    closure_signal = torch.sigmoid(readiness_mean - 0.6 * energy_mean)
    self_repair_signal = torch.sigmoid(0.5 - stress_mean)
    resource_cycle_signal = torch.sigmoid(0.8 * resource_mean + 0.4 * vitality_mean - 0.7 * energy_mean)
    denom = max(1e-8, self_repair_weight + closure_weight + resource_cycle_weight)
    return (
        closure_weight * closure_signal
        + self_repair_weight * self_repair_signal
        + resource_cycle_weight * resource_cycle_signal
    ) / denom


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
    noise_profile: str = "none",
    noise_strength: float = 0.0,
    noise_seed: int = 0,
    use_amp: bool = False,
    scaler=None,
    remap_loss_weight: float = 0.1,
    enable_autopoietic_objective: bool = False,
    autopoietic_loss_weight: float = 0.10,
    autopoietic_fitness_gain: float = 0.15,
    autopoietic_self_repair_weight: float = 0.35,
    autopoietic_closure_weight: float = 0.45,
    autopoietic_resource_cycle_weight: float = 0.20,
    detection_loss_weight: float = 0.1,
    emergent_signal_loss_weight: float = 0.05,
    memory_persistence_loss_weight: float = 0.05,
    paging_loss_weight: float = 0.01,
    force_curriculum_mode: str = "none",
    force_curriculum_strength: float = 0.0,
) -> tuple[float, torch.Tensor, float, float, float, float, float, float, float, float, float]:
    state = world.init(tcfg.batch_size)
    if genetic_memory is None:
        memory = model.init_memory(tcfg.batch_size, mcfg.memory_slots, mcfg.memory_dim, dev)
    else:
        memory = genetic_memory.repeat(tcfg.batch_size, 1, 1).detach()
    initial_memory_prior = memory.detach().clone()
    epoch_loss = 0.0
    step_times = []
    transfer_mismatch_total = 0.0
    remap_loss_total = 0.0
    detection_loss_total = 0.0
    emergent_signal_loss_total = 0.0
    memory_persistence_loss_total = 0.0
    paging_loss_total = 0.0
    autopoietic_score_total = 0.0
    autopoietic_loss_total = 0.0
    amp_enabled = use_amp and dev.type == "cuda"
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    
    for step_index in range(tcfg.unroll_steps):
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=amp_enabled):
            # Inject signals
            target_signal_type = torch.zeros(tcfg.batch_size, dtype=torch.long, device=dev)
            if isinstance(world, SignalingWorld):
                target_signal_type = world.inject_signals(tcfg.batch_size)
                obs = world.encode_observation_with_signals(state, signal_dim=mcfg.signal_dim, labels=target_signal_type)
            else:
                obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)
            
            clean_obs = obs 
            
            obs = _apply_observation_noise(
                obs,
                profile=noise_profile,
                seed=noise_seed,
                step=step_index,
                strength=noise_strength,
            )

            remap_code = torch.zeros(tcfg.batch_size, mcfg.max_remap_groups, device=dev)
            if random.random() < remap_probability:
                group = random.randrange(mcfg.max_remap_groups)
                remap_code[:, group] = 1.0

            out = model(obs, memory, remap_code)

            target = torch.cat(
                [
                    clean_obs[:, : mcfg.io_channels // 2],
                    out["readiness"][:, : mcfg.io_channels - (mcfg.io_channels // 2)],
                ],
                dim=1,
            )
            base_loss, logs = loss_fn(
                out,
                target,
                entropy_weight=tcfg.entropy_weight,
                energy_weight=tcfg.energy_weight,
                memory_consistency_weight=tcfg.memory_consistency_weight,
                remap_loss_weight=remap_loss_weight,
                target_remap_code=remap_code,
                detection_loss_weight=detection_loss_weight,
                target_signal_type=target_signal_type,
                emergent_signal_loss_weight=emergent_signal_loss_weight,
                memory_persistence_loss_weight=memory_persistence_loss_weight,
                initial_memory=initial_memory_prior,
                paging_loss_weight=paging_loss_weight,
            )
            remap_loss_total += logs["remap_loss"]
            detection_loss_total += logs["detection_loss"]
            emergent_signal_loss_total += logs["emergent_signal_loss"]
            memory_persistence_loss_total += logs["memory_persistence_loss"]
            paging_loss_total += logs["paging_loss"]
            
            transfer_mismatch = 0.0
            transfer_loss_tensor = out["io"].new_zeros(())
            if transfer_states and transfer_loss_weight > 0.0:
                transfer_loss_tensor, transfer_mismatch, _ = _transfer_mismatch_loss(
                    out_io=out["io"],
                    obs=clean_obs,
                    transfer_states=transfer_states,
                    mcfg=mcfg,
                    dev=dev,
                    sample_count=transfer_samples_per_step,
                    remap_probability=remap_probability,
                )
            
            total_loss = base_loss + transfer_loss_weight * transfer_loss_tensor
            if enable_autopoietic_objective and autopoietic_loss_weight > 0.0:
                autopoietic_score_t = _autopoietic_step_score(
                    out=out,
                    state=state,
                    self_repair_weight=autopoietic_self_repair_weight,
                    closure_weight=autopoietic_closure_weight,
                    resource_cycle_weight=autopoietic_resource_cycle_weight,
                )
                autopoietic_loss_t = 1.0 - autopoietic_score_t
                total_loss = total_loss + autopoietic_loss_weight * autopoietic_loss_t
                autopoietic_score_total += float(autopoietic_score_t.detach().item())
                autopoietic_loss_total += float(autopoietic_loss_t.detach().item())
            
            transfer_mismatch_total += transfer_mismatch

            # Apply actions to world
            action_field = out["io"].detach().float().mean(dim=1, keepdim=True).repeat(1, world.x * world.y * world.z)
            controls = world.random_controls(tcfg.batch_size, volatility=env_volatility, step_index=step_index)
            _inject_force_curriculum_controls(
                controls=controls,
                mode=force_curriculum_mode,
                env_volatility=env_volatility,
                strength=force_curriculum_strength,
                step_index=step_index,
            )
            state = world.step(state, action_field, controls)

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
        memory = out["memory"].detach()
        step_times.append((time.perf_counter() - t0) * 1000.0)


    mean_step_ms = sum(step_times) / max(1, len(step_times))
    mean_transfer_mismatch = transfer_mismatch_total / max(1, tcfg.unroll_steps)
    mean_remap_loss = remap_loss_total / max(1, tcfg.unroll_steps)
    mean_detection_loss = detection_loss_total / max(1, tcfg.unroll_steps)
    mean_emergent_signal_loss = emergent_signal_loss_total / max(1, tcfg.unroll_steps)
    mean_memory_persistence_loss = memory_persistence_loss_total / max(1, tcfg.unroll_steps)
    mean_paging_loss = paging_loss_total / max(1, tcfg.unroll_steps)
    mean_autopoietic_score = autopoietic_score_total / max(1, tcfg.unroll_steps)
    mean_autopoietic_loss = autopoietic_loss_total / max(1, tcfg.unroll_steps)
    final_memory = memory.mean(dim=0, keepdim=True).detach()
    
    return (
        epoch_loss / max(1, tcfg.unroll_steps),
        final_memory,
        mean_step_ms,
        mean_transfer_mismatch,
        mean_remap_loss,
        mean_detection_loss,
        mean_emergent_signal_loss,
        mean_memory_persistence_loss,
        mean_paging_loss,
        mean_autopoietic_score,
        mean_autopoietic_loss,
    )


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
    noise_profile: str = "none",
    noise_strength: float = 0.0,
    noise_seed: int = 0,
    autopoietic_fitness_weight: float = 0.0,
    genetic_memory_persistence_weight: float = 0.0,
    force_curriculum_mode: str = "none",
    force_curriculum_strength: float = 0.0,
) -> float:
    model.eval()
    state = world.init(batch)
    memory = model.init_memory(batch, mcfg.memory_slots, mcfg.memory_dim, dev)
    initial_memory = memory.clone()
    energy_total = 0.0
    remap_events = 0.0
    transfer_mismatch_total = 0.0
    mismatch_values: list[float] = []
    vitality_values: list[float] = []
    stress_values: list[float] = []
    energy_values: list[float] = []
    remap_steps: list[int] = []
    resource_values: list[float] = []
    for step_index in range(steps):
        clean_obs = world.encode_observation(state, signal_dim=mcfg.signal_dim)
        obs = _apply_observation_noise(
            clean_obs,
            profile=noise_profile,
            seed=noise_seed,
            step=step_index,
            strength=noise_strength,
        )
        remap = torch.zeros(batch, mcfg.max_remap_groups, device=dev)
        if random.random() < 0.3:
            remap[:, random.randrange(mcfg.max_remap_groups)] = 1.0
            remap_events += 1.0
            remap_steps.append(step_index)
        out = model(obs, memory, remap)
        memory = out["memory"]
        if transfer_states and transfer_fitness_weight > 0.0:
            _, transfer_mismatch, _ = _transfer_mismatch_loss(
                out_io=out["io"],
                obs=clean_obs,
                transfer_states=transfer_states,
                mcfg=mcfg,
                dev=dev,
                sample_count=transfer_samples_per_step,
                remap_probability=0.3,
            )
            transfer_mismatch_total += transfer_mismatch
        action_field = out["io"].float().mean(dim=1, keepdim=True).repeat(1, wcfg.x * wcfg.y * wcfg.z)
        controls = world.random_controls(batch, env_volatility, step_index=step_index)
        _inject_force_curriculum_controls(
            controls=controls,
            mode=force_curriculum_mode,
            env_volatility=env_volatility,
            strength=force_curriculum_strength,
            step_index=step_index,
        )
        state = world.step(state, action_field, controls=controls)
        mismatch_values.append(float(torch.mean((out["io"] @ out["io"].new_ones(mcfg.io_channels, 1)) ** 2).item()))
        vitality_values.append(float(state.life.mean().item()))
        stress_values.append(float(state.stress.mean().item()))
        energy_values.append(float(out["energy"].mean().item()))
        resource_values.append(float(state.resources[:, :1].mean().item()))
        energy_total += float(out["energy"].mean().item())
    
    # Memory persistence: how well did the model preserve its state vs its initial genetic prior
    # Higher similarity (lower distance) is better if weight is positive.
    memory_persistence = 1.0 / (1.0 + torch.norm(memory - initial_memory))
    
    vitality = float(state.life.mean().item())
    stress = float(state.stress.mean().item())
    energy = energy_total / max(1, steps)
    remap_bonus = remap_events / max(1.0, steps)
    transfer_mismatch = transfer_mismatch_total / max(1.0, steps)
    autopoietic_score = autopoietic_metrics(
        mismatch_values=mismatch_values,
        vitality_values=vitality_values,
        stress_values=stress_values,
        energy_values=energy_values,
        remap_steps=remap_steps,
        resource_values=resource_values,
    )["autopoietic_score"]
    # Higher is better: survive under stress using less energy.
    return (
        vitality
        - 0.45 * stress
        - 0.1 * energy
        + 0.05 * remap_bonus
        - transfer_fitness_weight * transfer_mismatch
        + autopoietic_fitness_weight * autopoietic_score
        + genetic_memory_persistence_weight * float(memory_persistence.item())
    )


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


def _resolve_model_config(
    profile: str,
    constructor_tape_path: str,
) -> tuple:
    if constructor_tape_path:
        tape = load_constructor_tape(constructor_tape_path)
        constructed = UniversalConstructor(base_profile=profile).build(tape, seed_state=0)
        return constructed.config, tape
    return model_config_for_profile(profile), None


@app.command()
def run(
    epochs: int = 20,
    batch_size: int = 16,
    unroll_steps: int = 12,
    lr: float = 2e-4,
    device: str = "cuda",
    profile: str = "base",
    world_profile: str = "",
    gating_mode: str = "sigmoid",
    topk_gating: int = 0,
    enable_dmd_gating: bool = False,
    enable_phase_gating: bool = False,
    enable_multi_scale_gating: bool = False,
    enable_qat: bool = False,
    coevolution: bool = False,
    population_size: int = 4,
    elite_fraction: float = 0.5,
    mutation_std: float = 0.01,
    enable_adaptive_loss: bool = False,
    adaptive_loss_alpha: float = 0.1,
    enable_curriculum: bool = False,
    curriculum_power: float = 1.0,
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
    enable_autopoietic_objective: bool = False,
    autopoietic_loss_weight: float = 0.10,
    autopoietic_fitness_gain: float = 0.15,
    autopoietic_self_repair_weight: float = 0.35,
    autopoietic_closure_weight: float = 0.45,
    autopoietic_resource_cycle_weight: float = 0.20,
    remap_loss_weight: float = 0.1,
    detection_loss_weight: float = 0.1,
    emergent_signal_loss_weight: float = 0.05,
    memory_persistence_loss_weight: float = 0.05,
    genetic_memory_persistence_weight: float = 0.05,
    paging_loss_weight: float = 0.01,
    noise_profile: str = "none",
    enable_noise_curriculum: bool = False,
    noise_strength_start: float = 0.2,
    noise_strength_end: float = 1.0,
    force_curriculum_mode: str = "none",
    force_curriculum_strength_start: float = 0.0,
    force_curriculum_strength_end: float = 1.0,
    use_amp: bool = True,
    allow_tf32: bool = True,
    compile_model: bool = False,
    strict_device: bool = True,
    constructor_tape_path: str = "",
    init_weights: str = "",
    memory_bank_path: str = "",
    seed: int = 7,
    metrics_path: str = "",
    save_path: str = "artifacts/model-core.pt",
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        mcfg, constructor_tape = _resolve_model_config(profile, constructor_tape_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not constructor_tape_path:
        mcfg.gating_mode = gating_mode
        mcfg.topk_gating = topk_gating
        mcfg.enable_dmd_gating = enable_dmd_gating
        mcfg.enable_phase_gating = enable_phase_gating
        mcfg.enable_multi_scale_gating = enable_multi_scale_gating
    try:
        wcfg = world_config_for_profile(world_profile) if world_profile.strip() else WorldConfig()
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    tcfg = TrainConfig(epochs=epochs, batch_size=batch_size, unroll_steps=unroll_steps, lr=lr, device=device)
    try:
        noise_profile_resolved = _resolve_noise_profile(noise_profile)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        force_curriculum_mode_resolved = _resolve_force_curriculum_mode(force_curriculum_mode)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if noise_strength_start < 0.0 or noise_strength_end < 0.0:
        raise typer.BadParameter("noise_strength_start and noise_strength_end must be >= 0.0")
    if force_curriculum_strength_start < 0.0 or force_curriculum_strength_end < 0.0:
        raise typer.BadParameter("force_curriculum_strength_start and force_curriculum_strength_end must be >= 0.0")
    if autopoietic_loss_weight < 0.0:
        raise typer.BadParameter("autopoietic_loss_weight must be >= 0.0")
    if autopoietic_fitness_gain < 0.0:
        raise typer.BadParameter("autopoietic_fitness_gain must be >= 0.0")
    if remap_loss_weight < 0.0:
        raise typer.BadParameter("remap_loss_weight must be >= 0.0")

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

    world = SignalingWorld(
        wcfg.x,
        wcfg.y,
        wcfg.z,
        wcfg.resource_channels,
        wcfg.decay,
        device=str(dev),
        actuation_delay_steps=wcfg.actuation_delay_steps,
        actuation_noise_std=wcfg.actuation_noise_std,
        sensor_latency_steps=wcfg.sensor_latency_steps,
        sensor_dropout_burst_prob=wcfg.sensor_dropout_burst_prob,
        surface_friction_scale=wcfg.surface_friction_scale,
        disturbance_correlation_horizon=wcfg.disturbance_correlation_horizon,
    )
    
    bank = None
    if memory_bank_path:
        bank = GeneticMemoryBank()
        bank.load(memory_bank_path)
        print({"info": f"Loaded memory bank from {memory_bank_path}"})

    genetic_memory = None
    if enable_genetic_memory:
        if bank:
            genetic_memory = bank.retrieve(profile)
            if genetic_memory is not None:
                genetic_memory = genetic_memory.to(dev)
                print({"info": f"Retrieved genetic memory for profile '{profile}' from bank"})
        
        if genetic_memory is None:
            genetic_memory = torch.zeros(1, mcfg.memory_slots, mcfg.memory_dim, device=dev)
            print({"info": "Initialized new genetic memory (zeros)"})

    amp_enabled = use_amp and dev.type == "cuda"

    def _lin_schedule(start: float, end: float, epoch_idx: int, power: float = 1.0) -> float:
        if tcfg.epochs <= 1:
            return end
        alpha = (epoch_idx / (tcfg.epochs - 1)) ** power
        return (1.0 - alpha) * start + alpha * end

    if not metrics_path:
        metrics_path = str(Path(save_path).with_suffix(".metrics.json"))
    metrics_records: list[dict] = []
    run_flags = {
        "profile": profile,
        "world_profile": world_profile.strip().lower() if world_profile.strip() else "custom-default",
        "gating_mode": mcfg.gating_mode,
        "topk_gating": mcfg.topk_gating,
        "enable_dmd_gating": mcfg.enable_dmd_gating,
        "enable_phase_gating": mcfg.enable_phase_gating,
        "enable_multi_scale_gating": mcfg.enable_multi_scale_gating,
        "enable_qat": enable_qat,
        "coevolution": coevolution,
        "enable_curriculum": enable_curriculum,
        "curriculum_power": curriculum_power,
        "enable_adaptive_loss": enable_adaptive_loss,
        "adaptive_loss_alpha": adaptive_loss_alpha,
        "enable_genetic_memory": enable_genetic_memory,
        "genetic_memory_decay": genetic_memory_decay,
        "embodiments": embodiment_names,
        "enable_embodiment_transfer_loss": enable_embodiment_transfer_loss,
        "transfer_loss_weight": transfer_loss_weight,
        "transfer_samples_per_step": transfer_samples_per_step,
        "transfer_fitness_weight": transfer_fitness_weight,
        "enable_autopoietic_objective": enable_autopoietic_objective,
        "autopoietic_loss_weight": autopoietic_loss_weight,
        "autopoietic_fitness_gain": autopoietic_fitness_gain,
        "autopoietic_self_repair_weight": autopoietic_self_repair_weight,
        "autopoietic_closure_weight": autopoietic_closure_weight,
        "autopoietic_resource_cycle_weight": autopoietic_resource_cycle_weight,
        "remap_loss_weight": remap_loss_weight,
        "detection_loss_weight": detection_loss_weight,
        "emergent_signal_loss_weight": emergent_signal_loss_weight,
        "genetic_memory_persistence_weight": genetic_memory_persistence_weight,
        "paging_loss_weight": paging_loss_weight,
        "noise_profile": noise_profile_resolved,
        "enable_noise_curriculum": enable_noise_curriculum,
        "noise_strength_start": noise_strength_start,
        "noise_strength_end": noise_strength_end,
        "force_curriculum_mode": force_curriculum_mode_resolved,
        "force_curriculum_strength_start": force_curriculum_strength_start,
        "force_curriculum_strength_end": force_curriculum_strength_end,
        "use_amp": amp_enabled,
        "allow_tf32": allow_tf32 and dev.type == "cuda",
        "compile_model": compile_model,
        "strict_device": strict_device,
        "init_weights": init_weights,
        "memory_bank_path": memory_bank_path,
        "constructor_tape_path": constructor_tape_path,
        "constructor_tape_version": None if constructor_tape is None else constructor_tape.version,
    }

    if not coevolution:
        model = ModelCore(**asdict(mcfg)).to(dev)
        inherited_tape_payload = None
        if init_weights:
            try:
                ckpt = _load_checkpoint_weights(model, init_weights, dev)
                inherited_tape_payload = ckpt.get("constructor_tape")
            except (ValueError, RuntimeError) as exc:
                raise typer.BadParameter(str(exc)) from exc
        
        if enable_qat:
            # QAT preparation must happen before optimizer creation
            model = prepare_qat_model(model)
            print({"info": "Quantization-Aware Training (QAT) enabled"})

        if compile_model and hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
        opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)
        scaler = _make_grad_scaler(amp_enabled)
        transfer_states = (
            _build_transfer_states(embodiment_names, mcfg, dev, seed=seed + 1234)
            if embodiment_names
            else None
        )
        
        controller = None
        if enable_adaptive_loss:
            controller = AdaptiveLossController(
                initial_weights={
                    "remap_loss_weight": remap_loss_weight,
                    "detection_loss_weight": detection_loss_weight,
                    "emergent_signal_loss_weight": emergent_signal_loss_weight,
                    "memory_persistence_loss_weight": memory_persistence_loss_weight,
                    "paging_loss_weight": paging_loss_weight,
                },
                alpha=adaptive_loss_alpha
            )

        for epoch in range(tcfg.epochs):
            if enable_curriculum:
                remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, epoch, power=curriculum_power)
                env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, epoch, power=curriculum_power)
            else:
                remap_probability = tcfg.remap_probability
                env_volatility = 0.0
            if noise_profile_resolved == "none":
                noise_strength = 0.0
            elif enable_noise_curriculum:
                noise_strength = _lin_schedule(noise_strength_start, noise_strength_end, epoch, power=curriculum_power)
            else:
                noise_strength = noise_strength_end
            if force_curriculum_mode_resolved == "none":
                force_curriculum_strength = 0.0
            elif enable_curriculum:
                force_curriculum_strength = _lin_schedule(
                    force_curriculum_strength_start,
                    force_curriculum_strength_end,
                    epoch,
                    power=curriculum_power,
                )
            else:
                force_curriculum_strength = force_curriculum_strength_end
            
            (
                mean_loss,
                memory_snapshot,
                mean_step_ms,
                mean_transfer_mismatch,
                mean_remap_loss,
                mean_detection_loss,
                mean_emergent_signal_loss,
                mean_memory_persistence_loss,
                mean_paging_loss,
                mean_autopoietic_score,
                mean_autopoietic_loss,
            ) = run_gradient_epoch(
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
                noise_profile=noise_profile_resolved,
                noise_strength=noise_strength,
                noise_seed=seed + epoch * 1009,
                use_amp=amp_enabled,
                scaler=scaler if amp_enabled else None,
                remap_loss_weight=remap_loss_weight,
                detection_loss_weight=detection_loss_weight,
                emergent_signal_loss_weight=emergent_signal_loss_weight,
                memory_persistence_loss_weight=memory_persistence_loss_weight,
                paging_loss_weight=paging_loss_weight,
                enable_autopoietic_objective=enable_autopoietic_objective,
                autopoietic_loss_weight=autopoietic_loss_weight,
                autopoietic_self_repair_weight=autopoietic_self_repair_weight,
                autopoietic_closure_weight=autopoietic_closure_weight,
                autopoietic_resource_cycle_weight=autopoietic_resource_cycle_weight,
                force_curriculum_mode=force_curriculum_mode_resolved,
                force_curriculum_strength=force_curriculum_strength,
            )
            
            if controller:
                new_weights = controller.step({
                    "loss": mean_loss,
                    "remap_loss": mean_remap_loss,
                    "detection_loss": mean_detection_loss,
                    "emergent_signal_loss": mean_emergent_signal_loss,
                    "memory_persistence_loss": mean_memory_persistence_loss,
                    "paging_loss": mean_paging_loss,
                })
                remap_loss_weight = new_weights["remap_loss_weight"]
                detection_loss_weight = new_weights["detection_loss_weight"]
                emergent_signal_loss_weight = new_weights["emergent_signal_loss_weight"]
                memory_persistence_loss_weight = new_weights["memory_persistence_loss_weight"]
                paging_loss_weight = new_weights["paging_loss_weight"]

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
                noise_profile=noise_profile_resolved,
                noise_strength=noise_strength,
                noise_seed=seed + epoch * 3001,
                autopoietic_fitness_weight=autopoietic_fitness_gain * autopoietic_loss_weight if enable_autopoietic_objective else 0.0,
                genetic_memory_persistence_weight=genetic_memory_persistence_weight,
                force_curriculum_mode=force_curriculum_mode_resolved,
                force_curriculum_strength=force_curriculum_strength,
            )
            metrics_records.append(
                {
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "fitness": fitness,
                    "mean_step_ms": mean_step_ms,
                    "mean_transfer_mismatch": mean_transfer_mismatch,
                    "mean_remap_loss": mean_remap_loss,
                    "mean_detection_loss": mean_detection_loss,
                    "mean_emergent_signal_loss": mean_emergent_signal_loss,
                    "mean_memory_persistence_loss": mean_memory_persistence_loss,
                    "mean_paging_loss": mean_paging_loss,
                    "mean_autopoietic_score": mean_autopoietic_score,
                    "autopoietic_loss_component": mean_autopoietic_loss,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                    "noise_strength": noise_strength,
                    "force_curriculum_strength": force_curriculum_strength,
                    "remap_loss_weight": remap_loss_weight,
                    "detection_loss_weight": detection_loss_weight,
                }
            )
            print(
                {
                    "epoch": epoch + 1,
                    "mean_loss": mean_loss,
                    "fitness": fitness,
                    "mean_step_ms": mean_step_ms,
                    "mean_transfer_mismatch": mean_transfer_mismatch,
                    "mean_remap_loss": mean_remap_loss,
                    "mean_detection_loss": mean_detection_loss,
                    "mean_emergent_signal_loss": mean_emergent_signal_loss,
                    "mean_memory_persistence_loss": mean_memory_persistence_loss,
                    "mean_paging_loss": mean_paging_loss,
                    "mean_autopoietic_score": mean_autopoietic_score,
                    "autopoietic_loss_component": mean_autopoietic_loss,
                    "device": str(dev),
                    "profile": profile,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                    "noise_profile": noise_profile_resolved,
                    "noise_strength": noise_strength,
                    "remap_loss_w": round(remap_loss_weight, 4),
                    "detect_loss_w": round(detection_loss_weight, 4),
                }
            )
        
        if bank:
            bank.store(profile, genetic_memory)
            bank.save(memory_bank_path)
            print({"info": f"Saved updated memory bank to {memory_bank_path}"})

        model_for_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model_for_save.state_dict(),
                "model_config": asdict(mcfg),
                "profile": profile,
                "genetic_memory": None if genetic_memory is None else genetic_memory.cpu(),
                "run_flags": run_flags,
                "constructor_tape": constructor_tape.to_payload() if constructor_tape is not None else inherited_tape_payload,
            },
            save_path,
        )
        Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_path).write_text(json.dumps({"flags": run_flags, "records": metrics_records}, indent=2), encoding="utf-8")
        print({"saved": save_path})
        return

    if init_weights:
        seed_model = ModelCore(**asdict(mcfg)).to(dev)
        inherited_tape_payload = None
        try:
            ckpt = _load_checkpoint_weights(seed_model, init_weights, dev)
            inherited_tape_payload = ckpt.get("constructor_tape")
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
    
    controller = None
    if enable_adaptive_loss:
        controller = AdaptiveLossController(
            initial_weights={
                "remap_loss_weight": remap_loss_weight,
                "detection_loss_weight": detection_loss_weight,
                "emergent_signal_loss_weight": emergent_signal_loss_weight,
                "memory_persistence_loss_weight": memory_persistence_loss_weight,
                "paging_loss_weight": paging_loss_weight,
            },
            alpha=adaptive_loss_alpha
        )

    elite_count = max(1, int(population_size * elite_fraction))
    for generation in range(tcfg.epochs):
        if enable_curriculum:
            remap_probability = _lin_schedule(remap_probability_start, remap_probability_end, generation, power=curriculum_power)
            env_volatility = _lin_schedule(env_volatility_start, env_volatility_end, generation, power=curriculum_power)
        else:
            remap_probability = tcfg.remap_probability
            env_volatility = 0.0
        if noise_profile_resolved == "none":
            noise_strength = 0.0
        elif enable_noise_curriculum:
            noise_strength = _lin_schedule(noise_strength_start, noise_strength_end, generation, power=curriculum_power)
        else:
            noise_strength = noise_strength_end
        if force_curriculum_mode_resolved == "none":
            force_curriculum_strength = 0.0
        elif enable_curriculum:
            force_curriculum_strength = _lin_schedule(
                force_curriculum_strength_start,
                force_curriculum_strength_end,
                generation,
                power=curriculum_power,
            )
        else:
            force_curriculum_strength = force_curriculum_strength_end
        
        mean_step_acc = 0.0
        mean_transfer_mismatch_acc = 0.0
        mean_remap_loss_acc = 0.0
        mean_detection_loss_acc = 0.0
        mean_emergent_signal_loss_acc = 0.0
        mean_memory_persistence_loss_acc = 0.0
        mean_paging_loss_acc = 0.0
        mean_epoch_loss_acc = 0.0

        for idx, model in enumerate(pop):
            model.train()
            (
                warmup_loss,
                _,
                step_ms,
                mean_transfer_mismatch,
                mean_remap_loss,
                mean_detection_loss,
                mean_emergent_signal_loss,
                mean_memory_persistence_loss,
                mean_paging_loss,
                mean_autopoietic_score,
                mean_autopoietic_loss,
            ) = run_gradient_epoch(
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
                noise_profile=noise_profile_resolved,
                noise_strength=noise_strength,
                noise_seed=seed + generation * 1009 + idx * 37,
                use_amp=amp_enabled,
                scaler=scalers[idx],
                remap_loss_weight=remap_loss_weight,
                detection_loss_weight=detection_loss_weight,
                emergent_signal_loss_weight=emergent_signal_loss_weight,
                memory_persistence_loss_weight=memory_persistence_loss_weight,
                paging_loss_weight=paging_loss_weight,
                enable_autopoietic_objective=enable_autopoietic_objective,
                autopoietic_loss_weight=autopoietic_loss_weight,
                autopoietic_self_repair_weight=autopoietic_self_repair_weight,
                autopoietic_closure_weight=autopoietic_closure_weight,
                autopoietic_resource_cycle_weight=autopoietic_resource_cycle_weight,
                force_curriculum_mode=force_curriculum_mode_resolved,
                force_curriculum_strength=force_curriculum_strength,
            )
            mean_step_acc += step_ms
            mean_transfer_mismatch_acc += mean_transfer_mismatch
            mean_remap_loss_acc += mean_remap_loss
            mean_detection_loss_acc += mean_detection_loss
            mean_emergent_signal_loss_acc += mean_emergent_signal_loss
            mean_memory_persistence_loss_acc += mean_memory_persistence_loss
            mean_paging_loss_acc += mean_paging_loss
            mean_epoch_loss_acc += warmup_loss

            print(
                {
                    "generation": generation + 1,
                    "agent": idx,
                    "warmup_loss": warmup_loss,
                    "mean_transfer_mismatch": mean_transfer_mismatch,
                    "mean_remap_loss": mean_remap_loss,
                    "mean_detection_loss": mean_detection_loss,
                    "mean_emergent_signal_loss": mean_emergent_signal_loss,
                    "mean_memory_persistence_loss": mean_memory_persistence_loss,
                    "mean_paging_loss": mean_paging_loss,
                    "mean_autopoietic_score": mean_autopoietic_score,
                    "autopoietic_loss_component": mean_autopoietic_loss,
                    "remap_probability": remap_probability,
                    "env_volatility": env_volatility,
                    "noise_strength": noise_strength,
                    "force_curriculum_strength": force_curriculum_strength,
                }
            )

        if controller:
            new_weights = controller.step({
                "loss": mean_epoch_loss_acc / population_size,
                "remap_loss": mean_remap_loss_acc / population_size,
                "detection_loss": mean_detection_loss_acc / population_size,
                "emergent_signal_loss": mean_emergent_signal_loss_acc / population_size,
                "memory_persistence_loss": mean_memory_persistence_loss_acc / population_size,
                "paging_loss": mean_paging_loss_acc / population_size,
            })
            remap_loss_weight = new_weights["remap_loss_weight"]
            detection_loss_weight = new_weights["detection_loss_weight"]
            emergent_signal_loss_weight = new_weights["emergent_signal_loss_weight"]
            memory_persistence_loss_weight = new_weights["memory_persistence_loss_weight"]
            paging_loss_weight = new_weights["paging_loss_weight"]

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
                noise_profile=noise_profile_resolved,
                noise_strength=noise_strength,
                noise_seed=seed + generation * 3001 + idx * 53,
                autopoietic_fitness_weight=autopoietic_fitness_gain * autopoietic_loss_weight if enable_autopoietic_objective else 0.0,
                genetic_memory_persistence_weight=genetic_memory_persistence_weight,
                force_curriculum_mode=force_curriculum_mode_resolved,
                force_curriculum_strength=force_curriculum_strength,
            )
            for idx, model in enumerate(pop)
        ]
        rank = sorted(range(len(pop)), key=lambda i: scores[i], reverse=True)
        elites = rank[:elite_count]
        best_idx = elites[0]
        print({
            "generation": generation + 1, 
            "best_agent": best_idx, 
            "best_fitness": scores[best_idx], 
            "mean_fitness": sum(scores) / len(scores),
            "remap_loss_w": round(remap_loss_weight, 4),
            "detect_loss_w": round(detection_loss_weight, 4),
        })
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
                "noise_strength": noise_strength,
                "force_curriculum_strength": force_curriculum_strength,
                "remap_loss_weight": remap_loss_weight,
                "detection_loss_weight": detection_loss_weight,
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
            noise_profile=noise_profile_resolved,
            noise_strength=noise_strength_end if noise_profile_resolved != "none" else 0.0,
            noise_seed=seed + 99991 + idx * 97,
            autopoietic_fitness_weight=autopoietic_fitness_gain * autopoietic_loss_weight if enable_autopoietic_objective else 0.0,
            genetic_memory_persistence_weight=genetic_memory_persistence_weight,
            force_curriculum_mode=force_curriculum_mode_resolved,
            force_curriculum_strength=force_curriculum_strength_end if force_curriculum_mode_resolved != "none" else 0.0,
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
            "constructor_tape": constructor_tape.to_payload() if constructor_tape is not None else inherited_tape_payload if init_weights else None,
        },
        save_path,
    )
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps({"flags": run_flags, "records": metrics_records}, indent=2), encoding="utf-8")
    print({"saved": save_path, "best_agent": best_idx, "best_fitness": final_scores[best_idx]})


if __name__ == "__main__":
    app()

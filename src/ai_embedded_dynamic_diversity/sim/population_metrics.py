from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ai_embedded_dynamic_diversity.models.core import ModelCore
    from ai_embedded_dynamic_diversity.sim.embodiments import Embodiment


def _pairwise_cosine_distances(vecs: list[torch.Tensor]) -> float:
    """Mean pairwise cosine distance (1 − similarity) for a list of 1-D vectors."""
    n = len(vecs)
    if n < 2:
        return 0.0
    stacked = torch.stack([v.float().flatten() for v in vecs], dim=0)  # [N, D]
    norms = stacked.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = stacked / norms
    sim_matrix = normed @ normed.t()  # [N, N]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim_matrix[i, j].item())
            count += 1
    return total / max(1, count)


def _lineage_entropy(lineage: list[int]) -> float:
    """Shannon entropy of parent index frequencies, normalised to [0,1]."""
    if not lineage:
        return 0.0
    counts: dict[int, int] = {}
    for p in lineage:
        counts[p] = counts.get(p, 0) + 1
    total = len(lineage)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    max_entropy = math.log(max(2, len(lineage)))
    return entropy / max_entropy


def _behavioral_species_count(io_traces: list[torch.Tensor], distance_threshold: float = 0.3) -> int:
    """Single-linkage clustering count: number of clusters at given cosine distance threshold."""
    n = len(io_traces)
    if n == 0:
        return 0
    means = [t.float().mean(dim=0) if t.ndim > 1 else t.float() for t in io_traces]
    # Union-find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            vi = means[i].flatten()
            vj = means[j].flatten()
            ni = vi.norm().clamp(min=1e-8)
            nj = vj.norm().clamp(min=1e-8)
            cos_dist = 1.0 - float((vi / ni).dot(vj / nj).item())
            if cos_dist < distance_threshold:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pi] = pj
    return len({find(i) for i in range(n)})


def genetic_diversity_index(
    population: list[ModelCore],
    io_traces: list[torch.Tensor],
    lineage: list[int],
) -> dict[str, float]:
    """
    Compute population-level genetic and behavioural diversity metrics.

    Args:
        population: list of ModelCore instances (one per agent).
        io_traces:  list of Tensors [T, io_channels] — per-agent IO outputs over last episode.
        lineage:    list of parent indices for each agent this generation (same length as population).

    Returns:
        Dict with keys: weight_div, behavior_div, lineage_entropy, species_count, species_frac, gdi
    """
    n = len(population)
    if n == 0:
        return {
            "weight_div": 0.0,
            "behavior_div": 0.0,
            "lineage_entropy": 0.0,
            "species_count": 0,
            "species_frac": 0.0,
            "gdi": 0.0,
        }

    # Weight diversity: mean pairwise cosine distance of flattened param vectors
    param_vecs: list[torch.Tensor] = []
    for model in population:
        with torch.no_grad():
            flat = torch.cat([p.data.cpu().flatten() for p in model.parameters()])
        param_vecs.append(flat)
    weight_div = _pairwise_cosine_distances(param_vecs)

    # Behavioural diversity: mean pairwise cosine distance of IO-trace means
    behavior_div = 0.0
    if io_traces:
        behavior_div = _pairwise_cosine_distances(io_traces)

    # Lineage entropy
    lin_entropy = _lineage_entropy(lineage)

    # Species count via single-linkage clustering
    species = _behavioral_species_count(io_traces) if io_traces else 1
    species_frac = species / max(1, n)

    gdi = (
        0.35 * weight_div
        + 0.35 * behavior_div
        + 0.20 * lin_entropy
        + 0.10 * species_frac
    )

    return {
        "weight_div": weight_div,
        "behavior_div": behavior_div,
        "lineage_entropy": lin_entropy,
        "species_count": species,
        "species_frac": species_frac,
        "gdi": gdi,
    }


def dof_coordination_metrics(
    io_traces: list[torch.Tensor],
    io_channels: int,
    activation_threshold: float = 0.05,
) -> dict:
    """
    Population-level DOF coordination metrics computed from IO traces.

    Args:
        io_traces:   list of Tensors [T, io_channels] — one per agent.
        io_channels: expected number of IO channels (for shape validation).
        activation_threshold: minimum mean |activation| to count a channel as active.

    Returns dict with:
        per_channel_firing_rate  — [io_channels] mean |activation| across agents & time
        channel_entropy          — Shannon entropy of per-channel mean activations (normalised)
        co_activation_top5       — top-5 correlated channel pairs [(i, j, pearson_r)]
        coverage_fraction        — fraction of channels exceeding activation_threshold
    """
    if not io_traces:
        return {
            "per_channel_firing_rate": [0.0] * io_channels,
            "channel_entropy": 0.0,
            "co_activation_top5": [],
            "coverage_fraction": 0.0,
        }

    # Stack all traces: [N*T, io_channels]
    stacked = torch.cat(
        [t.float() if t.ndim == 2 else t.float().unsqueeze(0) for t in io_traces],
        dim=0,
    )
    if stacked.shape[1] != io_channels:
        # Truncate or pad to expected channels
        if stacked.shape[1] > io_channels:
            stacked = stacked[:, :io_channels]
        else:
            pad = torch.zeros(stacked.shape[0], io_channels - stacked.shape[1], device=stacked.device)
            stacked = torch.cat([stacked, pad], dim=1)

    per_channel = stacked.abs().mean(dim=0)  # [io_channels]
    per_channel_list = per_channel.tolist()

    # Shannon entropy of normalised per-channel activations
    total = per_channel.sum().clamp(min=1e-8)
    probs = per_channel / total
    entropy = float(-(probs * (probs + 1e-8).log()).sum().item())
    max_entropy = math.log(max(2, io_channels))
    channel_entropy = min(1.0, entropy / max_entropy)

    # Coverage: fraction of channels active above threshold
    coverage_fraction = float((per_channel > activation_threshold).float().mean().item())

    # Top-5 Pearson correlations between channel pairs
    if stacked.shape[0] > 1:
        mean = stacked.mean(dim=0, keepdim=True)
        centred = stacked - mean
        std = centred.std(dim=0).clamp(min=1e-8)
        normed = centred / std           # [N*T, io_channels]
        cov = (normed.T @ normed) / max(1, stacked.shape[0] - 1)  # [C, C]
        # Zero out diagonal
        cov.fill_diagonal_(0.0)
        flat = cov.flatten()
        top_vals, top_idx = flat.abs().topk(min(5, flat.numel()))
        pairs = []
        C = io_channels
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            i, j = divmod(int(idx), C)
            if i < j:
                pairs.append((i, j, round(float(cov[i, j].item()), 4)))
        co_activation_top5 = pairs[:5]
    else:
        co_activation_top5 = []

    return {
        "per_channel_firing_rate": [round(v, 5) for v in per_channel_list],
        "channel_entropy": round(channel_entropy, 4),
        "co_activation_top5": co_activation_top5,
        "coverage_fraction": round(coverage_fraction, 4),
    }


def embodiment_diversity_breakdown(
    io_traces_by_embodiment: dict[str, list[torch.Tensor]],
    activation_threshold: float = 0.05,
) -> dict:
    """
    Per-embodiment behavioural diversity and cross-embodiment consistency.

    Args:
        io_traces_by_embodiment: dict mapping embodiment name → list of [T, io_channels]
            tensors (one per agent).
        activation_threshold: minimum mean |activation| to count a channel as active.

    Returns dict with:
        per_embodiment — {name: {"behavior_div": float, "coverage_fraction": float}}
        cross_embodiment_consistency — mean cosine similarity of per-embodiment mean
            vectors across the population (high = agent behaves similarly across
            embodiments, low = specialised)
    """
    per_embodiment: dict[str, dict] = {}
    emb_means: list[torch.Tensor] = []

    for emb_name, traces in io_traces_by_embodiment.items():
        if not traces:
            per_embodiment[emb_name] = {"behavior_div": 0.0, "coverage_fraction": 0.0}
            continue
        behavior_div = _pairwise_cosine_distances(traces)
        stacked = torch.cat([t.float() if t.ndim == 2 else t.float().unsqueeze(0) for t in traces], dim=0)
        per_channel = stacked.abs().mean(dim=0)
        coverage = float((per_channel > activation_threshold).float().mean().item())
        per_embodiment[emb_name] = {"behavior_div": round(behavior_div, 4), "coverage_fraction": round(coverage, 4)}
        # Population mean for this embodiment (mean across all agents × all steps)
        emb_means.append(stacked.mean(dim=0))

    # Cross-embodiment consistency: how similar are per-embodiment mean activation vectors?
    cross_consistency = 0.0
    if len(emb_means) >= 2:
        n = len(emb_means)
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                ni = emb_means[i].norm().clamp(min=1e-8)
                nj = emb_means[j].norm().clamp(min=1e-8)
                sim = float(((emb_means[i] / ni).dot(emb_means[j] / nj)).item())
                total += sim
                count += 1
        cross_consistency = round(total / max(1, count), 4)

    return {"per_embodiment": per_embodiment, "cross_embodiment_consistency": cross_consistency}


def dof_coverage_per_embodiment(
    io_traces_by_embodiment: dict[str, list[torch.Tensor]],
    activation_threshold: float = 0.05,
) -> dict[str, list[float]]:
    """Mean |io| per channel grouped by embodiment. Returns {name: [firing_rate_per_channel]}."""
    result: dict[str, list[float]] = {}
    for emb_name, traces in io_traces_by_embodiment.items():
        if not traces:
            result[emb_name] = []
            continue
        stacked = torch.cat([t.float() if t.ndim == 2 else t.float().unsqueeze(0) for t in traces], dim=0)
        result[emb_name] = [round(v, 5) for v in stacked.abs().mean(dim=0).tolist()]
    return result

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ai_embedded_dynamic_diversity.models.core import ModelCore


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

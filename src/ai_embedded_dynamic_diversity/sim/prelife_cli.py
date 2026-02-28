from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _substrate_params(substrate: str) -> dict[str, float]:
    name = substrate.strip().lower()
    table = {
        "bytecode_dense": {
            "resource_gain": 0.85,
            "self_mod_rate": 0.09,
            "replication_rate": 0.11,
            "symbio_rate": 0.025,
            "mutation_rate": 0.018,
        },
        "bytecode_sparse": {
            "resource_gain": 0.45,
            "self_mod_rate": 0.05,
            "replication_rate": 0.05,
            "symbio_rate": 0.012,
            "mutation_rate": 0.012,
        },
        "resource_limited": {
            "resource_gain": 0.25,
            "self_mod_rate": 0.03,
            "replication_rate": 0.025,
            "symbio_rate": 0.010,
            "mutation_rate": 0.010,
        },
        "sublike_control": {
            "resource_gain": 0.18,
            "self_mod_rate": 0.01,
            "replication_rate": 0.008,
            "symbio_rate": 0.004,
            "mutation_rate": 0.006,
        },
    }
    if name not in table:
        raise ValueError(f"Unknown substrate '{substrate}'")
    return table[name]


def _random_genome(rng: random.Random, length: int = 12) -> str:
    alphabet = "ABCMRSX"
    return "".join(rng.choice(alphabet) for _ in range(length))


def _copy_genome(genome: str, mutation_rate: float, rng: random.Random) -> tuple[str, float]:
    chars = list(genome)
    edits = 0
    for idx, ch in enumerate(chars):
        if rng.random() < mutation_rate and ch.isalpha():
            edits += 1
            chars[idx] = rng.choice("ABCMRSX")
    fidelity = 1.0 - edits / max(1, len(chars))
    return "".join(chars), max(0.0, min(1.0, fidelity))


def detect_self_modification_events(prev_genome: str, next_genome: str) -> bool:
    return prev_genome != next_genome


def detect_symbiogenesis_event(parent_ids: tuple[int, ...]) -> bool:
    return len(set(parent_ids)) >= 2


def _line_slope(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    x_mean = (n - 1) * 0.5
    y_mean = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den <= 1e-12:
        return 0.0
    return num / den


def run_prelife_simulation(
    substrate: str,
    steps: int,
    seed: int,
    initial_agents: int = 16,
    max_agents: int = 128,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be > 0")
    if initial_agents <= 0:
        raise ValueError("initial_agents must be > 0")
    if max_agents < initial_agents:
        raise ValueError("max_agents must be >= initial_agents")

    params = _substrate_params(substrate)
    rng = random.Random(seed)
    agents: list[dict] = []
    next_id = 1
    for _ in range(initial_agents):
        genome = _random_genome(rng)
        agents.append({"id": next_id, "genome": genome, "energy": rng.uniform(0.2, 0.6), "depth": 1, "parents": (next_id,)})
        next_id += 1

    replication_events = 0
    self_mod_events = 0
    symbio_events = 0
    first_replication_step: int | None = None
    fidelities: list[float] = []
    novelty_series: list[float] = []
    lineage_depth_series: list[float] = []
    population_series: list[int] = []

    for step in range(1, steps + 1):
        current_unique = len({agent["genome"] for agent in agents})
        novelty_series.append(float(current_unique))
        lineage_depth_series.append(float(sorted(agent["depth"] for agent in agents)[len(agents) // 2]))
        population_series.append(len(agents))

        newborns: list[dict] = []
        for agent in agents:
            agent["energy"] = max(0.0, float(agent["energy"]) + rng.uniform(0.0, params["resource_gain"]) - 0.2)

            base_self_mod = params["self_mod_rate"] * (1.25 if "M" in agent["genome"] else 1.0)
            if rng.random() < base_self_mod:
                old = str(agent["genome"])
                new_genome, _ = _copy_genome(old, params["mutation_rate"] * 1.8, rng)
                if detect_self_modification_events(old, new_genome):
                    self_mod_events += 1
                    agent["genome"] = new_genome

            can_replicate = agent["energy"] > 0.75 and "R" in agent["genome"]
            if can_replicate and rng.random() < params["replication_rate"]:
                child_genome, fidelity = _copy_genome(str(agent["genome"]), params["mutation_rate"], rng)
                fidelities.append(fidelity)
                replication_events += 1
                if first_replication_step is None:
                    first_replication_step = step
                child = {
                    "id": next_id,
                    "genome": child_genome,
                    "energy": float(agent["energy"]) * 0.5,
                    "depth": int(agent["depth"]) + 1,
                    "parents": (int(agent["id"]),),
                }
                next_id += 1
                newborns.append(child)
                agent["energy"] = float(agent["energy"]) * 0.5

        agents.extend(newborns)

        if len(agents) >= 2 and rng.random() < params["symbio_rate"]:
            i1 = rng.randrange(len(agents))
            i2 = rng.randrange(len(agents))
            if i1 != i2:
                a = agents[i1]
                b = agents[i2]
                half_a = str(a["genome"])[: len(str(a["genome"])) // 2]
                half_b = str(b["genome"])[len(str(b["genome"])) // 2 :]
                fused = half_a + half_b
                parents = (int(a["id"]), int(b["id"]))
                child = {
                    "id": next_id,
                    "genome": fused,
                    "energy": (float(a["energy"]) + float(b["energy"])) * 0.25,
                    "depth": max(int(a["depth"]), int(b["depth"])) + 1,
                    "parents": parents,
                }
                next_id += 1
                if detect_symbiogenesis_event(parents):
                    symbio_events += 1
                agents.append(child)

        agents = sorted(agents, key=lambda item: float(item["energy"]), reverse=True)[:max_agents]

    p50_depth = sorted(int(agent["depth"]) for agent in agents)[len(agents) // 2]
    p95_depth = sorted(int(agent["depth"]) for agent in agents)[int(0.95 * (len(agents) - 1))]
    novelty_growth = _line_slope(novelty_series)
    window = max(5, min(steps, 20))
    phase_step = None
    rep_cumulative = 0
    for idx in range(steps):
        if idx + 1 <= window:
            rep_cumulative += 0
        if first_replication_step is not None and idx + 1 >= first_replication_step:
            rep_cumulative += 1
        if (rep_cumulative / max(1, idx + 1)) >= 0.05:
            phase_step = idx + 1
            break

    return {
        "config": {
            "substrate": substrate,
            "steps": steps,
            "seed": seed,
            "initial_agents": initial_agents,
            "max_agents": max_agents,
        },
        "metrics": {
            "first_replication_step": first_replication_step,
            "replication_rate": replication_events / max(1.0, float(steps)),
            "self_modification_rate": self_mod_events / max(1.0, float(steps)),
            "lineage_depth_p50": float(p50_depth),
            "lineage_depth_p95": float(p95_depth),
            "novelty_growth_slope": novelty_growth,
            "description_copy_fidelity": sum(fidelities) / max(1, len(fidelities)),
            "symbiogenesis_event_count": symbio_events,
            "substrate_phase_transition_step": phase_step,
            "final_population": len(agents),
            "unique_genomes_final": len({agent["genome"] for agent in agents}),
        },
        "series": {
            "novelty": novelty_series,
            "lineage_depth_median": lineage_depth_series,
            "population": population_series,
        },
    }


@app.command()
def run(
    substrate: str = "bytecode_dense",
    steps: int = 400,
    seed: int = 17,
    initial_agents: int = 16,
    max_agents: int = 128,
    output: str = "artifacts/prelife-emergence.json",
) -> None:
    try:
        payload = run_prelife_simulation(
            substrate=substrate,
            steps=steps,
            seed=seed,
            initial_agents=initial_agents,
            max_agents=max_agents,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        {
            "output": str(out),
            "substrate": substrate,
            "replication_rate": payload["metrics"]["replication_rate"],
            "self_modification_rate": payload["metrics"]["self_modification_rate"],
            "symbiogenesis_event_count": payload["metrics"]["symbiogenesis_event_count"],
        }
    )


@app.command()
def report(
    input_path: str = "artifacts/prelife-emergence.json",
    markdown_out: str = "artifacts/prelife-emergence.md",
    csv_out: str = "artifacts/prelife-emergence.csv",
) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    metrics = payload.get("metrics", {})
    row = {
        "substrate": cfg.get("substrate", ""),
        "steps": cfg.get("steps", 0),
        "seed": cfg.get("seed", 0),
        "first_replication_step": metrics.get("first_replication_step", ""),
        "replication_rate": metrics.get("replication_rate", 0.0),
        "self_modification_rate": metrics.get("self_modification_rate", 0.0),
        "lineage_depth_p50": metrics.get("lineage_depth_p50", 0.0),
        "lineage_depth_p95": metrics.get("lineage_depth_p95", 0.0),
        "novelty_growth_slope": metrics.get("novelty_growth_slope", 0.0),
        "description_copy_fidelity": metrics.get("description_copy_fidelity", 0.0),
        "symbiogenesis_event_count": metrics.get("symbiogenesis_event_count", 0),
        "substrate_phase_transition_step": metrics.get("substrate_phase_transition_step", ""),
        "final_population": metrics.get("final_population", 0),
        "unique_genomes_final": metrics.get("unique_genomes_final", 0),
    }

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_out).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_lines = [
        "# Pre-Life Emergence Report",
        "",
        f"Source: `{input_path}`",
        "",
        "## Summary",
        "",
        f"- substrate: `{row['substrate']}`",
        f"- steps: `{row['steps']}`",
        f"- first_replication_step: `{row['first_replication_step']}`",
        f"- replication_rate: `{row['replication_rate']:.6f}`",
        f"- self_modification_rate: `{row['self_modification_rate']:.6f}`",
        f"- novelty_growth_slope: `{row['novelty_growth_slope']:.6f}`",
        f"- description_copy_fidelity: `{row['description_copy_fidelity']:.6f}`",
        f"- symbiogenesis_event_count: `{row['symbiogenesis_event_count']}`",
        "",
    ]
    Path(markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_out).write_text("\n".join(md_lines), encoding="utf-8")
    print({"markdown": markdown_out, "csv": csv_out, "substrate": row["substrate"]})


if __name__ == "__main__":
    app()

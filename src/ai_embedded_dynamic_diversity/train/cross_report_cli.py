from __future__ import annotations

import csv
import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False)


def _fmt(value: float) -> str:
    return f"{value:.6f}"


@app.command()
def run(
    input_path: str = "artifacts/cross-eval-summary.json",
    markdown_out: str = "artifacts/cross-eval-report.md",
    csv_out: str = "artifacts/cross-eval-report.csv",
) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    ranked = payload.get("ranked", [])
    if not ranked:
        raise typer.BadParameter(f"No ranked entries in {input_path}")

    best = ranked[0]
    cfg = payload.get("config", {})
    emb_names = list(cfg.get("embodiments", []))
    capability_profile = str(cfg.get("capability_profile", "none")).strip().lower()
    capability_enabled = capability_profile != "none"
    capability_score_weight = float(cfg.get("capability_score_weight", 0.0))

    rows = []
    for idx, item in enumerate(ranked, start=1):
        transfer_unweighted = item.get("overall_transfer_score_unweighted", item["overall_transfer_score"])
        transfer_weighted = item.get("overall_transfer_score_weighted", transfer_unweighted)
        overall_capability_score = float(item.get("overall_capability_score", 0.0))
        row = {
            "rank": idx,
            "checkpoint": item["checkpoint"],
            "overall_ranking_score": item["overall_transfer_score"],
            "overall_transfer_score": item["overall_transfer_score"],
            "overall_transfer_score_weighted": transfer_weighted,
            "overall_transfer_score_unweighted": transfer_unweighted,
            "overall_capability_score": overall_capability_score,
            "overall_mean_mismatch": item["overall_mean_mismatch"],
            "overall_mean_vitality": item["overall_mean_vitality"],
            "overall_recovery": item["overall_recovery"],
            "delta_vs_best": item["overall_transfer_score"] - best["overall_transfer_score"],
            "flags": json.dumps(item.get("flags", {}), sort_keys=True),
        }
        if capability_enabled and emb_names:
            rel_vals: list[float] = []
            auc_vals: list[float] = []
            evasion_vals: list[float] = []
            for emb in emb_names:
                emb_stats = item.get("by_embodiment", {}).get(emb, {})
                rel_vals.append(float(emb_stats.get("signal_reliability", 0.0)))
                auc_vals.append(float(emb_stats.get("signal_detection_auc", 0.0)))
                evasion_vals.append(float(emb_stats.get("evasion_success", 0.0)))
            row["overall_signal_reliability"] = sum(rel_vals) / max(1, len(rel_vals))
            row["overall_signal_detection_auc"] = sum(auc_vals) / max(1, len(auc_vals))
            row["overall_evasion_success"] = sum(evasion_vals) / max(1, len(evasion_vals))
        for emb in emb_names:
            emb_stats = item.get("by_embodiment", {}).get(emb, {})
            best_stats = best.get("by_embodiment", {}).get(emb, {})
            row[f"{emb}_transfer"] = emb_stats.get("transfer_score", 0.0)
            row[f"{emb}_delta"] = emb_stats.get("transfer_score", 0.0) - best_stats.get("transfer_score", 0.0)
            if capability_enabled:
                row[f"{emb}_capability"] = emb_stats.get("capability_score", 0.0)
                row[f"{emb}_signal_reliability"] = emb_stats.get("signal_reliability", 0.0)
                row[f"{emb}_signal_detection_auc"] = emb_stats.get("signal_detection_auc", 0.0)
                row[f"{emb}_evasion_success"] = emb_stats.get("evasion_success", 0.0)
        rows.append(row)

    Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
    with Path(csv_out).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = []
    md_lines.append("# Cross-Eval Report")
    md_lines.append("")
    md_lines.append(f"Source: `{input_path}`")
    md_lines.append("")
    weights = cfg.get("embodiment_weights", {})
    if weights:
        md_lines.append(f"Embodiment weights: `{json.dumps(weights, sort_keys=True)}`")
        md_lines.append("")
    if capability_enabled:
        md_lines.append(f"Capability profile: `{capability_profile}`")
        md_lines.append(f"Capability score weight: `{capability_score_weight}`")
        md_lines.append("")
    md_lines.append("## Top 5")
    md_lines.append("")
    if capability_enabled:
        md_lines.append("| Rank | Checkpoint | Ranking Score | Transfer (Weighted) | Transfer (Unweighted) | Capability | Delta vs Best | Mismatch | Vitality | Recovery |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows[:5]:
            md_lines.append(
                "| "
                + f"{row['rank']} | `{row['checkpoint']}` | {_fmt(row['overall_ranking_score'])} | {_fmt(row['overall_transfer_score_weighted'])} | {_fmt(row['overall_transfer_score_unweighted'])} | {_fmt(row['overall_capability_score'])} | {_fmt(row['delta_vs_best'])} | {_fmt(row['overall_mean_mismatch'])} | {_fmt(row['overall_mean_vitality'])} | {_fmt(row['overall_recovery'])} |"
            )
    else:
        md_lines.append("| Rank | Checkpoint | Transfer (Ranking) | Transfer (Unweighted) | Delta vs Best | Mismatch | Vitality | Recovery |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in rows[:5]:
            md_lines.append(
                "| "
                + f"{row['rank']} | `{row['checkpoint']}` | {_fmt(row['overall_transfer_score'])} | {_fmt(row['overall_transfer_score_unweighted'])} | {_fmt(row['delta_vs_best'])} | {_fmt(row['overall_mean_mismatch'])} | {_fmt(row['overall_mean_vitality'])} | {_fmt(row['overall_recovery'])} |"
            )

    if capability_enabled:
        md_lines.append("")
        md_lines.append("## Capability Proxies (Top 5)")
        md_lines.append("")
        md_lines.append("| Rank | Checkpoint | Capability | Signal Reliability | Signal Detection AUC | Evasion Success |")
        md_lines.append("|---|---|---:|---:|---:|---:|")
        for row in rows[:5]:
            md_lines.append(
                "| "
                + f"{row['rank']} | `{row['checkpoint']}` | {_fmt(row['overall_capability_score'])} | {_fmt(row.get('overall_signal_reliability', 0.0))} | {_fmt(row.get('overall_signal_detection_auc', 0.0))} | {_fmt(row.get('overall_evasion_success', 0.0))} |"
            )

    md_lines.append("")
    md_lines.append("## Per-Embodiment Deltas vs Best")
    md_lines.append("")
    header = "| Rank | Checkpoint | " + " | ".join(f"{emb} delta" for emb in emb_names) + " |"
    sep = "|---|---|" + "|".join(["---:" for _ in emb_names]) + "|"
    md_lines.append(header)
    md_lines.append(sep)
    for row in rows[:10]:
        vals = " | ".join(_fmt(row[f"{emb}_delta"]) for emb in emb_names)
        md_lines.append(f"| {row['rank']} | `{row['checkpoint']}` | {vals} |")

    md_lines.append("")
    md_lines.append("## Best Flags")
    md_lines.append("")
    md_lines.append("```json")
    md_lines.append(json.dumps(best.get("flags", {}), indent=2, sort_keys=True))
    md_lines.append("```")

    Path(markdown_out).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print({"markdown": markdown_out, "csv": csv_out, "best_checkpoint": best["checkpoint"], "best_transfer": best["overall_transfer_score"]})


if __name__ == "__main__":
    app()

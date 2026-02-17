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
    emb_names = list(payload.get("config", {}).get("embodiments", []))

    rows = []
    for idx, item in enumerate(ranked, start=1):
        transfer_unweighted = item.get("overall_transfer_score_unweighted", item["overall_transfer_score"])
        row = {
            "rank": idx,
            "checkpoint": item["checkpoint"],
            "overall_transfer_score": item["overall_transfer_score"],
            "overall_transfer_score_unweighted": transfer_unweighted,
            "overall_mean_mismatch": item["overall_mean_mismatch"],
            "overall_mean_vitality": item["overall_mean_vitality"],
            "overall_recovery": item["overall_recovery"],
            "delta_vs_best": item["overall_transfer_score"] - best["overall_transfer_score"],
            "flags": json.dumps(item.get("flags", {}), sort_keys=True),
        }
        for emb in emb_names:
            emb_stats = item.get("by_embodiment", {}).get(emb, {})
            best_stats = best.get("by_embodiment", {}).get(emb, {})
            row[f"{emb}_transfer"] = emb_stats.get("transfer_score", 0.0)
            row[f"{emb}_delta"] = emb_stats.get("transfer_score", 0.0) - best_stats.get("transfer_score", 0.0)
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
    weights = payload.get("config", {}).get("embodiment_weights", {})
    if weights:
        md_lines.append(f"Embodiment weights: `{json.dumps(weights, sort_keys=True)}`")
        md_lines.append("")
    md_lines.append("## Top 5")
    md_lines.append("")
    md_lines.append("| Rank | Checkpoint | Transfer (Ranking) | Transfer (Unweighted) | Delta vs Best | Mismatch | Vitality | Recovery |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows[:5]:
        md_lines.append(
            "| "
            + f"{row['rank']} | `{row['checkpoint']}` | {_fmt(row['overall_transfer_score'])} | {_fmt(row['overall_transfer_score_unweighted'])} | {_fmt(row['delta_vs_best'])} | {_fmt(row['overall_mean_mismatch'])} | {_fmt(row['overall_mean_vitality'])} | {_fmt(row['overall_recovery'])} |"
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

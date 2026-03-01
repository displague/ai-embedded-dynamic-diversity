from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    return float(math.sqrt(max(0.0, var)))


def _load_metrics(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize storyboard compare-left/compare-right metrics deltas.")
    parser.add_argument("--storyboard-dir", required=True, help="Directory containing *compare-left-metrics.json files.")
    parser.add_argument("--left-tag", default="model-core-champion-v07", help="Left checkpoint tag in file names.")
    parser.add_argument("--right-tag", default="model-core-champion-v09", help="Right checkpoint tag in file names.")
    parser.add_argument("--top-k", type=int, default=10, help="Rows to include in best/worst lists.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--output-md", default="", help="Optional markdown output path.")
    args = parser.parse_args()

    root = Path(args.storyboard_dir)
    if not root.exists():
        raise SystemExit(f"storyboard directory not found: {root}")

    rows: list[dict] = []
    pattern = f"*-{args.left_tag}-compare-left-metrics.json"
    for left_path in sorted(root.glob(pattern)):
        suffix = f"-{args.left_tag}-compare-left-metrics.json"
        prefix = left_path.name[: -len(suffix)]
        right_name = f"{prefix}-{args.right_tag}-compare-right-metrics.json"
        right_path = root / right_name
        if not right_path.exists():
            continue

        left = _load_metrics(left_path)
        right = _load_metrics(right_path)
        if not left or not right:
            continue

        l_mismatch = [float(x.get("mismatch", 0.0)) for x in left]
        r_mismatch = [float(x.get("mismatch", 0.0)) for x in right]
        l_vitality = [float(x.get("vitality", 0.0)) for x in left]
        r_vitality = [float(x.get("vitality", 0.0)) for x in right]

        scen_emb = prefix.split("-", maxsplit=1)
        scenario = scen_emb[0] if scen_emb else prefix
        embodiment = scen_emb[1] if len(scen_emb) > 1 else "unknown"

        rows.append(
            {
                "key": prefix,
                "scenario": scenario,
                "embodiment": embodiment,
                "mean_mismatch_left": _mean(l_mismatch),
                "mean_mismatch_right": _mean(r_mismatch),
                "mean_mismatch_delta_left_minus_right": _mean(l_mismatch) - _mean(r_mismatch),
                "mean_vitality_left": _mean(l_vitality),
                "mean_vitality_right": _mean(r_vitality),
                "mean_vitality_delta_left_minus_right": _mean(l_vitality) - _mean(r_vitality),
                "final_mismatch_delta_left_minus_right": float(l_mismatch[-1] - r_mismatch[-1]),
                "final_vitality_delta_left_minus_right": float(l_vitality[-1] - r_vitality[-1]),
                "mismatch_std_left": _std(l_mismatch),
                "mismatch_std_right": _std(r_mismatch),
            }
        )

    rows.sort(key=lambda x: x["mean_mismatch_delta_left_minus_right"])
    top_k = max(1, int(args.top_k))
    best = rows[:top_k]
    worst = rows[-top_k:]
    summary = {
        "storyboard_dir": str(root),
        "left_tag": args.left_tag,
        "right_tag": args.right_tag,
        "num_pairs": len(rows),
        "best_left_minus_right_by_mismatch": best,
        "worst_left_minus_right_by_mismatch": worst,
    }

    if args.output_json:
        p = Path(args.output_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.output_md:
        lines: list[str] = []
        lines.append("# Storyboard Compare Summary")
        lines.append("")
        lines.append(f"- Storyboard dir: `{root}`")
        lines.append(f"- Left tag: `{args.left_tag}`")
        lines.append(f"- Right tag: `{args.right_tag}`")
        lines.append(f"- Pair count: `{len(rows)}`")
        lines.append("")
        lines.append("## Best Left-vs-Right (Lower Mismatch)")
        lines.append("")
        lines.append("| Key | Mean Mismatch Delta | Mean Vitality Delta | Final Mismatch Delta |")
        lines.append("|---|---:|---:|---:|")
        for row in best:
            lines.append(
                "| "
                + f"{row['key']} | {row['mean_mismatch_delta_left_minus_right']:.6f} | {row['mean_vitality_delta_left_minus_right']:.6f} | {row['final_mismatch_delta_left_minus_right']:.6f} |"
            )
        lines.append("")
        lines.append("## Worst Left-vs-Right (Higher Mismatch)")
        lines.append("")
        lines.append("| Key | Mean Mismatch Delta | Mean Vitality Delta | Final Mismatch Delta |")
        lines.append("|---|---:|---:|---:|")
        for row in reversed(worst):
            lines.append(
                "| "
                + f"{row['key']} | {row['mean_mismatch_delta_left_minus_right']:.6f} | {row['mean_vitality_delta_left_minus_right']:.6f} | {row['final_mismatch_delta_left_minus_right']:.6f} |"
            )
        p = Path(args.output_md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        {
            "storyboard_dir": str(root),
            "pairs": len(rows),
            "json": args.output_json,
            "md": args.output_md,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

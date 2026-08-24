#!/usr/bin/env python3
"""Plot seed-run metrics saved by run_seeds_dynamic.sh.

The input format is the `metrics_long.tsv` file produced by
`run_seeds_dynamic.sh`.  Aggregation is done with the Python standard library;
matplotlib is only required when writing plots.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import math
from pathlib import Path
from statistics import fmean


Record = dict[str, str | float | int]


METRIC_ALIASES = {
    "training/sps": "sps",
    "training/actor_loss": "actor_loss",
    "training/critic_loss": "critic_loss",
    "training/main_crl_loss": "main_crl",
    "training/sample_entropy": "entropy",
    "training/log_alpha": "log_alpha",
    "training/mean_inbatch_negatives": "inbatch_negs",
    "training/future_vs_negative_logit_gap": "pos_neg_gap",
    "eval/episode_reward": "reward",
    "eval/episode_cost": "cost",
    "eval/episode_success": "success",
    "eval/episode_success_easy": "success_easy",
    "eval/episode_success_hard": "success_hard",
    "eval/episode_success_any": "success_any",
    "eval/episode_done": "done",
    "eval/episode_dist": "dist",
    "eval/avg_episode_length": "length",
}


def _canonical_metric(metric: str, category: str = "") -> str:
    if category and category != "eval":
        if metric == "reward":
            return "step_reward"
        if metric == "cost":
            return "step_cost"
        if metric == "success":
            return "step_success"
    return METRIC_ALIASES.get(metric, metric)


def _parse_run_arg(value: str) -> tuple[str, Path]:
    """Parses LABEL=PATH or PATH into a display label and run directory."""
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
        run_dir = Path(path.strip())
    else:
        run_dir = Path(value)
        label = run_dir.name
    if not label:
        raise ValueError(f"Empty run label in {value!r}.")
    return label, run_dir


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _to_int(value: str) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _load_runs(run_args: list[str], *, allow_mixed_envs: bool = False) -> list[Record]:
    records: list[Record] = []
    dropped_nonfinite: dict[str, int] = {}
    for run_arg in run_args:
        label, run_dir = _parse_run_arg(run_arg)
        metrics_path = run_dir / "metrics_long.tsv"
        if not metrics_path.exists():
            parent_metrics = run_dir.parent / "metrics_long.tsv"
            if parent_metrics.exists():
                raise FileNotFoundError(
                    f"Missing metrics file: {metrics_path}. It looks like {run_dir} is a single-seed "
                    f"directory; pass the aggregate run directory instead: {run_dir.parent}"
                )
            raise FileNotFoundError(
                f"Missing metrics file: {metrics_path}. Expected an aggregate run directory containing "
                "metrics_long.tsv, e.g. results/<env_id>/<method_name>/<run_name>."
            )
        with metrics_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                value = _to_float(row.get("value", ""))
                if not math.isfinite(value):
                    dropped_nonfinite[label] = dropped_nonfinite.get(label, 0) + 1
                    continue
                records.append(
                    {
                        "method": label,
                        "run_dir": str(run_dir),
                        "run_id": row.get("run_id", ""),
                        "seed": row.get("seed", ""),
                        "gpu": row.get("gpu", ""),
                        "env_id": row.get("env_id", ""),
                        "epoch": _to_int(row.get("epoch", "")),
                        "env_steps": _to_int(row.get("env_steps", "")),
                        "category": row.get("category", ""),
                        "metric": _canonical_metric(row.get("metric", ""), row.get("category", "")),
                        "value": value,
                    }
                )
    for label, count in sorted(dropped_nonfinite.items()):
        print(f"Warning: dropped {count} non-finite metric row(s) from {label!r}.")
    if not records:
        raise ValueError("No metric rows loaded. Did the runs finish at least one printed epoch?")
    env_ids = sorted({str(row["env_id"]) for row in records if str(row["env_id"])})
    if len(env_ids) > 1 and not allow_mixed_envs:
        raise ValueError(
            "Refusing to compare runs from different environments: "
            f"{', '.join(env_ids)}. Pass --allow-mixed-envs only when this is intentional."
        )
    return records


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = fmean(values)
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))


def _aggregate_exact(records: list[Record], metrics: set[str], x_key: str) -> list[Record]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in records:
        metric = _canonical_metric(str(row["metric"]), str(row.get("category", "")))
        if metric not in metrics:
            continue
        key = (str(row["method"]), metric, int(row[x_key]))
        groups.setdefault(key, []).append(float(row["value"]))

    summary: list[Record] = []
    for (method, metric, x_value), values in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "metric": metric,
                x_key: x_value,
                "mean": fmean(values),
                "std": _std(values),
                "count": len(values),
            }
        )
    return summary


def _dedupe_curve(points: list[tuple[int, float]]) -> list[tuple[int, float]]:
    latest_by_x: dict[int, float] = {}
    for x, y in points:
        latest_by_x[x] = y
    return sorted(latest_by_x.items())


def _interp_curve(points: list[tuple[int, float]], x: int) -> float | None:
    if not points:
        return None
    if len(points) == 1:
        return points[0][1] if points[0][0] == x else None
    if x < points[0][0] or x > points[-1][0]:
        return None
    idx = bisect_left(points, (x, -math.inf))
    if idx < len(points) and points[idx][0] == x:
        return points[idx][1]
    if idx == 0 or idx >= len(points):
        return None
    x0, y0 = points[idx - 1]
    x1, y1 = points[idx]
    if x1 == x0:
        return y1
    frac = (x - x0) / (x1 - x0)
    return y0 + frac * (y1 - y0)


def _aggregate_interpolated(records: list[Record], metrics: set[str], x_key: str) -> list[Record]:
    curves: dict[tuple[str, str, str], list[tuple[int, float]]] = {}
    grid_by_metric: dict[str, set[int]] = {}
    for row in records:
        metric = _canonical_metric(str(row["metric"]), str(row.get("category", "")))
        if metric not in metrics:
            continue
        x_value = int(row[x_key])
        curves.setdefault((str(row["method"]), metric, str(row["seed"])), []).append(
            (x_value, float(row["value"]))
        )
        grid_by_metric.setdefault(metric, set()).add(x_value)

    deduped = {key: _dedupe_curve(points) for key, points in curves.items()}
    groups: dict[tuple[str, str, int], list[float]] = {}
    for (method, metric, _seed), points in deduped.items():
        for x_value in sorted(grid_by_metric.get(metric, ())):
            value = _interp_curve(points, x_value)
            if value is None:
                continue
            groups.setdefault((method, metric, x_value), []).append(value)

    summary: list[Record] = []
    for (method, metric, x_value), values in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "metric": metric,
                x_key: x_value,
                "mean": fmean(values),
                "std": _std(values),
                "count": len(values),
            }
        )
    return summary


def _aggregate(records: list[Record], metrics: set[str], x_key: str, align: str) -> list[Record]:
    metrics = {_canonical_metric(metric) for metric in metrics}
    if align == "exact":
        return _aggregate_exact(records, metrics, x_key)
    if align == "interpolate":
        return _aggregate_interpolated(records, metrics, x_key)
    raise ValueError(f"Unknown alignment mode: {align}")


def _write_summary(summary: list[Record], x_key: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "metric", x_key, "mean", "std", "count"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print(output_path)


def _save_metric_plot(
    summary: list[Record],
    metric: str,
    x_key: str,
    output_path: Path,
    title: str,
    show_std: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it in this environment, for example: "
            "python -m pip install matplotlib"
        ) from exc

    metric_rows = [row for row in summary if row["metric"] == metric]
    if not metric_rows:
        print(f"Warning: metric {metric!r} was not found; skipping.")
        return

    by_method: dict[str, list[Record]] = {}
    for row in metric_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for method, rows in by_method.items():
        rows = sorted(rows, key=lambda row: int(row[x_key]))
        x = [int(row[x_key]) for row in rows]
        mean = [float(row["mean"]) for row in rows]
        std = [float(row["std"]) for row in rows]
        ax.plot(x, mean, label=method, linewidth=2)
        if show_std:
            lower = [m - s for m, s in zip(mean, std)]
            upper = [m + s for m, s in zip(mean, std)]
            ax.fill_between(x, lower, upper, alpha=0.18)

    ax.set_xlabel("Environment steps" if x_key == "env_steps" else "Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot mean/std curves from run_seeds_dynamic.sh metrics_long.tsv files.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run directory to plot. Use LABEL=PATH to set the legend label. "
            "Repeat for multiple methods/run IDs."
        ),
    )
    parser.add_argument(
        "--metric",
        action="append",
        required=True,
        help="Metric name to plot, e.g. reward, cost, success, critic_loss. Repeat for multiple metrics.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/seed_metrics",
        help="Directory where PNG plots and summary TSV are saved.",
    )
    parser.add_argument(
        "--x",
        choices=("env_steps", "epoch"),
        default="env_steps",
        help="X axis for curves.",
    )
    parser.add_argument(
        "--align",
        choices=("interpolate", "exact"),
        default="interpolate",
        help=(
            "How to align seed curves before mean/std aggregation. "
            "interpolate handles different print/env-step grids; exact preserves old exact-x grouping."
        ),
    )
    parser.add_argument("--no-std", action="store_true", help="Disable mean +/- std shading.")
    parser.add_argument(
        "--allow-mixed-envs",
        action="store_true",
        help="Allow one plot to combine records with different env_id values.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write the aggregated TSV; do not import matplotlib or write PNG plots.",
    )
    parser.add_argument("--title-prefix", default="", help="Optional prefix added to every plot title.")
    parser.add_argument("--summary-name", default="summary.tsv", help="Filename for the aggregated mean/std TSV.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    requested_metrics = {_canonical_metric(metric) for metric in args.metric}
    records = _load_runs(args.run, allow_mixed_envs=args.allow_mixed_envs)
    available_metrics = {str(row["metric"]) for row in records}
    methods = sorted({str(row["method"]) for row in records})
    available_by_method = {
        method: {str(row["metric"]) for row in records if row["method"] == method}
        for method in methods
    }
    for method in methods:
        for metric in sorted(requested_metrics.difference(available_by_method[method])):
            print(f"Warning: metric {metric!r} was not found for method {method!r}.")

    summary = _aggregate(records, requested_metrics, args.x, args.align)
    if not summary:
        available = ", ".join(sorted(available_metrics))
        raise ValueError(f"None of the requested metrics were found. Available metrics: {available}")

    _write_summary(summary, args.x, output_dir / args.summary_name)

    if args.summary_only:
        return

    for metric in requested_metrics:
        safe_name = metric.replace("/", "_").replace(" ", "_")
        title = f"{args.title_prefix}{metric}" if args.title_prefix else metric
        _save_metric_plot(
            summary,
            metric,
            args.x,
            output_dir / f"{safe_name}.png",
            title,
            show_std=not args.no_std,
        )


if __name__ == "__main__":
    main()

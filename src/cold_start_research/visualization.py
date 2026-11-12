from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List


COLUMNS = [
    "epoch",
    "train_loss",
    "precision5",
    "ndcg5",
    "map5",
    "precision7",
    "ndcg7",
    "map7",
    "precision10",
    "ndcg10",
    "map10",
]


def read_tanp_log(log_path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"TOP-\d+:?", "", line)
        parts = line.split()
        if len(parts) != len(COLUMNS):
            raise ValueError(
                f"Неожиданный формат строки в {log_path}: {raw_line!r}"
            )
        values = [float(value) for value in parts]
        rows.append(dict(zip(COLUMNS, values)))

    if not rows:
        raise ValueError(f"В логе нет строк с метриками: {log_path}")
    return rows


def save_metrics_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_metrics_figure(rows: List[Dict[str, float]], output_stem: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in rows],
        color="#172554",
        linewidth=1.8,
    )
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    for key, label, color in (
        ("ndcg5", "NDCG@5", "#2563EB"),
        ("ndcg7", "NDCG@7", "#0D9488"),
        ("ndcg10", "NDCG@10", "#F97316"),
    ):
        axes[1].plot(
            epochs,
            [row[key] for row in rows],
            label=label,
            color=color,
            linewidth=1.8,
        )
    axes[1].set_title("Ranking quality")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("NDCG")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_outputs(
    log_path: Path,
    run_name: str,
    results_dir: Path,
    figures_dir: Path,
) -> None:
    rows = read_tanp_log(log_path)
    csv_path = results_dir / f"{run_name}_metrics.csv"
    figure_stem = figures_dir / f"{run_name}_metrics"
    save_metrics_csv(rows, csv_path)
    save_metrics_figure(rows, figure_stem)
    print(f"Метрики: {csv_path}")
    print(f"Рисунки: {figure_stem.with_suffix('.png')}, {figure_stem.with_suffix('.pdf')}")


def save_suite_summary(
    runs: List[Dict[str, object]],
    output_path: Path,
    figures_dir: Path | None = None,
) -> None:
    fieldnames = ["run_name", "dataset", "decoder", "seed"] + COLUMNS
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, object]] = []
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            rows = read_tanp_log(Path(run["log_path"]))
            final_row = rows[-1]
            summary_row = {
                "run_name": run["run_name"],
                "dataset": run["dataset"],
                "decoder": run["decoder"],
                "seed": run["seed"],
                **final_row,
            }
            writer.writerow(summary_row)
            summary_rows.append(summary_row)
    print(f"Сводная таблица: {output_path}")
    if figures_dir is not None:
        figure_stem = figures_dir / output_path.stem
        save_suite_summary_figure(summary_rows, figure_stem)
        print(
            "Сводные рисунки: "
            f"{figure_stem.with_suffix('.png')}, "
            f"{figure_stem.with_suffix('.pdf')}"
        )


def read_suite_summary(summary_path: Path) -> List[Dict[str, object]]:
    with summary_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows: List[Dict[str, object]] = []
        for row in reader:
            rows.append(
                {
                    **row,
                    **{
                        column: float(row[column])
                        for column in COLUMNS
                    },
                }
            )
    if not rows:
        raise ValueError(f"В сводной таблице нет строк: {summary_path}")
    return rows


def save_suite_summary_figure(
    rows: List[Dict[str, object]],
    output_stem: Path,
) -> None:
    import matplotlib.pyplot as plt

    dataset_labels = {
        "lastfm_20": "LastFM",
        "movielens_1m": "MovieLens-1M",
        "gowalla_reconstructed": "Gowalla (reconstructed)",
    }
    colors = ["#2563EB", "#0D9488", "#F97316"]
    cutoffs = (5, 7, 10)
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharex=True)

    for axis, (prefix, title) in zip(
        axes,
        (("precision", "Precision"), ("ndcg", "NDCG"), ("map", "MAP")),
    ):
        for row, color in zip(rows, colors):
            axis.plot(
                cutoffs,
                [float(row[f"{prefix}{cutoff}"]) for cutoff in cutoffs],
                label=dataset_labels.get(str(row["dataset"]), str(row["dataset"])),
                color=color,
                linewidth=2.2,
            )
        axis.set_title(title)
        axis.set_xlabel("K")
        axis.set_xticks(cutoffs)
        axis.set_ylim(0.4, 0.92)
        axis.grid(axis="y", alpha=0.22)

    axes[0].set_ylabel("Metric value")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("TaNP Gating-FiLM baseline, seed 2020", y=1.12)
    figure.tight_layout()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

"""
Generate the two report-specific figures used in group55_report.tex.

These do NOT retrain any model. They read the training dataset for the class
balance plot and use the already-recorded evaluation metrics (taken directly
from the Stage 1 / Stage 2 result plots produced by the pipelines) for the
combined summary plot.

Outputs into ../results/:
    report_dataset_distribution.png
    report_stage_summary.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

LABEL_MAP = {"opinion": "Opinion", "fact": "Fact",
             "opinions": "Opinion", "facts": "Fact"}


def dataset_distribution():
    df = pd.read_csv(ROOT / "data" / "dataset.csv", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]
    labels = df["classification"].apply(
        lambda r: LABEL_MAP.get(str(r).strip().lower(), str(r).strip()))
    counts = labels.value_counts().reindex(["Fact", "Opinion"])

    fig, ax = plt.subplots(figsize=(5, 3.4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#4c72b0", "#dd8452"])
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3,
                f"{v}\n({v / total * 100:.1f}%)", ha="center", fontsize=10)
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Training set class balance (n = {total})")
    ax.set_ylim(0, counts.max() * 1.18)
    plt.tight_layout()
    fig.savefig(RESULTS / "report_dataset_distribution.png", dpi=150)
    plt.close()
    print("saved report_dataset_distribution.png")


def stage_summary():
    # F1-macro values as recorded by the two pipelines.
    # Stage 1: 5-fold CV on training data (stage1_model_comparison.png).
    # Stage 2: internal 20% held-out validation split (stage2_comparison.png).
    models = [
        ("Majority\nbaseline\n(S1)", 0.341, "#dd8452"),
        ("Random Forest\n(S1 main)", 0.853, "#4c72b0"),
        ("Zero-shot MNLI\nbaseline (S2)", 0.548, "#dd8452"),
        ("Fine-tuned BART\n(S2 main)", 0.935, "#4c72b0"),
    ]
    names = [m[0] for m in models]
    vals = [m[1] for m in models]
    colors = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 1.0)
    ax.set_title("Stage 1 vs Stage 2 — main methods and baselines")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4c72b0"),
               plt.Rectangle((0, 0), 1, 1, color="#dd8452")]
    ax.legend(handles, ["Main method", "Baseline"], loc="upper left")
    plt.tight_layout()
    fig.savefig(RESULTS / "report_stage_summary.png", dpi=150)
    plt.close()
    print("saved report_stage_summary.png")


if __name__ == "__main__":
    dataset_distribution()
    stage_summary()

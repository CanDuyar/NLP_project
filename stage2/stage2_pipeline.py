"""
Stage 2 – LLM-Based Classification Pipeline
=============================================
Classifies texts as Fact or Opinion using google/flan-t5-large with:

  Baseline : Zero-shot prompting (no examples, minimal prompt).
  Main     : Few-Shot + Chain-of-Thought prompting backed by a curated
             knowledge base.

Both approaches are evaluated when gold labels are available so the
report can clearly show the improvement the main method achieves over
the baseline — as required by the assignment.

Usage:
    python stage2_pipeline.py --test data/validation.csv
    python stage2_pipeline.py --test data/dataset.csv
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, save_classifications, classification_report_dict
from stage2.llm_classifier import classify_single, classify_zero_shot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, title: str, filename: str):
    """Save a confusion matrix PNG to the results directory."""
    labels = ["Fact", "Opinion"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix: %s", filename)


def _plot_comparison(baseline_metrics: dict, main_metrics: dict):
    """Bar chart comparing baseline vs main method on Accuracy and F1-macro."""
    methods    = ["Zero-Shot\n(Baseline)", "Few-Shot + CoT\n(Main)"]
    accuracies = [baseline_metrics["accuracy"], main_metrics["accuracy"]]
    f1_scores  = [baseline_metrics["macro_f1"],  main_metrics["macro_f1"]]

    x     = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, f1_scores,  width, label="F1-macro",  color="#dd8452")

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", fontsize=9,
        )

    ax.set_ylabel("Score")
    ax.set_title("Stage 2 – Baseline vs Few-Shot + CoT")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stage2_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved Stage 2 comparison plot.")


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_stage2(test_path: str = None):
    """
    Full Stage 2 pipeline.

    Parameters
    ----------
    test_path : CSV with texts to classify (id + text columns; label optional for eval)
    """
    output_csv = RESULTS_DIR / "group55_classifications_2.csv"

    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_dataset(test_path)
    texts      = df["text"].tolist()
    ids        = df["id"].tolist() if "id" in df.columns else list(range(len(df)))
    gold       = df["label"].tolist() if "label" in df.columns else None
    has_labels = gold is not None

    logger.info("Stage 2: classifying %d texts ...", len(texts))

    # ── 2. Baseline: Zero-Shot classification ─────────────────────────────
    logger.info("--- Running BASELINE (Zero-Shot) ---")
    baseline_preds = []
    for i, text in enumerate(texts):
        logger.info("[%d/%d] Zero-shot: %.60s ...", i + 1, len(texts), text)
        label = classify_zero_shot(text)
        baseline_preds.append(label)

    # ── 3. Main method: Few-Shot + Chain-of-Thought ───────────────────────
    logger.info("--- Running MAIN METHOD (Few-Shot + CoT) ---")
    main_preds = []
    for i, text in enumerate(texts):
        logger.info("[%d/%d] Few-Shot+CoT: %.60s ...", i + 1, len(texts), text)
        label, raw = classify_single(text)
        main_preds.append(label)

    # ── 4. Evaluation (if gold labels available) ──────────────────────────
    if has_labels:
        baseline_eval = classification_report_dict(gold, baseline_preds)
        main_eval     = classification_report_dict(gold, main_preds)

        logger.info("=== Stage 2 BASELINE (Zero-Shot) Evaluation ===")
        logger.info("Accuracy : %.4f", baseline_eval["accuracy"])
        logger.info("F1-macro : %.4f", baseline_eval["macro_f1"])
        logger.info("\n%s", baseline_eval["classification_report_str"])

        logger.info("=== Stage 2 MAIN METHOD (Few-Shot + CoT) Evaluation ===")
        logger.info("Accuracy : %.4f", main_eval["accuracy"])
        logger.info("F1-macro : %.4f", main_eval["macro_f1"])
        logger.info("\n%s", main_eval["classification_report_str"])

        # Confusion matrices
        _plot_confusion_matrix(
            gold, baseline_preds,
            title="Stage 2 Zero-Shot Baseline — Confusion Matrix",
            filename="stage2_baseline_cm.png",
        )
        _plot_confusion_matrix(
            gold, main_preds,
            title="Stage 2 Few-Shot + CoT — Confusion Matrix",
            filename="stage2_fewshot_cot_cm.png",
        )

        # Side-by-side comparison bar chart
        _plot_comparison(baseline_eval, main_eval)

    # ── 5. Save output CSV (main method predictions) ──────────────────────
    save_classifications(ids, main_preds, output_csv)
    logger.info("Stage 2 complete. Classifications saved to %s", output_csv)

    return main_preds


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 – LLM Classification Pipeline")
    parser.add_argument("--test", type=str, help="Path to test CSV")
    args = parser.parse_args()

    run_stage2(test_path=args.test)


if __name__ == "__main__":
    main()
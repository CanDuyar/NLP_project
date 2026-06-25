"""
Stage 2 - LLM-Based Classification Pipeline
=============================================
Baseline : Zero-shot classification with facebook/bart-large-mnli.
Main     : facebook/bart-base fine-tuned on the labelled training data, with
           knowledge-base linguistic markers injected into each input.

The official test set released on 18.06 has no gold labels, so it cannot be
used to measure accuracy - that is fine, since it is meant for generating the
final submission, not for evaluation. To still get a baseline-vs-main
comparison for the report, this pipeline always holds out a stratified slice
of the labelled training data (data/dataset.csv by default) the same way
Stage 1 does, evaluates both methods on it, and only afterwards runs the
fine-tuned model on whatever test/validation file was actually requested.

Usage:
    python -m stage2.stage2_pipeline --train data/dataset.csv --test data/validationset.csv
    python -m stage2.stage2_pipeline --train data/dataset.csv
"""

import sys
import logging
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, save_classifications, classification_report_dict
from stage2.llm_classifier import classify_zero_shot
from stage2.finetune import train_bart_classifier, predict, save_model

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path(__file__).parent.parent / "models" / "bart_finetuned"


# ── Evaluation / plotting helpers ──────────────────────────────────────────

def _plot_confusion_matrix(y_true, y_pred, title: str, filename: str):
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
    methods = ["Zero-Shot MNLI\n(Baseline)", "Fine-tuned BART\n(Main)"]
    accuracies = [baseline_metrics["accuracy"], main_metrics["accuracy"]]
    f1_scores = [baseline_metrics["macro_f1"], main_metrics["macro_f1"]]

    x = np.arange(len(methods))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4c72b0")
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="F1-macro", color="#dd8452")

    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", fontsize=9,
        )

    ax.set_ylabel("Score")
    ax.set_title("Stage 2 - Baseline vs Fine-tuned BART")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stage2_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved Stage 2 comparison plot.")


def _plot_learning_curve(history: dict):
    if not history.get("val_accuracy"):
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, history["train_loss"], color="#4c72b0", marker="o", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss", color="#4c72b0")

    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_accuracy"], color="#dd8452", marker="s", label="Val accuracy")
    ax2.plot(epochs, history["val_f1_macro"], color="#55a868", marker="^", label="Val F1-macro")
    ax2.set_ylabel("Validation score")

    ax1.set_title("Stage 2 - BART fine-tuning progress")
    fig.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stage2_learning_curve.png", dpi=150)
    plt.close()
    logger.info("Saved fine-tuning learning curve.")


# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_stage2(train_path: str = None, test_path: str = None, epochs: int = 4):
    """
    Full Stage 2 pipeline.

    Parameters
    ----------
    train_path : labelled CSV used to fine-tune BART and build the internal
                 evaluation split (defaults to data/dataset.csv)
    test_path  : the test/validation CSV to generate final submission
                 predictions for. May or may not carry gold labels - the
                 internal evaluation above does not depend on it either way.
    epochs     : fine-tuning epochs for the main method
    """
    output_csv = RESULTS_DIR / "group55_classifications_2.csv"
    train_path = train_path or "data/dataset.csv"

    # ── 1. Load labelled data, build internal train/eval split ────────────
    full_df = load_dataset(train_path)
    if "label" not in full_df.columns:
        raise ValueError(
            "Stage 2 needs a labelled CSV (e.g. data/dataset.csv) to fine-tune "
            "on and to evaluate baseline vs main method."
        )

    train_df, eval_df = train_test_split(
        full_df, test_size=0.20, stratify=full_df["label"], random_state=11
    )
    logger.info("Stage 2 internal split - train: %d, eval: %d", len(train_df), len(eval_df))

    # ── 2. Baseline: zero-shot BART-MNLI on the internal eval split ───────
    logger.info("--- Running BASELINE (Zero-Shot BART-MNLI) ---")
    eval_texts = eval_df["text"].tolist()
    baseline_preds = []
    for i, text in enumerate(eval_texts):
        if (i + 1) % 20 == 0 or i == len(eval_texts) - 1:
            logger.info("  Zero-shot baseline: %d/%d", i + 1, len(eval_texts))
        baseline_preds.append(classify_zero_shot(text))

    # ── 3. Main method: fine-tune BART-base on the internal train split ───
    logger.info("--- Fine-tuning BART-base (MAIN METHOD) ---")
    model, tokenizer, history = train_bart_classifier(
        train_df["text"].tolist(), train_df["label"].tolist(),
        val_texts=eval_texts, val_labels=eval_df["label"].tolist(),
        epochs=epochs,
    )
    save_model(model, tokenizer, MODEL_DIR)
    _plot_learning_curve(history)

    main_preds = predict(model, tokenizer, eval_texts)

    # ── 4. Evaluation on the internal split (always runs) ──────────────────
    gold = eval_df["label"].tolist()
    baseline_eval = classification_report_dict(gold, baseline_preds)
    main_eval = classification_report_dict(gold, main_preds)

    logger.info("=== Stage 2 BASELINE (Zero-Shot BART-MNLI) Evaluation ===")
    logger.info("Accuracy : %.4f", baseline_eval["accuracy"])
    logger.info("F1-macro : %.4f", baseline_eval["macro_f1"])
    logger.info("\n%s", baseline_eval["classification_report_str"])

    logger.info("=== Stage 2 MAIN METHOD (Fine-tuned BART) Evaluation ===")
    logger.info("Accuracy : %.4f", main_eval["accuracy"])
    logger.info("F1-macro : %.4f", main_eval["macro_f1"])
    logger.info("\n%s", main_eval["classification_report_str"])

    _plot_confusion_matrix(
        gold, baseline_preds,
        title="Stage 2 Zero-Shot Baseline - Confusion Matrix",
        filename="stage2_baseline_cm.png",
    )
    _plot_confusion_matrix(
        gold, main_preds,
        title="Stage 2 Fine-tuned BART - Confusion Matrix",
        filename="stage2_finetuned_cm.png",
    )
    _plot_comparison(baseline_eval, main_eval)

    # ── 5. Final predictions for submission ─────────────────────────────────
    if test_path:
        final_df = load_dataset(test_path)
    else:
        final_df = eval_df  # nothing else supplied, fall back to the held-out split

    final_texts = final_df["text"].tolist()
    final_ids = final_df["id"].tolist() if "id" in final_df.columns else list(range(len(final_df)))
    final_preds = predict(model, tokenizer, final_texts)

    if "label" in final_df.columns:
        ext_eval = classification_report_dict(final_df["label"].tolist(), final_preds)
        logger.info("=== Stage 2 - Fine-tuned BART on the supplied test set ===")
        logger.info("Accuracy : %.4f", ext_eval["accuracy"])
        logger.info("F1-macro : %.4f", ext_eval["macro_f1"])

    save_classifications(final_ids, final_preds, output_csv)
    logger.info("Stage 2 complete. Classifications saved to %s", output_csv)

    return final_preds


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2 - LLM Classification Pipeline")
    parser.add_argument("--train", type=str, help="Labelled CSV for fine-tuning + internal evaluation")
    parser.add_argument("--test", type=str, help="Test/validation CSV for final submission predictions")
    parser.add_argument("--epochs", type=int, default=4, help="Fine-tuning epochs")
    args = parser.parse_args()

    run_stage2(train_path=args.train, test_path=args.test, epochs=args.epochs)


if __name__ == "__main__":
    main()

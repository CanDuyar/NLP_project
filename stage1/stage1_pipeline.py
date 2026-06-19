"""
Stage 1 – Main Pipeline
=========================
Fact vs Opinion classification using traditional NLP features.

Usage:
    python stage1_pipeline.py --train data/dataset.csv
    python stage1_pipeline.py --train data/dataset.csv --test data/validation.csv
"""

import sys
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_dataset, save_classifications
from stage1.feature_extractor import extract_features_batch
from stage1.classifier import (
    compare_classifiers, train_final_model, evaluate,
    plot_feature_importance, plot_cv_comparison
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_stage1(
    train_path: str = None,
    test_path:  str = None,
):
    """
    Full Stage 1 pipeline: feature extraction + ML classification.

    Parameters
    ----------
    train_path : path to labelled training CSV (must have text + label columns)
    test_path  : optional separate test CSV; if None, 20% of train is held out
    """
    output_csv = RESULTS_DIR / "group55_classifications_1.csv"

    # ── 1. Load data ──────────────────────────────────────────────────────
    full_df = load_dataset(train_path)
    if "text" not in full_df.columns:
        raise ValueError("Training CSV must contain a 'text' column.")
    if "label" not in full_df.columns:
        raise ValueError("Training CSV must contain a 'label' column.")

    if test_path:
        train_df = full_df
        test_df  = load_dataset(test_path)
    else:
        train_df, test_df = train_test_split(
            full_df, test_size=0.20,
            stratify=full_df["label"], random_state=42
        )

    logger.info("Train size: %d | Test size: %d", len(train_df), len(test_df))

    # ── 2. Feature extraction ─────────────────────────────────────────────
    logger.info("Extracting features from training set ...")
    X_train_df = extract_features_batch(train_df["text"].tolist())
    logger.info("Extracting features from test set ...")
    X_test_df  = extract_features_batch(test_df["text"].tolist())

    feature_names = X_train_df.columns.tolist()
    X_train = X_train_df.values
    X_test  = X_test_df.values
    y_train = train_df["label"].values
    y_test  = test_df["label"].values if "label" in test_df.columns else None

    logger.info("Feature matrix shape — train: %s  test: %s",
                X_train.shape, X_test.shape)

    # ── 3. Cross-validation model comparison ─────────────────────────────
    logger.info("Running 5-fold cross-validation on training set ...")
    cv_df = compare_classifiers(X_train, y_train, cv=5)
    logger.info("\nCV Results:\n%s", cv_df.to_string(index=False))
    plot_cv_comparison(cv_df)

    # ── 4. Select and train best model ────────────────────────────────────
    # Exclude the dummy baseline from auto-selection
    non_baseline = cv_df[~cv_df["model"].str.contains("Baseline")]
    best_model_name = non_baseline.iloc[0]["model"] if len(non_baseline) else cv_df.iloc[0]["model"]
    logger.info("Best model selected: %s", best_model_name)
    clf = train_final_model(X_train, y_train, model_name=best_model_name)

    # ── 5. Evaluate on held-out test set ─────────────────────────────────
    if y_test is not None:
        results = evaluate(clf, X_test, y_test,
                           labels=("Fact", "Opinion"),
                           model_name=best_model_name)

        # Evaluate majority-class baseline for comparison
        from stage1.classifier import get_classifiers
        baseline = get_classifiers()["Baseline_Majority"]
        baseline.fit(X_train, y_train)
        baseline_results = evaluate(baseline, X_test, y_test,
                                    labels=("Fact", "Opinion"),
                                    model_name="Baseline_Majority")

        summary = {
            "best_model":        best_model_name,
            "test_accuracy":     results["accuracy"],
            "test_f1_macro":     results["f1_macro"],
            "baseline_accuracy": baseline_results["accuracy"],
            "baseline_f1_macro": baseline_results["f1_macro"],
        }
        logger.info("Evaluation summary: %s", summary)

    # ── 6. Feature importance ─────────────────────────────────────────────
    plot_feature_importance(clf, feature_names, model_name=best_model_name)

    # ── 7. Predict and save output CSV ────────────────────────────────────
    y_pred   = clf.predict(X_test)
    test_ids = test_df["id"].tolist() if "id" in test_df.columns else list(range(len(test_df)))
    save_classifications(test_ids, y_pred, output_csv)
    logger.info("Stage 1 complete. Classifications saved to: %s", output_csv)

    # ── 8. Feature distribution plots ────────────────────────────────────
    _plot_feature_distributions(X_train_df, y_train)

    return clf, cv_df


def _plot_feature_distributions(X_df: pd.DataFrame, y: np.ndarray):
    """Plot distributions of key features split by class label."""
    key_feats = [
        "hedge_ratio", "fp_ratio", "vader_abs", "adj_ratio",
        "num_ratio", "opinion_lexicon", "ner_density", "ttr",
    ]
    key_feats = [f for f in key_feats if f in X_df.columns]
    if not key_feats:
        return

    n = len(key_feats)
    fig, axes = plt.subplots(2, max(n // 2, 1), figsize=(14, 7))
    axes = axes.flatten()

    for i, feat in enumerate(key_feats):
        ax = axes[i]
        for lbl, color in [("Fact", "#4c72b0"), ("Opinion", "#dd8452")]:
            vals = X_df[feat][y == lbl].dropna()
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=lbl, density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
        ax.set_xlabel("Value")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Stage 1 – Feature Distributions by Class", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stage1_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Feature distribution plots saved.")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1 – Traditional NLP Pipeline")
    parser.add_argument("--train", type=str, help="Path to labelled training CSV")
    parser.add_argument("--test",  type=str, help="Path to test CSV (optional)")
    args = parser.parse_args()

    run_stage1(
        train_path=args.train,
        test_path=args.test,
    )


if __name__ == "__main__":
    main()

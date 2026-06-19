"""
Stage 1 – Machine Learning Classifier
========================================
Trains and evaluates multiple classifiers on the extracted linguistic features.
Includes a majority-class baseline and selects the best model via cross-validation.

No transformers are used — only traditional ML (sklearn).
"""

import logging
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate, GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from typing import List

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Preprocessing pipeline builder ──────────────────────────────────────────

def make_preprocessor():
    """Impute NaN + StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# ── Classifier definitions ───────────────────────────────────────────────────

def get_classifiers():
    """Return a dict {name: sklearn_estimator} of candidate classifiers."""
    pre = make_preprocessor()

    return {
        # Baseline (always predicts the majority class)
        "Baseline_Majority": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", DummyClassifier(strategy="most_frequent", random_state=42)),
        ]),

        # Logistic Regression
        "LogisticRegression": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", LogisticRegression(
                max_iter=1000, C=1.0,
                class_weight="balanced", random_state=42
            )),
        ]),

        # Random Forest
        "RandomForest": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=None,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),

        # Gradient Boosting
        "GradientBoosting": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, random_state=42
            )),
        ]),

        # SVM (RBF kernel)
        "SVM_RBF": Pipeline([
            ("pre", make_preprocessor()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True, random_state=42
            )),
        ]),
    }


# ── Cross-validation comparison ───────────────────────────────────────────────

def compare_classifiers(X: np.ndarray, y: np.ndarray,
                         cv: int = 5) -> pd.DataFrame:
    """Run stratified k-fold CV for every classifier, return summary DataFrame."""
    clfs = get_classifiers()
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    records = []

    for name, clf in clfs.items():
        logger.info("  CV evaluating: %s", name)
        scores = cross_validate(
            clf, X, y, cv=cv_obj,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            n_jobs=1
        )
        records.append({
            "model":       name,
            "acc_mean":    scores["test_accuracy"].mean(),
            "acc_std":     scores["test_accuracy"].std(),
            "f1_mean":     scores["test_f1_macro"].mean(),
            "f1_std":      scores["test_f1_macro"].std(),
        })
        logger.info("    Accuracy %.3f ± %.3f | F1-macro %.3f ± %.3f",
                    records[-1]["acc_mean"], records[-1]["acc_std"],
                    records[-1]["f1_mean"],  records[-1]["f1_std"])

    df = pd.DataFrame(records).sort_values("f1_mean", ascending=False)
    return df


# ── Training the final model ──────────────────────────────────────────────────

def train_final_model(X_train: np.ndarray, y_train: np.ndarray,
                       model_name: str = "RandomForest"):
    """Fit the chosen model on the full training split."""
    clf = get_classifiers()[model_name]
    clf.fit(X_train, y_train)
    logger.info("Trained %s on %d samples.", model_name, len(y_train))
    return clf


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray,
              labels=("Fact", "Opinion"), model_name: str = "model") -> dict:
    """Evaluate a fitted classifier and save plots."""
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, labels=list(labels), zero_division=0)

    logger.info("=== %s Evaluation ===", model_name)
    logger.info("Accuracy : %.4f", acc)
    logger.info("F1-macro : %.4f", f1)
    logger.info("\n%s", report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=list(labels))
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{model_name} — Confusion Matrix\nAcc={acc:.3f}, F1={f1:.3f}")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"stage1_{model_name}_cm.png", dpi=150)
    plt.close()

    return {
        "model":        model_name,
        "accuracy":     acc,
        "f1_macro":     f1,
        "report":       report,
        "confusion_matrix": cm.tolist(),
    }


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(clf, feature_names: List[str],
                             model_name: str = "RandomForest", top_n: int = 20):
    """Plot and save feature importances for tree-based models."""
    try:
        inner = clf.named_steps["clf"]
        if hasattr(inner, "feature_importances_"):
            importances = inner.feature_importances_
        elif hasattr(inner, "coef_"):
            importances = np.abs(inner.coef_[0])
        else:
            return

        idx  = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in idx]
        top_imp      = importances[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_features[::-1], top_imp[::-1], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
        plt.tight_layout()
        fig.savefig(RESULTS_DIR / f"stage1_{model_name}_feature_importance.png", dpi=150)
        plt.close()
        logger.info("Feature importance plot saved.")
    except Exception as e:
        logger.warning("Could not plot feature importance: %s", e)


# ── CV comparison plot ────────────────────────────────────────────────────────

def plot_cv_comparison(cv_df: pd.DataFrame):
    """Bar-plot of CV F1-macro for all models."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4c72b0" if "Baseline" not in m else "#dd8452"
              for m in cv_df["model"]]
    bars = ax.bar(cv_df["model"], cv_df["f1_mean"], color=colors,
                  yerr=cv_df["f1_std"], capsize=5)
    ax.set_ylabel("F1-macro (CV mean)")
    ax.set_title("Stage 1 – Model Comparison (5-fold CV)")
    ax.set_ylim(0, 1.0)
    ax.set_xticklabels(cv_df["model"], rotation=25, ha="right")
    for bar, val in zip(bars, cv_df["f1_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stage1_model_comparison.png", dpi=150)
    plt.close()
    logger.info("CV comparison plot saved.")

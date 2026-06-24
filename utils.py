"""
Shared utilities for NLP Fact vs Opinion Classification Project
"""

import os
import re
import csv
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
KB_DIR       = PROJECT_ROOT / "knowledge_base"

for d in [DATA_DIR, RESULTS_DIR, KB_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Label normalisation ────────────────────────────────────────────────────
LABEL_MAP = {
    "opinion": "Opinion",
    "fact":    "Fact",
    "opinions": "Opinion",
    "facts":    "Fact",
}


def normalise_label(raw: str) -> str:
    return LABEL_MAP.get(str(raw).strip().lower(), str(raw).strip())


# ── CSV I/O ────────────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load the project dataset and normalise columns / labels."""
    # utf-8-sig transparently strips a UTF-8 BOM (present in data/dataset.csv)
    # so the first column header is not read as "﻿Content".
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Try to find text and label columns regardless of exact naming
    text_candidates  = ["text", "article", "sentence", "content", "body"]
    label_candidates = ["label","class","category","type","gold","classification"]
    id_candidates    = ["id", "idx", "index", "article_id", "text_id"]

    text_col  = next((c for c in text_candidates  if c in df.columns), df.columns[0])
    label_col = next((c for c in label_candidates if c in df.columns), None)
    id_col    = next((c for c in id_candidates    if c in df.columns), None)

    out = pd.DataFrame()
    out["text"] = df[text_col].astype(str)

    if label_col:
        out["label"] = df[label_col].apply(normalise_label)
    if id_col:
        out["id"] = df[id_col]
    else:
        out["id"] = range(len(out))

    logger.info("Loaded %d samples from %s", len(out), path)
    if label_col:
        logger.info("Label distribution:\n%s", out["label"].value_counts().to_string())
    return out


def save_classifications(ids, labels, path: str, extra_cols: dict = None):
    """
    Save classification results in the required CSV format:
        id, label  (+ any extra columns)
    """
    df = pd.DataFrame({"id": ids, "label": labels})
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    df.to_csv(path, index=False)
    logger.info("Saved %d classifications to %s", len(df), path)
    return df


# ── Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text: str, *, lower: bool = False) -> str:
    """Light-touch clean: strip HTML entities, extra whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"&\w+;", " ", text)             # HTML entities
    text = re.sub(r"\s+", " ", text).strip()
    if lower:
        text = text.lower()
    return text


# ── Evaluation helpers ─────────────────────────────────────────────────────

def classification_report_dict(y_true, y_pred, labels=None):
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report
    )
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    report = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "macro_f1":  f1_score(y_true, y_pred, average="macro",     zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": {}
    }
    for lbl in labels:
        mask_t = [1 if y == lbl else 0 for y in y_true]
        mask_p = [1 if y == lbl else 0 for y in y_pred]
        report["per_class"][lbl] = {
            "precision": precision_score(mask_t, mask_p, zero_division=0),
            "recall":    recall_score(mask_t, mask_p, zero_division=0),
            "f1":        f1_score(mask_t, mask_p, zero_division=0),
        }
    report["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    report["classification_report_str"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return report

"""
Stage 2 - Zero-Shot Baseline (facebook/bart-large-mnli)
===========================================================
This module implements the Stage 2 baseline only: zero-shot classification
through BART's MNLI checkpoint, used via HuggingFace's zero-shot-classification
pipeline. The text is treated as the premise and "This statement is a fact" /
"This statement is an opinion" as candidate hypotheses; whichever hypothesis
the model assigns the highest entailment probability to becomes the
prediction. No examples, no fine-tuning, no knowledge base - exactly the kind
of approach the assignment says is "not enough" on its own for Stage 2.

The main method (fine-tuned BART with knowledge-base context) lives in
finetune.py, since it needs a training step rather than just inference.
"""

import logging

from transformers import pipeline

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/bart-large-mnli"
CANDIDATE_LABELS = ["a fact", "an opinion"]
HYPOTHESIS_TEMPLATE = "This statement is {}."
LABEL_LOOKUP = {"a fact": "Fact", "an opinion": "Opinion"}

_zero_shot_pipe = None


def _get_pipeline():
    global _zero_shot_pipe
    if _zero_shot_pipe is None:
        logger.info("Loading %s for the Stage 2 zero-shot baseline ...", MODEL_ID)
        _zero_shot_pipe = pipeline("zero-shot-classification", model=MODEL_ID)
        logger.info("Zero-shot model loaded.")
    return _zero_shot_pipe


def classify_zero_shot(text: str) -> str:
    """Stage 2 baseline: zero-shot Fact/Opinion classification via BART-MNLI."""
    pipe = _get_pipeline()
    result = pipe(text, candidate_labels=CANDIDATE_LABELS,
                  hypothesis_template=HYPOTHESIS_TEMPLATE)
    top_label = result["labels"][0]
    return LABEL_LOOKUP[top_label]

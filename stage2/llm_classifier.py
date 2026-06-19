"""
Stage 2 – LLM Classifier (google/flan-t5-large via HuggingFace Transformers)
==============================================================================
Implements three classification modes used in the pipeline:

  1. classify_zero_shot   – Zero-shot baseline: single prompt, no examples.
  2. classify_single      – Few-Shot + Chain-of-Thought (primary method):
                            retrieves semantically similar examples from the
                            knowledge base, appends them as context, and asks
                            the model to reason step-by-step before labelling.
  3. classify_with_voting – Self-Consistency wrapper: calls classify_single
                            n_votes times at varying temperatures and takes a
                            majority vote for robustness.

All three return a canonical label string: "Fact" or "Opinion".

NOTE: Flan-T5-large requires ~3 GB RAM and ~800 MB disk space for the model
weights.  On first run, HuggingFace will download the weights automatically.
Subsequent runs use the local cache (~/.cache/huggingface).
Expected runtime on CPU: ~1–3 seconds per sample.
"""

import re
import time
import logging
from typing import List, Tuple
from transformers import pipeline

logger = logging.getLogger(__name__)

# Knowledge base utilities
from knowledge_base.knowledge_base import (
    retrieve_examples,
    format_examples_for_prompt,
    get_marker_summary,
)

# ── Model setup ───────────────────────────────────────────────────────────────
MODEL_ID = "google/flan-t5-large"
logger.info("Loading %s ... (first run will download ~800 MB)", MODEL_ID)

classifier_pipe = pipeline(
    "text2text-generation",
    model=MODEL_ID
)
logger.info("Model loaded successfully.")


def _run_model(prompt: str, temperature: float = 0.0) -> str:
    """Run the loaded model on a prompt and return the generated text."""
    outputs = classifier_pipe(
        prompt,
        max_new_tokens=150,
        temperature=max(0.1, temperature) if temperature > 0 else None,
        do_sample=True if temperature > 0 else False,
    )
    return outputs[0]["generated_text"]


def _extract_label(response_text: str) -> str:
    """
    Parse the model output and return a canonical label.

    Priority:
      1. Explicit LABEL: <label> pattern (used in CoT prompt).
      2. Bare 'Fact' or 'Opinion' anywhere in the response.
      3. Keyword frequency fallback.
    """
    match = re.search(r"LABEL\s*:\s*(Fact|Opinion)", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    match = re.search(r"\b(Fact|Opinion)\b", response_text, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    lower = response_text.lower()
    return "Fact" if lower.count("fact") >= lower.count("opinion") else "Opinion"


# ── Zero-shot baseline ────────────────────────────────────────────────────────

def classify_zero_shot(text: str) -> str:
    """
    Zero-shot classification: no examples, minimal prompt.
    Used as the Stage 2 baseline for comparison purposes.
    """
    prompt = (
        f"Classify the following text as exactly one of: Fact or Opinion.\n"
        f"Text: {text}\n"
        f"Output only the word Fact or Opinion."
    )
    raw = _run_model(prompt, temperature=0.0)
    return _extract_label(raw)


# ── Few-Shot + Chain-of-Thought (primary method) ──────────────────────────────

def classify_single(text: str) -> Tuple[str, str]:
    """
    Classify one text using Few-Shot + Chain-of-Thought prompting.

    Retrieves the most relevant labelled examples from the knowledge base
    and prepends them to the prompt together with linguistic marker guidance,
    then asks the model to reason step-by-step before outputting LABEL.

    Returns
    -------
    label : str  – "Fact" or "Opinion"
    raw   : str  – raw model response (for logging / debugging)
    """
    marker_summary = get_marker_summary()
    examples_block = format_examples_for_prompt(
        retrieve_examples(text, n_fact=2, n_opinion=2)
    )

    system_context = (
        "You are a linguistic analyst. Classify text as Fact or Opinion.\n"
        f"{marker_summary}"
    )

    prompt = (
        f"{system_context}\n\n"
        f"EXAMPLES:\n{examples_block}\n"
        f"---\n"
        f"Now classify step by step, then output LABEL:\n\n"
        f"TARGET: {text}\n"
        f"Output LABEL: "
    )

    raw = _run_model(prompt, temperature=0.0)
    label = _extract_label(raw)
    return label, raw


# ── Self-Consistency / majority-vote wrapper ──────────────────────────────────

def classify_with_voting(
    text: str,
    n_votes: int = 3,
    temperatures: List[float] = None,
    retry_delay: float = 0.5,
) -> Tuple[str, List[str], List[str]]:
    """
    Run classify_single n_votes times at different temperatures and return
    the majority-vote label (self-consistency approach).

    Parameters
    ----------
    text        : input text to classify
    n_votes     : number of independent classification calls
    temperatures: per-call temperatures; defaults to slight variation around 0
    retry_delay : seconds to wait between calls

    Returns
    -------
    final_label : majority-vote label
    labels      : individual labels from each call
    responses   : raw response strings from each call
    """
    if temperatures is None:
        step = 0.15 / max(n_votes - 1, 1)
        temperatures = [round(i * step, 2) for i in range(n_votes)]

    labels: List[str] = []
    responses: List[str] = []

    for i in range(n_votes):
        temp = temperatures[i] if i < len(temperatures) else 0.0
        try:
            lbl, raw = classify_single(text)
            labels.append(lbl)
            responses.append(raw)
            logger.debug("  Vote %d/%d -> %s (temp=%.2f)", i + 1, n_votes, lbl, temp)
        except Exception as exc:
            logger.warning("  Vote %d/%d failed: %s", i + 1, n_votes, exc)
            labels.append("Opinion")
            responses.append(str(exc))

        if i < n_votes - 1:
            time.sleep(retry_delay)

    fact_votes = labels.count("Fact")
    opinion_votes = labels.count("Opinion")
    final_label = "Fact" if fact_votes >= opinion_votes else "Opinion"

    logger.debug(
        "  Voting result: Fact=%d, Opinion=%d -> %s",
        fact_votes, opinion_votes, final_label,
    )
    return final_label, labels, responses
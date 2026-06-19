"""
Stage 1 – Feature Extraction
==============================
Extracts traditional NLP features without any transformer models:
  • POS tag distributions
  • Named Entity Recognition statistics
  • VADER sentiment / emotional tone
  • Readability scores (Flesch, Gunning-Fog, SMOG, Coleman-Liau)
  • Lexical diversity and stylistic markers
  • First-person / hedging / epistemic verb counts
  • Knowledge-base opinion/fact indicator word counts
  • Error analysis: spelling and grammatical error density
"""

import re
import logging
import math
import nltk
import numpy as np
import pandas as pd
import textstat
from collections import Counter
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, words as nltk_words
from typing import List

logger = logging.getLogger(__name__)

# ── NLTK assets ───────────────────────────────────────────────────────────────
_REQUIRED = [
    ("tokenizers/punkt_tab",                   "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng",  "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab",          "maxent_ne_chunker_tab"),
    ("corpora/words",                           "words"),
    ("sentiment/vader_lexicon",                 "vader_lexicon"),
    ("corpora/stopwords",                       "stopwords"),
]
for _path, _pkg in _REQUIRED:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

_STOP          = set(stopwords.words("english"))
_SIA           = SentimentIntensityAnalyzer()
_ENGLISH_VOCAB = set(w.lower() for w in nltk_words.words())


# ── Opinion / Fact word lists (knowledge base) ────────────────────────────────

OPINION_LEXICON = {
    # Belief / perspective markers
    "i think", "i believe", "i feel", "in my opinion", "in my view",
    "it seems", "it appears", "arguably", "perhaps", "possibly",
    "probably", "likely", "unlikely", "supposedly", "allegedly",
    "apparently", "presumably", "contentious", "controversial",
    "debatable", "subjective", "questionable",
    # Evaluative adjectives
    "good", "bad", "best", "worst", "great", "terrible", "wonderful",
    "awful", "beautiful", "ugly", "amazing", "horrible", "fantastic",
    "boring", "exciting", "interesting", "dull", "excellent", "poor",
    "superior", "inferior",
    # Hedge words
    "might", "could", "should", "would", "may", "ought",
    "seem", "appear", "suggest", "indicate", "tend",
}

FACT_LEXICON = {
    # Certainty / attribution markers
    "according to", "studies show", "research indicates", "data shows",
    "statistics reveal", "scientists found", "researchers discovered",
    "the report states", "evidence suggests", "it is known that",
    "has been proven", "is defined as", "is classified as",
    "on record", "officially", "documented", "confirmed", "verified",
    "measured", "calculated", "recorded", "published",
}

HEDGING_WORDS = {
    "might", "may", "could", "would", "should", "seem", "appears",
    "suggests", "indicates", "possibly", "perhaps", "probably",
    "likely", "supposedly", "allegedly", "apparently", "presumably",
    "arguably", "believe", "think", "feel", "tend", "generally",
}

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"}

EPISTEMIC_VERBS = {
    "think", "believe", "feel", "know", "suppose", "assume", "expect",
    "suspect", "imagine", "guess", "wonder", "doubt", "hope",
}

REPORTING_VERBS = {
    "said", "stated", "claimed", "argued", "noted", "reported",
    "announced", "confirmed", "revealed", "showed", "found",
    "concluded", "demonstrated", "suggested", "indicated",
}


# ── Core feature extractor ────────────────────────────────────────────────────

def _safe_divide(a, b, default=0.0):
    return a / b if b else default


def _count_spelling_errors(words_lower: List[str]) -> int:
    """
    Count tokens that are purely alphabetic but absent from the NLTK English
    word list.  Proper nouns and very short words (len <= 2) are excluded to
    avoid false positives.
    """
    errors = 0
    for w in words_lower:
        if len(w) <= 2:
            continue
        if w not in _ENGLISH_VOCAB:
            errors += 1
    return errors


def extract_features(text: str) -> dict:
    """Return a flat dict of numeric features for one text sample."""
    text_lower  = text.lower()
    words_raw   = word_tokenize(text)
    words_clean = [w for w in words_raw if w.isalpha()]
    words_lower = [w.lower() for w in words_clean]
    sents       = sent_tokenize(text)

    n_words = len(words_clean)
    n_sents = max(len(sents), 1)
    n_chars = len(text)

    # ── POS tagging ──────────────────────────────────────────────────────
    pos_tags   = pos_tag(words_raw)
    pos_counts = Counter(tag for _, tag in pos_tags)
    total_pos  = max(sum(pos_counts.values()), 1)

    n_nouns = sum(v for k, v in pos_counts.items() if k.startswith("NN"))
    n_verbs = sum(v for k, v in pos_counts.items() if k.startswith("VB"))
    n_adjs  = sum(v for k, v in pos_counts.items() if k.startswith("JJ"))
    n_advs  = sum(v for k, v in pos_counts.items() if k.startswith("RB"))
    n_det   = pos_counts.get("DT", 0)
    n_modal = pos_counts.get("MD", 0)
    n_cd    = pos_counts.get("CD", 0)   # cardinal numbers
    n_nnp   = sum(v for k, v in pos_counts.items() if k in ("NNP", "NNPS"))

    # ── Named Entity Recognition ──────────────────────────────────────────
    try:
        ne_tree   = ne_chunk(pos_tags, binary=False)
        ne_labels = [subtree.label() for subtree in ne_tree if hasattr(subtree, "label")]
        ne_counts = Counter(ne_labels)
    except Exception:
        ne_counts = Counter()

    n_ner_total  = sum(ne_counts.values())
    n_ner_person = ne_counts.get("PERSON", 0)
    n_ner_org    = ne_counts.get("ORGANIZATION", 0)
    n_ner_gpe    = ne_counts.get("GPE", 0)

    # ── Sentiment / emotion (VADER) ───────────────────────────────────────
    vader     = _SIA.polarity_scores(text)
    sent_neg  = vader["neg"]
    sent_neu  = vader["neu"]
    sent_pos  = vader["pos"]
    sent_comp = vader["compound"]
    sent_abs  = abs(sent_comp)

    sent_compounds = [_SIA.polarity_scores(s)["compound"] for s in sents]
    sent_variance  = float(np.var(sent_compounds)) if len(sent_compounds) > 1 else 0.0

    # ── Readability ───────────────────────────────────────────────────────
    try:
        fk_grade = textstat.flesch_kincaid_grade(text)
        fk_ease  = textstat.flesch_reading_ease(text)
        fog      = textstat.gunning_fog(text)
        smog     = textstat.smog_index(text)
        cli      = textstat.coleman_liau_index(text)
        ari      = textstat.automated_readability_index(text)
    except Exception:
        fk_grade = fk_ease = fog = smog = cli = ari = 0.0

    # ── Lexical diversity ─────────────────────────────────────────────────
    types         = set(words_lower)
    ttr           = _safe_divide(len(types), n_words)
    content       = [w for w in words_lower if w not in _STOP and len(w) > 2]
    content_ratio = _safe_divide(len(content), n_words)
    avg_word_len  = np.mean([len(w) for w in words_clean]) if words_clean else 0.0
    avg_sent_len  = _safe_divide(n_words, n_sents)

    # ── Opinion / Fact lexicon counts ─────────────────────────────────────
    opinion_hits = sum(1 for phrase in OPINION_LEXICON if phrase in text_lower)
    fact_hits    = sum(1 for phrase in FACT_LEXICON    if phrase in text_lower)

    # ── Hedging words ─────────────────────────────────────────────────────
    hedge_count = sum(1 for w in words_lower if w in HEDGING_WORDS)
    hedge_ratio = _safe_divide(hedge_count, n_words)

    # ── First-person pronouns ─────────────────────────────────────────────
    fp_count = sum(1 for w in words_lower if w in FIRST_PERSON)
    fp_ratio = _safe_divide(fp_count, n_words)

    # ── Epistemic / reporting verbs ───────────────────────────────────────
    epist_count  = sum(1 for w in words_lower if w in EPISTEMIC_VERBS)
    report_count = sum(1 for w in words_lower if w in REPORTING_VERBS)

    # ── Punctuation patterns ──────────────────────────────────────────────
    n_excl  = text.count("!")
    n_quest = text.count("?")
    n_quote = text.count('"') + text.count("'")

    # ── Number / statistics density ───────────────────────────────────────
    num_tokens   = re.findall(r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*percent)?\b", text)
    num_ratio    = _safe_divide(len(num_tokens), n_words)
    has_percent  = int(bool(re.search(r"\d+\s*%|\d+\s*percent", text_lower)))
    has_citation = int(bool(re.search(
        r"\b(according to|source:|cited|reference|ibid|op\.?\s*cit\.?)\b",
        text_lower
    )))

    # ── Sentence complexity (subordinating conjunctions) ──────────────────
    subj_conj  = {"although", "because", "since", "while", "whereas",
                  "unless", "until", "whenever", "if", "even though"}
    subj_count = sum(1 for w in words_lower if w in subj_conj)

    # ── Error Analysis: spelling error density ────────────────────────────
    # Count tokens not found in the NLTK English vocabulary.
    # High error density may correlate with informal / opinion writing.
    spell_errors       = _count_spelling_errors(words_lower)
    spell_error_ratio  = _safe_divide(spell_errors, n_words)

    # ── Assemble feature dict ─────────────────────────────────────────────
    feats = {
        # Basic statistics
        "n_words":           n_words,
        "n_sents":           n_sents,
        "n_chars":           n_chars,
        "avg_word_len":      avg_word_len,
        "avg_sent_len":      avg_sent_len,
        # POS ratios
        "noun_ratio":        _safe_divide(n_nouns, total_pos),
        "verb_ratio":        _safe_divide(n_verbs, total_pos),
        "adj_ratio":         _safe_divide(n_adjs,  total_pos),
        "adv_ratio":         _safe_divide(n_advs,  total_pos),
        "modal_ratio":       _safe_divide(n_modal, total_pos),
        "num_token_ratio":   _safe_divide(n_cd,    total_pos),
        "prop_noun_ratio":   _safe_divide(n_nnp,   total_pos),
        # NER
        "ner_total":         n_ner_total,
        "ner_person":        n_ner_person,
        "ner_org":           n_ner_org,
        "ner_gpe":           n_ner_gpe,
        "ner_density":       _safe_divide(n_ner_total, n_words),
        # Sentiment
        "vader_neg":         sent_neg,
        "vader_neu":         sent_neu,
        "vader_pos":         sent_pos,
        "vader_compound":    sent_comp,
        "vader_abs":         sent_abs,
        "sent_variance":     sent_variance,
        # Readability
        "fk_grade":          fk_grade,
        "fk_ease":           fk_ease,
        "gunning_fog":       fog,
        "smog":              smog,
        "coleman_liau":      cli,
        "ari":               ari,
        # Lexical diversity
        "ttr":               ttr,
        "content_ratio":     content_ratio,
        # Lexicon-based
        "opinion_lexicon":   opinion_hits,
        "fact_lexicon":      fact_hits,
        "lexicon_diff":      opinion_hits - fact_hits,
        # Hedging / stance
        "hedge_count":       hedge_count,
        "hedge_ratio":       hedge_ratio,
        "fp_count":          fp_count,
        "fp_ratio":          fp_ratio,
        "epist_count":       epist_count,
        "report_count":      report_count,
        # Punctuation
        "n_exclamation":     n_excl,
        "n_question":        n_quest,
        "n_quote":           n_quote,
        # Numbers / evidence
        "num_ratio":         num_ratio,
        "has_percent":       has_percent,
        "has_citation":      has_citation,
        # Syntactic complexity
        "subj_count":        subj_count,
        # Error Analysis
        "spell_errors":      spell_errors,
        "spell_error_ratio": spell_error_ratio,
    }
    return feats


def extract_features_batch(texts: List[str], show_progress: bool = True) -> pd.DataFrame:
    """Extract features for a list of texts, return DataFrame."""
    from tqdm import tqdm
    results = []
    it = tqdm(texts, desc="Extracting features", unit="doc") if show_progress else texts
    for text in it:
        results.append(extract_features(str(text)))
    return pd.DataFrame(results)

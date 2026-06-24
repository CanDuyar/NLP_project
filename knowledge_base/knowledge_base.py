"""
Knowledge Base for Stage 2 – Fact vs Opinion Classification
=============================================================
Structured linguistic and pragmatic knowledge for the Stage 2 classifier.

WHAT THE PIPELINE ACTUALLY USES
-------------------------------
The fine-tuned BART-base method (stage2/finetune.py) uses this KB in a
RAG-style way: `compute_marker_counts` / `marker_feature_string` retrieve which
linguistic-marker categories fire for a given text from LINGUISTIC_MARKERS (the
"document store") and prepend a compact "[kb] ..." tag to the model input. The
model then learns from labelled data how those retrieved signals relate to the
Fact / Opinion label. This is the only KB code wired into the running pipeline.

AUXILIARY / NOT WIRED INTO THE PIPELINE
---------------------------------------
The few-shot example pool (FEW_SHOT_EXAMPLES) and the lexical retrieval helpers
(`retrieve_examples`, `format_examples_for_prompt`, `get_marker_summary`) are
provided for *prompting-based* approaches (few-shot / chain-of-thought against
an instruction-following LLM). Plain BART-base is not instruction-tuned, so the
current method does not prompt it — these utilities are kept as a reusable
resource (and serialised to knowledge_base.json) but are NOT invoked by
stage2_pipeline. Do not describe them as part of the implemented method unless
you actually wire them in.
"""

import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict


logger = logging.getLogger(__name__)

KB_DIR = Path(__file__).parent


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Few-Shot Example Pool
#     Each entry: text, label, rationale
# ══════════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    # ── FACTS ────────────────────────────────────────────────────────────────
    {
        "text":  "The Amazon rainforest covers approximately 5.5 million square kilometres and is home to around 10 percent of all species on Earth.",
        "label": "Fact",
        "rationale": (
            "This sentence presents two quantifiable, verifiable measurements "
            "(area in km² and biodiversity percentage). It uses precise numerical "
            "data without hedging, subjective evaluation, or first-person stance. "
            "Both figures can be independently verified against scientific sources."
        ),
    },
    {
        "text":  "According to the World Health Organization, approximately 2 billion people lack access to safe drinking water.",
        "label": "Fact",
        "rationale": (
            "Attributed to a recognised authority (WHO) and backed by a specific "
            "statistic. 'According to' is a classic reporting phrase that signals "
            "an external, verifiable source rather than personal opinion."
        ),
    },
    {
        "text":  "The human genome contains approximately 3 billion base pairs and encodes around 20,000 protein-coding genes.",
        "label": "Fact",
        "rationale": (
            "Contains concrete scientific measurements with no subjective or "
            "evaluative language. The information is established scientific "
            "consensus documented in peer-reviewed literature."
        ),
    },
    {
        "text":  "Germany reunified on 3 October 1990 after the fall of the Berlin Wall in 1989.",
        "label": "Fact",
        "rationale": (
            "States a historical event with a precise date. Historical facts can "
            "be verified against primary sources and encyclopaedic references. "
            "No opinion markers or hedging are present."
        ),
    },
    {
        "text":  "The unemployment rate in the United States fell to 3.4 percent in January 2023, the lowest since 1969.",
        "label": "Fact",
        "rationale": (
            "Cites a government-published economic statistic with a specific date "
            "and comparative context. Uses objective data rather than personal "
            "assessment of the economy's health."
        ),
    },
    {
        "text":  "The Great Barrier Reef stretches over 2,300 kilometres along the coast of Queensland, Australia.",
        "label": "Fact",
        "rationale": (
            "Provides a geographic measurement (length and location) that is "
            "objectively verifiable. No personal stance, evaluative adjective, "
            "or uncertainty marker is present."
        ),
    },
    {
        "text":  "Researchers at MIT developed an algorithm capable of detecting certain cancers from blood samples with 95 percent accuracy.",
        "label": "Fact",
        "rationale": (
            "Reports a research finding attributed to a specific institution "
            "with a quantified performance metric. Research results are "
            "verifiable through the published study."
        ),
    },
    {
        "text":  "The French Revolution began in 1789 and led to the abolition of the monarchy and the rise of Napoleon Bonaparte.",
        "label": "Fact",
        "rationale": (
            "Describes historical events with documented dates and consequences. "
            "Both the start year and the political outcomes are part of the "
            "established historical record."
        ),
    },

    # ── OPINIONS ─────────────────────────────────────────────────────────────
    {
        "text":  "I believe the government's handling of the economic crisis was nothing short of catastrophic.",
        "label": "Opinion",
        "rationale": (
            "Opens with 'I believe', a classic first-person epistemic marker. "
            "The superlative evaluation ('nothing short of catastrophic') is "
            "inherently subjective and emotive rather than data-backed. The same "
            "policy could be judged very differently by another observer."
        ),
    },
    {
        "text":  "In my view, electric vehicles are far superior to traditional combustion engine cars in every meaningful way.",
        "label": "Opinion",
        "rationale": (
            "'In my view' immediately signals a personal perspective. 'Far "
            "superior … in every meaningful way' is an absolute evaluative claim "
            "that cannot be uniformly verified — it depends on criteria chosen "
            "by the author."
        ),
    },
    {
        "text":  "This new regulation seems like an overreach that will likely stifle innovation and damage the economy.",
        "label": "Opinion",
        "rationale": (
            "'Seems like' and 'will likely' are hedging expressions indicating "
            "prediction rather than established fact. 'Overreach' is a subjective "
            "political judgement, and the economic impact is speculative."
        ),
    },
    {
        "text":  "Social media has arguably done more to divide society than to connect it.",
        "label": "Opinion",
        "rationale": (
            "'Arguably' is a classic hedge that signals the author's awareness "
            "that their claim is contestable. The comparison is a value judgement "
            "that depends on how 'divide' and 'connect' are measured and weighted."
        ),
    },
    {
        "text":  "The film is a stunning visual achievement, though its narrative is disappointingly shallow.",
        "label": "Opinion",
        "rationale": (
            "Evaluative adjectives ('stunning', 'disappointingly shallow') "
            "represent aesthetic judgements that vary between critics. Such "
            "assessments are inherently subjective and cannot be objectively "
            "verified."
        ),
    },
    {
        "text":  "Perhaps the most troubling aspect of modern politics is the apparent decline in civil discourse.",
        "label": "Opinion",
        "rationale": (
            "'Perhaps' signals uncertainty and subjective interpretation. "
            "'Most troubling' is an evaluative superlative. 'Apparent decline' "
            "combines a hedging word with a value-loaded description that "
            "different people may assess very differently."
        ),
    },
    {
        "text":  "The city should invest heavily in green infrastructure rather than continuing to subsidise fossil fuels.",
        "label": "Opinion",
        "rationale": (
            "The modal 'should' expresses a normative prescription — the author's "
            "view of what ought to be done — rather than reporting what is the "
            "case. Policy recommendations are inherently opinion-based."
        ),
    },
    {
        "text":  "Many experts believe the central bank made a serious error by raising interest rates so aggressively.",
        "label": "Opinion",
        "rationale": (
            "Even attributed to 'many experts', the core claim ('serious error') "
            "is an evaluative judgement. 'Believe' signals subjective assessment. "
            "Other equally qualified experts hold the opposite view."
        ),
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Linguistic Marker Reference
# ══════════════════════════════════════════════════════════════════════════════

LINGUISTIC_MARKERS = {
    "opinion_strong": [
        "i think", "i believe", "i feel", "in my opinion", "in my view",
        "it is my conviction", "i am convinced", "personally",
    ],
    "opinion_hedge": [
        "perhaps", "possibly", "probably", "arguably", "seemingly",
        "apparently", "presumably", "supposedly", "may", "might", "could",
        "would suggest", "seems", "appears",
    ],
    "opinion_evaluative": [
        "best", "worst", "beautiful", "terrible", "amazing", "awful",
        "outstanding", "disastrous", "brilliant", "appalling", "superior",
        "inferior", "excellent", "dreadful",
    ],
    "opinion_normative": [
        "should", "must", "ought to", "need to", "has to", "have to",
        "it is necessary", "it is essential", "it is vital",
    ],
    "fact_attribution": [
        "according to", "as reported by", "studies show", "research indicates",
        "data reveals", "statistics demonstrate", "the report states",
        "scientists found", "researchers discovered", "officially",
    ],
    "fact_quantitative": [
        "percent", "%", "million", "billion", "kilometres", "degrees",
        "square", "per capita", "ratio", "rate",
    ],
    "fact_temporal": [
        "in 19", "in 20", "on january", "on february", "on march",
        "founded in", "established in", "born in", "died in",
    ],
    "first_person": [
        "i ", "me ", "my ", "mine", "myself", "we ", "us ", "our ", "ours",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Simple TF-IDF-like retrieval for Few-Shot selection
# ══════════════════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> List[str]:
    import re
    return re.findall(r"\b[a-z]{3,}\b", text.lower())


def _build_idf(corpus: List[List[str]]) -> Dict[str, float]:
    n = len(corpus)
    df = defaultdict(int)
    for doc in corpus:
        for token in set(doc):
            df[token] += 1
    return {tok: math.log((n + 1) / (cnt + 1)) + 1 for tok, cnt in df.items()}


# Pre-compute IDF over the few-shot pool
_POOL_TOKENS = [_tokenize(ex["text"]) for ex in FEW_SHOT_EXAMPLES]
_IDF = _build_idf(_POOL_TOKENS)


def _tfidf_vec(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    tf_counts = defaultdict(int)
    for t in tokens:
        tf_counts[t] += 1
    n = max(len(tokens), 1)
    return {t: (cnt / n) * idf.get(t, 1.0) for t, cnt in tf_counts.items()}


def _cosine(v1: dict, v2: dict) -> float:
    shared = set(v1) & set(v2)
    if not shared:
        return 0.0
    dot  = sum(v1[k] * v2[k] for k in shared)
    mag1 = math.sqrt(sum(x ** 2 for x in v1.values()))
    mag2 = math.sqrt(sum(x ** 2 for x in v2.values()))
    return dot / (mag1 * mag2 + 1e-9)


def retrieve_examples(
    query_text: str,
    n_fact: int = 2,
    n_opinion: int = 2,
) -> List[Dict]:
    """
    Return the most lexically similar Few-Shot examples to query_text,
    with balanced representation: n_fact Fact examples + n_opinion Opinion examples.
    """
    q_toks = _tokenize(query_text)
    q_vec  = _tfidf_vec(q_toks, _IDF)

    scored_fact    = []
    scored_opinion = []

    for i, ex in enumerate(FEW_SHOT_EXAMPLES):
        ex_vec  = _tfidf_vec(_POOL_TOKENS[i], _IDF)
        sim     = _cosine(q_vec, ex_vec)
        if ex["label"] == "Fact":
            scored_fact.append((sim, ex))
        else:
            scored_opinion.append((sim, ex))

    top_facts    = [ex for _, ex in sorted(scored_fact,    key=lambda x: x[0], reverse=True)[:n_fact]]
    top_opinions = [ex for _, ex in sorted(scored_opinion, key=lambda x: x[0], reverse=True)[:n_opinion]]

    # Interleave Fact / Opinion for balanced prompt context
    combined = []
    for f, o in zip(top_facts, top_opinions):
        combined.extend([f, o])
    return combined


def format_examples_for_prompt(examples: List[dict]) -> str:
    """Format few-shot examples as a numbered, structured block for the LLM prompt."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}:")
        lines.append(f"  Text: \"{ex['text']}\"")
        lines.append(f"  Label: {ex['label']}")
        lines.append(f"  Reasoning: {ex['rationale']}")
        lines.append("")
    return "\n".join(lines)


def get_marker_summary() -> str:
    """Return a concise textual summary of linguistic markers for the system prompt."""
    lines = [
        "Key linguistic signals for classification:",
        "",
        "OPINION signals:",
        "  • First-person stance: 'I think', 'I believe', 'In my view', 'personally'",
        "  • Hedging language: 'perhaps', 'arguably', 'seemingly', 'appears', modal verbs",
        "  • Evaluative adjectives: 'best', 'terrible', 'outstanding', 'appalling'",
        "  • Normative claims: 'should', 'must', 'ought to'",
        "  • Emotional or subjective tone (VADER compound score far from 0)",
        "",
        "FACT signals:",
        "  • Attribution phrases: 'According to', 'Studies show', 'Researchers found'",
        "  • Quantitative data: specific percentages, measurements, counts",
        "  • Temporal precision: exact dates, years, historical timelines",
        "  • Named entities with verifiable real-world referents",
        "  • Neutral, third-person, declarative sentences",
        "",
        "Edge cases to consider:",
        "  • A sentence may ATTRIBUTE an opinion to someone ('Experts believe X') — classify",
        "    based on whether the ATTRIBUTED claim itself is presented as verifiable.",
        "  • Statistical predictions ('likely to grow by 3%') are closer to Fact when",
        "    attributed to a model or study, but Opinion when a personal forecast.",
        "  • Hyperbole ('the worst decision ever') is Opinion even without 'I think'.",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Marker injection for Stage 2 (RAG-style context for the LLM)
#     Stage 1 turns these markers into numeric ML features. Stage 2 instead
#     hands a short summary of them straight to the language model as part
#     of its input, which is the same "retrieve relevant knowledge, then let
#     the generator use it" idea the assignment's RAG diagram describes -
#     here the "document store" is just the marker lexicon above.
# ══════════════════════════════════════════════════════════════════════════════

def compute_marker_counts(text: str) -> Dict[str, int]:
    """Count how often each marker category from LINGUISTIC_MARKERS shows up."""
    lower = text.lower()
    return {
        category: sum(lower.count(phrase) for phrase in phrases)
        for category, phrases in LINGUISTIC_MARKERS.items()
    }


def marker_feature_string(text: str) -> str:
    """
    Compact tag summarising which marker categories fired for this text,
    meant to be prepended to the raw text before it goes into the model.
    Only non-zero categories are listed so the tag stays short.
    """
    counts = compute_marker_counts(text)
    hits = [f"{cat}:{n}" for cat, n in counts.items() if n > 0]
    if not hits:
        return "[kb] none"
    return "[kb] " + " ".join(hits)


# ── Persist KB to JSON for reference ─────────────────────────────────────────

def save_knowledge_base():
    kb = {
        "few_shot_examples":  FEW_SHOT_EXAMPLES,
        "linguistic_markers": LINGUISTIC_MARKERS,
    }
    path = KB_DIR / "knowledge_base.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    logger.info("Knowledge base saved to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_knowledge_base()
    print("Knowledge base saved.")

    # Quick retrieval test
    query = "The new policy appears to be a serious mistake for the economy."
    exs = retrieve_examples(query, n_fact=2, n_opinion=2)
    print(f"\nTop examples for: '{query}'")
    for ex in exs:
        print(f"  [{ex['label']}] {ex['text'][:70]}…")

"""
Context Recall evaluator.

Measures what fraction of the gold reference context was successfully retrieved
by the pipeline. Uses sentence-level fuzzy containment so chunk-size differences
(gold ~4 000 chars vs pipeline 256-token chunks) don't penalise retrieval.

Algorithm
---------
1. Split the gold reference context into sentences.
2. For each gold sentence, check whether *any* retrieved node contains it
   (using asymmetric word-containment ≥ threshold).
3. Context Recall = (matched sentences) / (total gold sentences).

Why asymmetric containment instead of Jaccard?
  Jaccard = |A ∩ B| / |A ∪ B| is symmetric and penalises when the retrieved
  chunk is longer than the gold sentence.  For recall, we want to know "did
  the retriever surface text covering this gold sentence?", so the right
  metric is:  containment = |gold_words ∩ retrieved_words| / |gold_words|

  A threshold of 0.8 means 80 % of the gold sentence's distinctive words
  must appear in the retrieved chunk.  This tolerates minor wording
  differences from MinerU while preventing trivial matches on stop words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Drops blanks and very short fragments."""
    parts = _SENT_RE.split(text.strip())
    return [s.strip() for s in parts if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Asymmetric word-containment
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _containment(gold: str, candidate: str) -> float:
    """Fraction of gold-sentence words found in candidate text."""
    g = _word_set(gold)
    c = _word_set(candidate)
    if not g:
        return 1.0
    return len(g & c) / len(g)


def _sentence_contained(sentence: str, node_texts: list[str], threshold: float) -> bool:
    """Return True if *any* node text has word-containment ≥ threshold with the sentence."""
    for text in node_texts:
        if _containment(sentence, text) >= threshold:
            return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ContextRecallResult:
    """Result of a single Context Recall evaluation."""
    score: float                 # 0.0 – 1.0
    matched: int                 # number of gold sentences found
    total: int                   # total gold sentences
    unmatched_sentences: list[str]  # gold sentences NOT found in retrieved


def compute_context_recall(
    reference_contexts: list[str],
    retrieved_node_texts: list[str],
    threshold: float = 0.80,
) -> ContextRecallResult:
    """
    Compute Context Recall for a single query.

    Parameters
    ----------
    reference_contexts : list[str]
        Gold context passages (usually 1 long passage from the dataset).
    retrieved_node_texts : list[str]
        Text content of every node returned by the pipeline for this query.
    threshold : float
        Word-containment threshold for a sentence to count as "matched".
        Default 0.80 means 80 % of gold sentence words must appear in a chunk.

    Returns
    -------
    ContextRecallResult
    """
    # Flatten gold contexts into sentences
    gold_sentences: list[str] = []
    for ctx in reference_contexts:
        gold_sentences.extend(_split_sentences(ctx))

    if not gold_sentences:
        return ContextRecallResult(score=1.0, matched=0, total=0, unmatched_sentences=[])

    matched = 0
    unmatched: list[str] = []

    for sent in gold_sentences:
        if _sentence_contained(sent, retrieved_node_texts, threshold):
            matched += 1
        else:
            unmatched.append(sent)

    return ContextRecallResult(
        score=matched / len(gold_sentences),
        matched=matched,
        total=len(gold_sentences),
        unmatched_sentences=unmatched,
    )

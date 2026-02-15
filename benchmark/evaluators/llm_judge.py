"""
LLM-as-Judge evaluators: Faithfulness and Correctness.

Both evaluators use Claude (the same model the pipeline uses) with structured
rubric prompts.  Each returns a 1–5 score and a one-sentence reasoning string.

Design decisions
----------------
* **Faithfulness** checks whether *every claim* in the answer can be traced to
  the retrieved context.  This catches hallucinations introduced by the
  generation step, regardless of whether the answer is factually correct.

* **Correctness** checks whether the generated answer conveys the same
  information as the gold reference answer.  This catches retrieval failures
  where the pipeline found wrong context but the LLM faithfully summarised it.

Both prompts ask for JSON output ``{"score": int, "reasoning": str}`` so
results are machine-parseable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import anthropic

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CLIENT: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set")
        _CLIENT = anthropic.Anthropic(api_key=api_key)
    return _CLIENT


def _call_judge(system_prompt: str, user_prompt: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Claude and parse the JSON response."""
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text.strip()
    # Try to extract JSON from the response (handles markdown fences)
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    # Fallback: return raw text as reasoning with score 0
    return {"score": 0, "reasoning": f"Failed to parse judge response: {text[:200]}"}


# ---------------------------------------------------------------------------
# Faithfulness
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = """\
You are an impartial evaluation judge. Your task is to assess FAITHFULNESS:
whether every factual claim in the ANSWER is supported by the provided CONTEXT.

Scoring rubric (1-5):
  5 — Every claim in the answer is directly supported by the context.
  4 — Almost all claims are supported; one minor unsupported detail.
  3 — Most claims are supported, but one or two notable claims lack support.
  2 — Several claims are unsupported or contradict the context.
  1 — The answer is largely fabricated or contradicts the context.

Respond with ONLY a JSON object: {"score": <int 1-5>, "reasoning": "<one sentence>"}
"""

_FAITHFULNESS_USER = """\
QUERY:
{query}

CONTEXT (retrieved passages):
{context}

ANSWER:
{answer}
"""


@dataclass
class JudgeResult:
    """Result from an LLM judge evaluation."""
    score: int       # 1-5
    reasoning: str


def evaluate_faithfulness(
    query: str,
    context: str,
    answer: str,
    model: str = "claude-sonnet-4-20250514",
) -> JudgeResult:
    """
    Evaluate whether the answer is faithful to the retrieved context.

    Parameters
    ----------
    query : str
        The user query.
    context : str
        Concatenated text of all retrieved nodes.
    answer : str
        The pipeline's generated answer.
    model : str
        Anthropic model to use as judge.

    Returns
    -------
    JudgeResult
    """
    user_msg = _FAITHFULNESS_USER.format(query=query, context=context, answer=answer)
    result = _call_judge(_FAITHFULNESS_SYSTEM, user_msg, model=model)
    return JudgeResult(score=result.get("score", 0), reasoning=result.get("reasoning", ""))


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

_CORRECTNESS_SYSTEM = """\
You are an impartial evaluation judge. Your task is to assess CORRECTNESS:
whether the GENERATED ANSWER conveys the same key information as the
REFERENCE ANSWER.

Focus on factual alignment — the generated answer does NOT need to use the
same wording, but it must capture the same essential facts and relationships.

IMPORTANT: Do NOT penalize the generated answer for being longer, more
detailed, or more verbose than the reference answer. Extra correct
information that goes beyond the reference is perfectly acceptable and
should NOT reduce the score. Only penalize for MISSING or INCORRECT facts.

Scoring rubric (1-5):
  5 — The generated answer captures all key facts from the reference answer.
      Extra detail or context beyond the reference is fine.
  4 — Nearly all key facts are present; one minor omission.
  3 — Most key facts are present, but one or two important facts are missing
      or stated incorrectly.
  2 — Several key facts are missing or incorrect.
  1 — The generated answer is mostly wrong or irrelevant.

Respond with ONLY a JSON object: {"score": <int 1-5>, "reasoning": "<one sentence>"}
"""

_CORRECTNESS_USER = """\
QUERY:
{query}

REFERENCE ANSWER (ground truth):
{reference_answer}

GENERATED ANSWER (to evaluate):
{generated_answer}
"""


def evaluate_correctness(
    query: str,
    reference_answer: str,
    generated_answer: str,
    model: str = "claude-sonnet-4-20250514",
) -> JudgeResult:
    """
    Evaluate whether the generated answer matches the gold reference answer.

    Parameters
    ----------
    query : str
        The user query.
    reference_answer : str
        Gold reference answer from the dataset.
    generated_answer : str
        The pipeline's generated answer.
    model : str
        Anthropic model to use as judge.

    Returns
    -------
    JudgeResult
    """
    user_msg = _CORRECTNESS_USER.format(
        query=query, reference_answer=reference_answer, generated_answer=generated_answer
    )
    result = _call_judge(_CORRECTNESS_SYSTEM, user_msg, model=model)
    return JudgeResult(score=result.get("score", 0), reasoning=result.get("reasoning", ""))

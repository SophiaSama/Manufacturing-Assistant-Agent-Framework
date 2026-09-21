"""Evaluation judges for NGA Manufacturing Assistant.

Supports both:
1. TypeSafe AI Jev Judge (System One calibrated scoring, sub-50ms, non-generative)
2. Classical LLM-as-a-judge (generative text completion)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Rubric definitions for manufacturing domain evaluation (0–5)
MANUFACTURING_JUDGE_CRITERIA = [
    "0: Completely incorrect, irrelevant, or fully hallucinated.",
    "1: Severe factual errors with minimal overlap to the golden answer.",
    "2: Partially correct, but missing critical facts, values, or introduces noticeable hallucinations.",
    "3: Mostly correct; captures primary facts with minor omissions or slight phrasing imprecision.",
    "4: Accurate, well-grounded, and complete with only minor phrasing variations.",
    "5: Perfect factual match to the golden answer; completely accurate, grounded, and comprehensive.",
]

MANUFACTURING_JUDGE_INSTRUCTIONS = (
    "You are an expert manufacturing quality evaluator. "
    "Score the candidate answer against the golden reference answer on "
    "correctness (accurate facts and specs), groundedness (no hallucinated numbers or fake references), "
    "and completeness (all critical parts covered)."
)


def make_jev_judge(
    client: Any | None = None,
    use_rounding: bool = True,
) -> Callable[[str, str], int]:
    """Create a Jev-powered judge that returns a 0–5 integer score.
    
    Compatible with existing `judge: Callable[[str, str], int]` signatures in scoring.py.
    """
    if client is None:
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()

    from typesafe_sdk import Score

    def judge(candidate: str, golden: str) -> int:
        state = f"[GOLDEN REFERENCE ANSWER]:\n{golden}\n\n[CANDIDATE ANSWER TO EVALUATE]:\n{candidate}"
        questions = {
            "score": Score(
                instructions=MANUFACTURING_JUDGE_INSTRUCTIONS,
                criteria=MANUFACTURING_JUDGE_CRITERIA,
            )
        }
        try:
            res = client.system_one(state=state, questions=questions)
            val = res.answers["score"].score
            if use_rounding:
                return max(0, min(5, round(val)))
            return max(0, min(5, int(val)))
        except Exception as e:
            logger.warning(f"Jev judge evaluation failed: {e}")
            return 0

    return judge


def evaluate_with_jev(
    candidate: str,
    golden: str,
    client: Any | None = None,
) -> Dict[str, Any]:
    """Full diagnostic evaluation using Jev returning rich metrics and calibrated distributions."""
    if client is None:
        from typesafe_sdk import TypeSafeClient
        client = TypeSafeClient()

    from typesafe_sdk import Noul, Score

    state = f"[GOLDEN REFERENCE ANSWER]:\n{golden}\n\n[CANDIDATE ANSWER TO EVALUATE]:\n{candidate}"
    questions = {
        "score": Score(
            instructions=MANUFACTURING_JUDGE_INSTRUCTIONS,
            criteria=MANUFACTURING_JUDGE_CRITERIA,
        ),
        "is_faithful": Noul(
            instructions="Does the candidate answer contain zero hallucinated specs, values, or documents?"
        ),
        "covers_core_intent": Noul(
            instructions="Does the candidate answer address the core question answered by the golden reference?"
        ),
    }

    t0 = time.perf_counter()
    res = client.system_one(state=state, questions=questions)
    latency_ms = (time.perf_counter() - t0) * 1000

    score_ans = res.answers["score"]
    faithful_ans = res.answers["is_faithful"]
    core_ans = res.answers["covers_core_intent"]

    return {
        "score_int": max(0, min(5, round(score_ans.score))),
        "score_continuous": score_ans.score,
        "confidence": getattr(score_ans, "confidence", None),
        "probabilities": getattr(score_ans, "probabilities", {}),
        "is_faithful": faithful_ans.noul,
        "covers_core_intent": core_ans.noul,
        "latency_ms": latency_ms,
        "model": getattr(res, "model", "jev"),
    }


def make_llm_judge(settings: Any | None = None) -> Callable[[str, str], int]:
    """Create a classical generative LLM-as-a-judge returning 0–5."""
    if settings is None:
        from nga.config import Settings
        settings = Settings.from_env()

    from langchain_core.messages import HumanMessage
    from nga.providers.factory import make_chat_model

    llm = make_chat_model(settings)

    def judge(candidate: str, golden: str) -> int:
        prompt = (
            "You are an expert manufacturing quality evaluator.\n"
            "Score the candidate answer vs the golden answer on:\n"
            "  - Correctness (does it state the right facts?)\n"
            "  - Groundedness (no hallucinated numbers or doc references?)\n"
            "  - Completeness (does it cover all key points?)\n\n"
            f"Golden: {golden}\n\nCandidate: {candidate}\n\n"
            "Return ONLY an integer 0–5 (5=perfect). No explanation."
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            return max(0, min(5, int(content.strip().split()[0])))
        except Exception as e:
            logger.warning(f"LLM judge evaluation failed: {e}")
            return 0

    return judge


def get_judge(backend: str = "auto") -> Callable[[str, str], int]:
    """Factory to get the appropriate judge backend.
    
    Args:
        backend: 'auto', 'jev', or 'llm'.
                 'auto' selects Jev if TYPESAFE_API_KEY is present, else falls back to LLM.
    """
    has_typesafe = bool(os.getenv("TYPESAFE_API_KEY"))

    if backend == "jev" or (backend == "auto" and has_typesafe):
        logger.info("Using TypeSafe Jev Judge for evaluation.")
        return make_jev_judge()
    
    logger.info("Using LLM-as-a-judge for evaluation.")
    return make_llm_judge()

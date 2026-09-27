"""Evaluation judges for NGA Manufacturing Assistant.

Supports both:
1. TypeSafe AI Jev Judge (System One calibrated scoring, sub-50ms, non-generative)
2. Classical LLM-as-a-judge (generative text completion)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

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


FAILURE_MODE_CRITERIA = {
    "none": "The answer is accurate, well-grounded, and complete.",
    "inaccurate_specs": "Contains incorrect or hallucinated numbers, limits, tolerances, or units.",
    "missing_procedure": "Omitted important assembly, safety, or verification steps.",
    "hallucinated_reference": "Cites non-existent SOPs, drawings, work orders, or stations.",
    "intent_mismatch": "Addressed a different question or misunderstood the inquiry.",
}


@dataclass
class JevEvaluationResult:
    """Full diagnostic result of TypeSafe parallel questions fan-out evaluation."""

    score_int: int
    score_continuous: float
    confidence: float | None
    probabilities: dict[str, float]
    is_faithful: float
    specs_accurate: float
    completeness: float
    covers_core_intent: float
    failure_mode: str
    failure_mode_probabilities: dict[str, float]
    latency_ms: float
    model: str = "jev"
    gated_reason: str | None = None

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __iter__(self):
        return iter(asdict(self))

    def keys(self):
        return asdict(self).keys()

    def values(self):
        return asdict(self).values()

    def items(self):
        return asdict(self).items()


class JevScore(int):
    """An int subclass representing judge score (0-5) with attached TypeSafe diagnostics."""

    diagnostics: JevEvaluationResult | None

    def __new__(cls, value: int, diagnostics: JevEvaluationResult | None = None):
        obj = super().__new__(cls, value)
        obj.diagnostics = diagnostics
        return obj


def make_jev_judge(
    client: Any | None = None,
    use_rounding: bool = True,
    apply_safety_gating: bool = True,
) -> Callable[[str, str], int]:
    """Create a Jev-powered judge that returns a JevScore (int subclass with diagnostics).

    Compatible with existing `judge: Callable[[str, str], int]` signatures in scoring.py.
    """
    if client is None:
        from typesafe_sdk import TypeSafeClient

        client = TypeSafeClient()

    def judge(candidate: str, golden: str) -> JevScore:
        try:
            diag = evaluate_with_jev(
                candidate=candidate,
                golden=golden,
                client=client,
                apply_safety_gating=apply_safety_gating,
                use_rounding=use_rounding,
            )
            return JevScore(diag.score_int, diagnostics=diag)
        except Exception as e:
            logger.warning(f"Jev judge evaluation failed: {e}")
            return JevScore(0, diagnostics=None)

    return judge


def evaluate_with_jev(
    candidate: str,
    golden: str,
    client: Any | None = None,
    apply_safety_gating: bool = True,
    use_rounding: bool = True,
) -> JevEvaluationResult:
    """Full diagnostic evaluation using TypeSafe parallel questions fan-out.

    Batches 5 evaluation dimensions into a single System One API call:
    1. Score: 0–5 calibrated continuous/discrete rating
    2. is_faithful: Noul check for hallucination-free evidence grounding
    3. specs_accurate: Noul check for exact numerical/tolerance accuracy
    4. completeness: Noul check for full coverage of core intent & steps
    5. failure_mode: Choice classification of root-cause defect
    """
    if client is None:
        from typesafe_sdk import TypeSafeClient

        client = TypeSafeClient()

    from typesafe_sdk import Choice, Noul, Score

    state = f"[GOLDEN REFERENCE ANSWER]:\n{golden}\n\n[CANDIDATE ANSWER TO EVALUATE]:\n{candidate}"
    questions = {
        "score": Score(
            instructions=MANUFACTURING_JUDGE_INSTRUCTIONS,
            criteria=MANUFACTURING_JUDGE_CRITERIA,
        ),
        "is_faithful": Noul(
            instructions="Does the candidate answer contain zero hallucinated specs, values, or non-existent document references?"
        ),
        "specs_accurate": Noul(
            instructions="Are all numerical figures, torque specs, pressures, dimensions, and tolerance ranges strictly accurate according to the golden reference?"
        ),
        "completeness": Noul(
            instructions="Does the candidate answer address all core questions, steps, and prerequisites covered in the golden reference?"
        ),
        "failure_mode": Choice(
            instructions="If the candidate answer is defective or incomplete, what is the primary failure mode?",
            criteria=FAILURE_MODE_CRITERIA,
        ),
    }

    t0 = time.perf_counter()
    res = client.system_one(state=state, questions=questions)
    latency_ms = (time.perf_counter() - t0) * 1000

    answers = getattr(res, "answers", {})
    score_ans = answers.get("score")
    faithful_ans = answers.get("is_faithful")
    specs_ans = answers.get("specs_accurate")
    comp_ans = answers.get("completeness")
    failure_ans = answers.get("failure_mode")

    raw_score = getattr(score_ans, "score", 0.0) if score_ans else 0.0
    score_val = round(raw_score) if use_rounding else int(raw_score)
    score_val = max(0, min(5, score_val))

    faith_prob = getattr(faithful_ans, "noul", 1.0) if faithful_ans else 1.0
    specs_prob = getattr(specs_ans, "noul", 1.0) if specs_ans else 1.0
    comp_prob = getattr(comp_ans, "noul", 1.0) if comp_ans else 1.0
    failure_choice = getattr(failure_ans, "choice", "none") if failure_ans else "none"
    failure_probs = getattr(failure_ans, "probabilities", {}) if failure_ans else {}

    gated_reason = None
    if apply_safety_gating:
        # Gating 1: Severe hallucination or non-existent reference clamp score to <= 1
        if faith_prob < 0.3 or failure_choice == "hallucinated_reference":
            if score_val > 1:
                gated_reason = (
                    f"Clamped from {score_val} to 1 due to severe hallucination "
                    f"(faith={faith_prob:.2f}, failure_mode={failure_choice})"
                )
                score_val = 1
        # Gating 2: Severe specification inaccuracy clamp score to <= 2
        elif specs_prob < 0.3 or failure_choice == "inaccurate_specs":
            if score_val > 2:
                gated_reason = (
                    f"Clamped from {score_val} to 2 due to inaccurate specifications "
                    f"(specs_acc={specs_prob:.2f}, failure_mode={failure_choice})"
                )
                score_val = 2

    return JevEvaluationResult(
        score_int=score_val,
        score_continuous=raw_score,
        confidence=getattr(score_ans, "confidence", None) if score_ans else None,
        probabilities=getattr(score_ans, "probabilities", {}) if score_ans else {},
        is_faithful=faith_prob,
        specs_accurate=specs_prob,
        completeness=comp_prob,
        covers_core_intent=comp_prob,
        failure_mode=failure_choice,
        failure_mode_probabilities=failure_probs,
        latency_ms=latency_ms,
        model=getattr(res, "model", "jev"),
        gated_reason=gated_reason,
    )


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

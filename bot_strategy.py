"""
Forecasting strategy utilities: decomposition, base rates, time horizon,
aggregation, calibration logging, and Tavily budget control.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    DateQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    clean_indents,
    structure_output,
)
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FORECAST_TEMPERATURES = (0.2, 0.45, 0.7)


class ResolutionAnalysis(BaseModel):
    yes_conditions: list[str] = Field(
        description="Exact conditions that would resolve the question Yes"
    )
    no_conditions: list[str] = Field(
        description="Exact conditions that would resolve the question No"
    )
    ambiguities: list[str] = Field(
        description="Ambiguities or edge cases in the resolution criteria"
    )
    key_definitions: list[str] = Field(
        description="Important terms that need precise interpretation"
    )


class DecomposedQueries(BaseModel):
    queries: list[str] = Field(
        description="Prioritized web search queries, most important first, max 8"
    )


class BaseRateAnalysis(BaseModel):
    reference_class: str
    historical_base_rate: str
    reasoning: str
    update_direction: str = Field(
        description="How current evidence should move the prior: up, down, or unclear"
    )


@dataclass
class QuestionContext:
    """Cached per-question analysis used across research and forecasting."""

    resolution_analysis: str
    decomposed_queries: list[str]
    base_rate_analysis: str
    time_horizon: str
    difficulty: str
    community_anchor: str
    type_guidance: str


@dataclass
class TavilyBudget:
    """Conservative Tavily credit guard for free tier."""

    max_per_question: int = 2
    max_per_run: int = 40
    searches_used: int = 0
    _cache: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> TavilyBudget:
        return cls(
            max_per_question=int(os.getenv("TAVILY_MAX_SEARCHES_PER_QUESTION", "2")),
            max_per_run=int(os.getenv("TAVILY_MAX_SEARCHES_PER_RUN", "40")),
        )

    def _lock_for(self, question_key: str) -> asyncio.Lock:
        if question_key not in self._locks:
            self._locks[question_key] = asyncio.Lock()
        return self._locks[question_key]

    async def cached_or_fetch(
        self, question_key: str, fetch: Any
    ) -> str:
        """Deduplicate concurrent research fetches for the same question."""
        cached = self.get_cached(question_key)
        if cached:
            return cached
        async with self._lock_for(question_key):
            cached = self.get_cached(question_key)
            if cached:
                return cached
            result = await fetch()
            self.set_cache(question_key, result)
            return result

    def remaining_for_question(self, question_key: str) -> int:
        if question_key in self._cache:
            return 0
        run_remaining = self.max_per_run - self.searches_used
        return max(0, min(self.max_per_question, run_remaining))

    def record_search(self, question_key: str) -> None:
        self.searches_used += 1

    def get_cached(self, question_key: str) -> str | None:
        return self._cache.get(question_key)

    def set_cache(self, question_key: str, research: str) -> None:
        self._cache[question_key] = research


class CalibrationLogger:
    """Append-only log for post-hoc calibration analysis."""

    def __init__(self, log_path: str | Path | None = None) -> None:
        default_path = os.getenv("CALIBRATION_LOG_PATH", "calibration_log.jsonl")
        self.log_path = Path(log_path or default_path)

    def log_forecast(
        self,
        *,
        question: MetaculusQuestion,
        prediction: Any,
        community_prediction: Any | None,
        question_type: str,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question_id": question.id_of_question,
            "post_id": question.id_of_post,
            "page_url": question.page_url,
            "question_type": question_type,
            "question_text": question.question_text[:500],
            "prediction": self._serialize_prediction(prediction),
            "community_prediction": community_prediction,
            "close_time": (
                question.close_time.isoformat() if question.close_time else None
            ),
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
        logger.info("Calibration log appended for %s", question.page_url)

    @staticmethod
    def _serialize_prediction(prediction: Any) -> Any:
        if isinstance(prediction, float):
            return round(prediction, 4)
        if isinstance(prediction, NumericDistribution):
            return {
                "percentiles": [
                    {"p": p.percentile, "v": p.value}
                    for p in prediction.declared_percentiles
                ]
            }
        if isinstance(prediction, PredictedOptionList):
            return [
                {"option": o.option_name, "prob": o.probability}
                for o in prediction.predicted_options
            ]
        return str(prediction)


def question_cache_key(question: MetaculusQuestion) -> str:
    return str(question.id_of_question or question.page_url or question.question_text)


def compute_time_horizon_brief(question: MetaculusQuestion) -> str:
    now = datetime.now(timezone.utc)
    close = question.close_time
    if close is None:
        return clean_indents(
            """
            Time horizon: unknown close date — assume medium horizon; lean slightly
            toward base rates and status quo; do not over-weight breaking news.
            """
        )

    if close.tzinfo is None:
        close = close.replace(tzinfo=timezone.utc)
    delta = close - now
    days = max(delta.total_seconds() / 86400, 0)

    if days <= 14:
        horizon = "short"
        guidance = (
            "Weight very recent news heavily. Status-quo bias should be weaker. "
            "Small probability shifts from new developments are justified."
        )
    elif days <= 90:
        horizon = "medium"
        guidance = (
            "Balance recent news with base rates. Status quo matters but trends can matter."
        )
    elif days <= 365:
        horizon = "long"
        guidance = (
            "Strong status-quo / base-rate bias. Recent news is a modest update, not a verdict."
        )
    else:
        horizon = "very_long"
        guidance = (
            "Forecast almost entirely from reference-class base rates. "
            "Current headlines are weak evidence unless they change structural drivers."
        )

    return clean_indents(
        f"""
        Time horizon: {horizon} ({days:.0f} days until close on {close.date().isoformat()}).
        Calibration guidance: {guidance}
        """
    )


def community_anchor_brief(question: MetaculusQuestion) -> str:
    community = getattr(question, "community_prediction_at_access_time", None)
    if community is None:
        return "Community anchor: no community prediction available — rely on research and base rates."

    if isinstance(question, BinaryQuestion):
        return clean_indents(
            f"""
            Community anchor: Metaculus crowd currently at {community:.1%} Yes.
            Treat this as a weak prior to sanity-check your forecast, not as ground truth.
            Deviate only with specific cited evidence from research.
            """
        )
    return (
        f"Community anchor: crowd prediction available ({community}). "
        "Use as weak sanity check only."
    )


def question_type_guidance(question: MetaculusQuestion) -> str:
    if isinstance(question, BinaryQuestion):
        return clean_indents(
            """
            Binary-specific guidance:
            - Check prediction markets and expert consensus if mentioned in research.
            - Start from base rate, then update with question-specific evidence.
            - Avoid extreme probabilities (<5% or >95%) without ironclad resolution evidence.
            - Explicitly compare Yes vs No scenarios against resolution criteria.
            """
        )
    if isinstance(question, MultipleChoiceQuestion):
        return clean_indents(
            f"""
            Multiple-choice guidance:
            - Assign non-trivial probability to most options: {question.options}
            - Status quo / most likely option should get meaningful weight.
            - Reserve 5–15% total for surprise outcomes across minor options.
            """
        )
    if isinstance(question, (NumericQuestion, DateQuestion)):
        log_note = ""
        if isinstance(question, NumericQuestion) and question.zero_point is not None:
            log_note = " This is log-scaled — respect the zero point and widen the low tail."
        return clean_indents(
            f"""
            Numeric/date guidance:
            - Research the current value and recent trend before forecasting.
            - Use wide 10th–90th percentile spreads; humility on unknown unknowns.
            - Respect question bounds and units exactly.{log_note}
            """
        )
    return ""


async def parse_resolution_criteria(
    question: MetaculusQuestion, llm: GeneralLlm
) -> str:
    prompt = clean_indents(
        f"""
        Parse this forecasting question's resolution criteria precisely.

        Question: {question.question_text}
        Resolution criteria: {question.resolution_criteria}
        Fine print: {question.fine_print}

        List Yes conditions, No conditions, ambiguities, and key definitions.
        Be literal — quote ambiguous phrases. Do not forecast.
        """
    )
    reasoning = await llm.invoke(prompt)
    parsed: ResolutionAnalysis = await structure_output(
        reasoning,
        ResolutionAnalysis,
        model=llm,
        num_validation_samples=1,
    )
    return clean_indents(
        f"""
        ## Resolution analysis
        **Yes if:** {"; ".join(parsed.yes_conditions) or "see criteria"}
        **No if:** {"; ".join(parsed.no_conditions) or "see criteria"}
        **Ambiguities:** {"; ".join(parsed.ambiguities) or "none noted"}
        **Key definitions:** {"; ".join(parsed.key_definitions) or "none noted"}
        """
    )


async def decompose_question(
    question: MetaculusQuestion, llm: GeneralLlm, resolution_analysis: str
) -> list[str]:
    prompt = clean_indents(
        f"""
        Decompose this forecast question into 5–8 prioritized web search queries.

        Cover: status quo, Yes triggers, No triggers, expert/market views, recent news.
        Most important query first. Each query should be concise (under 15 words).

        Question: {question.question_text}
        {resolution_analysis}
        """
    )
    reasoning = await llm.invoke(prompt)
    decomposed: DecomposedQueries = await structure_output(
        reasoning, DecomposedQueries, model=llm, num_validation_samples=1
    )
    return decomposed.queries[:8]


async def estimate_base_rate(
    question: MetaculusQuestion, llm: GeneralLlm, resolution_analysis: str
) -> str:
    """Vultr-only reference class analysis (no extra Tavily search)."""
    prompt = clean_indents(
        f"""
        You are a superforecaster estimating a reference-class base rate.

        Using general knowledge (no web search), identify the best reference class
        for this question and estimate how often similar events resolve Yes.

        Do NOT give a final forecast for this specific question — only the outside view.

        Question: {question.question_text}
        {resolution_analysis}

        Provide: reference class name, estimated historical frequency (qualitative or %),
        reasoning, and whether current-specific evidence likely pushes up/down/unclear.
        """
    )
    reasoning = await llm.invoke(prompt)
    parsed: BaseRateAnalysis = await structure_output(
        reasoning, BaseRateAnalysis, model=llm, num_validation_samples=1
    )
    return clean_indents(
        f"""
        ## Base rate (reference class)
        **Reference class:** {parsed.reference_class}
        **Historical frequency:** {parsed.historical_base_rate}
        **Reasoning:** {parsed.reasoning}
        **Update direction from specifics:** {parsed.update_direction}
        """
    )


async def build_question_context(
    question: MetaculusQuestion,
    fast_llm: GeneralLlm,
) -> QuestionContext:
    resolution = await parse_resolution_criteria(question, fast_llm)
    queries, base_rate = await asyncio.gather(
        decompose_question(question, fast_llm, resolution),
        estimate_base_rate(question, fast_llm, resolution),
    )
    difficulty = _assess_difficulty(question, resolution)
    return QuestionContext(
        resolution_analysis=resolution,
        decomposed_queries=queries,
        base_rate_analysis=base_rate,
        time_horizon=compute_time_horizon_brief(question),
        difficulty=difficulty,
        community_anchor=community_anchor_brief(question),
        type_guidance=question_type_guidance(question),
    )


def _assess_difficulty(question: MetaculusQuestion, resolution_analysis: str) -> str:
    text = (
        question.question_text + resolution_analysis + (question.fine_print or "")
    ).lower()
    hard_signals = (
        "conditional",
        "ambiguous",
        "multiple",
        "complex",
        "geopolit",
        "regulation",
    )
    if any(signal in text for signal in hard_signals):
        return "hard"
    return "standard"


def enrich_research_with_context(research: str, context: QuestionContext) -> str:
    return clean_indents(
        f"""
        {context.resolution_analysis}

        {context.base_rate_analysis}

        {context.time_horizon}

        {context.community_anchor}

        ## Web research
        {research}
        """
    )


def build_forecast_preamble(context: QuestionContext) -> str:
    return clean_indents(
        f"""
        {context.resolution_analysis}
        {context.base_rate_analysis}
        {context.time_horizon}
        {context.community_anchor}
        {context.type_guidance}
        {research_grounding_instructions()}
        """
    )


def research_grounding_instructions() -> str:
    return clean_indents(
        """
        Grounding rules:
        - Start from the base rate, then update with cited research facts.
        - Cite source numbers or URLs from the research report for key claims.
        - If research is thin or contradictory, widen uncertainty.
        - Do not invent news, polls, or events unsupported by research.
        """
    )


def trim_outliers_float(values: Sequence[float], trim_fraction: float = 0.2) -> list[float]:
    if len(values) <= 2:
        return list(values)
    sorted_vals = sorted(values)
    n_trim = max(1, int(len(sorted_vals) * trim_fraction))
    if len(sorted_vals) <= 2 * n_trim:
        return sorted_vals
    return sorted_vals[n_trim:-n_trim]


def aggregate_binary_trimmed(predictions: list[float]) -> float:
    trimmed = trim_outliers_float(predictions)
    return float(statistics.median(trimmed))


def _distribution_center(dist: NumericDistribution) -> float:
    percentiles = dist.declared_percentiles
    if not percentiles:
        cdf = dist.get_cdf()
        mid = len(cdf) // 2
        return cdf[mid].value
    for target in (0.5, 0.4, 0.6):
        for p in percentiles:
            if abs(p.percentile - target) < 0.15:
                return p.value
    return percentiles[len(percentiles) // 2].value


def aggregate_numeric_trimmed(
    predictions: list[NumericDistribution],
    question: NumericQuestion | DateQuestion,
) -> NumericDistribution:
    if len(predictions) == 1:
        return predictions[0]
    centers = [_distribution_center(p) for p in predictions]
    median_center = statistics.median(centers)
    deviations = [abs(c - median_center) for c in centers]
    cutoff = statistics.median(deviations) * 2.5 + 1e-9
    kept = [
        p
        for p, dev in zip(predictions, deviations)
        if dev <= cutoff or len(predictions) <= 3
    ]
    if not kept:
        kept = predictions
    cdfs = [p.get_cdf() for p in kept]
    x_axis = [pt.value for pt in cdfs[0]]
    all_percentiles = [[pt.percentile for pt in cdf] for cdf in cdfs]
    median_percentiles = np.median(np.array(all_percentiles), axis=0).tolist()
    median_cdf = [
        Percentile(value=v, percentile=p) for v, p in zip(x_axis, median_percentiles)
    ]
    return NumericDistribution.from_question(median_cdf, question)


def aggregate_multiple_choice_trimmed(
    predictions: list[PredictedOptionList],
) -> PredictedOptionList:
    if not predictions:
        raise ValueError("No predictions")
    option_names = [o.option_name for o in predictions[0].predicted_options]
    trimmed_preds = predictions
    if len(predictions) >= 4:
        # Drop the forecast farthest from mean vector
        vectors = []
        for pred in predictions:
            vectors.append(
                [o.probability for o in pred.predicted_options]
            )
        mean_vec = np.mean(vectors, axis=0)
        distances = [float(np.linalg.norm(v - mean_vec)) for v in vectors]
        worst_idx = int(np.argmax(distances))
        trimmed_preds = [p for i, p in enumerate(predictions) if i != worst_idx]

    averaged: dict[str, float] = {name: 0.0 for name in option_names}
    for pred in trimmed_preds:
        for opt in pred.predicted_options:
            averaged[opt.option_name] += opt.probability
    n = len(trimmed_preds)
    options = [
        PredictedOption(option_name=name, probability=prob / n)
        for name, prob in averaged.items()
    ]
    total = sum(o.probability for o in options)
    if total > 0:
        options = [
            PredictedOption(option_name=o.option_name, probability=o.probability / total)
            for o in options
        ]
    return PredictedOptionList(predicted_options=options)

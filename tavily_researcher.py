"""
Tavily-backed research for the forecasting bot (free-tier conservative).

Uses basic search depth, strict per-question/run budgets, in-memory caching,
decomposition-driven queries, and parallel search execution.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from forecasting_tools import GeneralLlm, MetaculusQuestion, clean_indents
from tavily import AsyncTavilyClient

from bot_strategy import QuestionContext, TavilyBudget, question_cache_key

logger = logging.getLogger(__name__)

VULTR_INFERENCE_BASE_URL = "https://api.vultrinference.com/v1"

# Free tier: basic depth only, minimal results, no raw content extraction.
TAVILY_SEARCH_DEPTH = "basic"
TAVILY_MAX_RESULTS = 4

# Cap concurrent Tavily HTTP calls (parallel within a question, bounded globally).
_tavily_http_semaphore: asyncio.Semaphore | None = None


def _get_tavily_semaphore() -> asyncio.Semaphore:
    global _tavily_http_semaphore
    if _tavily_http_semaphore is None:
        limit = int(os.getenv("TAVILY_MAX_CONCURRENT_SEARCHES", "3"))
        _tavily_http_semaphore = asyncio.Semaphore(limit)
    return _tavily_http_semaphore


def make_vultr_llm(
    *,
    model: str | None = None,
    temperature: float = 0.3,
    timeout: float = 90,
    allowed_tries: int = 2,
) -> GeneralLlm:
    """Build a GeneralLlm pointed at Vultr Serverless Inference."""
    api_key = os.getenv("VULTR_SERVERLESS_INFERENCE_API_KEY")
    if not api_key or api_key.strip() in {
        "REPLACE_ME",
        "your-api-key-here",
    }:
        raise ValueError(
            "VULTR_SERVERLESS_INFERENCE_API_KEY is missing or still a placeholder."
        )

    resolved_model = model or os.getenv(
        "VULTR_INFERENCE_MODEL", "meta-llama/Llama-3.3-70B-Instruct"
    )
    litellm_model = (
        resolved_model
        if resolved_model.startswith("openai/")
        else f"openai/{resolved_model}"
    )

    return GeneralLlm(
        model=litellm_model,
        api_key=api_key,
        base_url=VULTR_INFERENCE_BASE_URL,
        temperature=temperature,
        timeout=timeout,
        allowed_tries=allowed_tries,
    )


def _format_search_result_block(
    label: str, response: dict[str, Any], *, max_snippet_chars: int = 600
) -> str:
    lines = [f"### {label}"]
    answer = response.get("answer")
    if answer:
        lines.append(f"Summary: {answer}")

    results = response.get("results") or []
    if not results:
        lines.append("_No results returned._")
        return "\n".join(lines)

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or "unknown"
        published = result.get("published_date") or "unknown date"
        content = (result.get("content") or "").strip()
        if len(content) > max_snippet_chars:
            content = content[:max_snippet_chars] + "…"

        lines.append(
            clean_indents(
                f"""
                [{index}] **{title}**
                - URL: {url}
                - Published: {published}
                - Excerpt: {content or "(no excerpt)"}
                """
            )
        )

    return "\n".join(lines)


async def _run_budgeted_search(
    client: AsyncTavilyClient,
    query: str,
    *,
    topic: str = "general",
    time_range: str | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "search_depth": TAVILY_SEARCH_DEPTH,
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": "basic",
    }
    if topic:
        kwargs["topic"] = topic
    if time_range:
        kwargs["time_range"] = time_range
    async with _get_tavily_semaphore():
        return await client.search(query, **kwargs)


def _query_search_params(query: str) -> tuple[str, str | None]:
    topic = "news" if any(
        w in query.lower() for w in ("news", "latest", "recent", "today")
    ) else "general"
    time_range = "week" if topic == "news" else None
    return topic, time_range


async def _fetch_tavily_source_pack(
    question: MetaculusQuestion,
    context: QuestionContext,
    budget: TavilyBudget,
) -> str:
    """Execute parallel Tavily basic searches for decomposed queries."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.strip() in {"REPLACE_ME", "your-api-key-here"}:
        raise ValueError("TAVILY_API_KEY is missing or still a placeholder.")

    cache_key = question_cache_key(question)
    allowed = budget.remaining_for_question(cache_key)
    if allowed <= 0:
        return "_Tavily budget exhausted; rely on base-rate and resolution analysis._"

    client = AsyncTavilyClient(api_key=api_key)
    queries = context.decomposed_queries[:allowed]
    if not queries:
        queries = [question.question_text[:120]]

    run_slots = budget.max_per_run - budget.searches_used
    queries = queries[: max(0, run_slots)]
    if not queries:
        return "_Tavily run budget exhausted._"

    async def run_one(query: str) -> tuple[str, dict[str, Any] | BaseException]:
        topic, time_range = _query_search_params(query)
        try:
            result = await _run_budgeted_search(
                client, query, topic=topic, time_range=time_range
            )
            return query, result
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)
            return query, exc

    logger.info(
        "Running %s Tavily searches in parallel for %s",
        len(queries),
        question.page_url,
    )
    pairs = await asyncio.gather(*(run_one(q) for q in queries))

    searches: list[tuple[str, dict[str, Any]]] = []
    for query, result in pairs:
        if isinstance(result, dict):
            budget.record_search(cache_key)
            searches.append((query, result))
    if not searches:
        return "_No Tavily results (budget or API error)._"

    sections = [
        _format_search_result_block(f"Search: {query[:80]}", resp)
        for query, resp in searches
    ]
    return "\n\n".join(sections)


async def gather_tavily_sources(
    question: MetaculusQuestion,
    context: QuestionContext,
    budget: TavilyBudget,
) -> str:
    """Run parallel Tavily searches with per-question deduplication."""
    cache_key = question_cache_key(question)
    return await budget.cached_or_fetch(
        cache_key,
        lambda: _fetch_tavily_source_pack(question, context, budget),
    )


async def synthesize_grounded_research(
    question: MetaculusQuestion,
    source_pack: str,
    context: QuestionContext,
    summarizer: GeneralLlm,
) -> str:
    """Turn Tavily hits + strategic context into a citation-preserving brief."""
    prompt = clean_indents(
        f"""
        You are a research assistant to a professional forecaster.
        Using ONLY the sourced material and context below, write a concise research brief.

        Rules:
        - Do NOT forecast probabilities.
        - Every important claim must cite [n] from the web sources.
        - Copy URLs verbatim in a Sources section.
        - Integrate base-rate context with web findings; note conflicts.
        - Label uncertainty and research gaps.

        {context.resolution_analysis}
        {context.base_rate_analysis}

        Question: {question.question_text}
        Resolution criteria: {question.resolution_criteria}
        {question.fine_print}

        Web sources:
        {source_pack}

        Format:
        ## Key facts (with [n] citations)
        ## Resolution-relevant signals
        ## Prediction markets (Kalshi, Polymarket — exact prices if found)
        ## Base rate vs current evidence
        ## Gaps / unknowns
        ## Sources (URL list)
        """
    )
    return await summarizer.invoke(prompt)


async def run_tavily_research(
    question: MetaculusQuestion,
    context: QuestionContext,
    budget: TavilyBudget,
    summarizer: GeneralLlm | None = None,
) -> str:
    """End-to-end conservative Tavily research for one question."""
    source_pack = await gather_tavily_sources(question, context, budget)
    if summarizer is None or source_pack.startswith("_"):
        return source_pack

    try:
        return await synthesize_grounded_research(
            question, source_pack, context, summarizer
        )
    except Exception as exc:
        logger.warning("Tavily synthesis failed (%s); returning source pack.", exc)
        return source_pack

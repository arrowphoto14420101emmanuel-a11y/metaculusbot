"""
Tavily-backed research for the forecasting bot (free-tier conservative).

Uses basic search depth, strict per-question/run budgets, in-memory caching,
and decomposition-driven queries so each credit counts.
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
    return await client.search(query, **kwargs)


async def gather_tavily_sources(
    question: MetaculusQuestion,
    context: QuestionContext,
    budget: TavilyBudget,
) -> str:
    """Run a minimal set of Tavily basic searches driven by decomposition."""
    cache_key = question_cache_key(question)
    cached = budget.get_cached(cache_key)
    if cached:
        return cached

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.strip() in {"REPLACE_ME", "your-api-key-here"}:
        raise ValueError("TAVILY_API_KEY is missing or still a placeholder.")

    allowed = budget.remaining_for_question(cache_key)
    if allowed <= 0:
        return "_Tavily budget exhausted; rely on base-rate and resolution analysis._"

    client = AsyncTavilyClient(api_key=api_key)
    queries = context.decomposed_queries[:allowed]
    if not queries:
        queries = [question.question_text[:120]]

    searches: list[dict[str, Any]] = []
    for query in queries:
        if budget.searches_used >= budget.max_per_run:
            logger.warning("Tavily run budget exhausted at %s searches", budget.searches_used)
            break
        topic = "news" if any(
            w in query.lower() for w in ("news", "latest", "recent", "today")
        ) else "general"
        time_range = "week" if topic == "news" else None
        try:
            result = await _run_budgeted_search(
                client, query, topic=topic, time_range=time_range
            )
            searches.append(result)
            budget.record_search(cache_key)
        except Exception as exc:
            logger.warning("Tavily search failed for %r: %s", query, exc)

    if not searches:
        return "_No Tavily results (budget or API error)._"

    sections = [
        _format_search_result_block(f"Search: {queries[i][:80]}", resp)
        for i, resp in enumerate(searches)
    ]
    source_pack = "\n\n".join(sections)
    budget.set_cache(cache_key, source_pack)
    return source_pack


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

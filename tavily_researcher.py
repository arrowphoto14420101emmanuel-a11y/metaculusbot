"""
Tavily-backed research for the forecasting bot.

Fetches web sources via Tavily, formats them with explicit citations, then
optionally synthesizes a grounded brief with an LLM so forecasts can reference
real URLs and facts.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from forecasting_tools import GeneralLlm, MetaculusQuestion, clean_indents
from tavily import AsyncTavilyClient

logger = logging.getLogger(__name__)

VULTR_INFERENCE_BASE_URL = "https://api.vultrinference.com/v1"


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
    # LiteLLM routes OpenAI-compatible endpoints via the openai/ prefix.
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
    label: str, response: dict[str, Any], *, max_snippet_chars: int = 1200
) -> str:
    lines = [f"### {label}"]
    answer = response.get("answer")
    if answer:
        lines.append(f"Tavily answer: {answer}")

    results = response.get("results") or []
    if not results:
        lines.append("_No results returned._")
        return "\n".join(lines)

    for index, result in enumerate(results, start=1):
        title = result.get("title") or "Untitled"
        url = result.get("url") or "unknown"
        published = result.get("published_date") or "unknown date"
        score = result.get("score")
        score_text = f", relevance={score:.2f}" if isinstance(score, (int, float)) else ""
        content = (result.get("content") or "").strip()
        raw_content = (result.get("raw_content") or "").strip()
        snippet = raw_content or content
        if len(snippet) > max_snippet_chars:
            snippet = snippet[:max_snippet_chars] + "…"

        lines.append(
            clean_indents(
                f"""
                [{index}] **{title}**
                - URL: {url}
                - Published: {published}{score_text}
                - Excerpt: {snippet or "(no excerpt)"}
                """
            )
        )

    return "\n".join(lines)


async def gather_tavily_sources(question: MetaculusQuestion) -> str:
    """Run multiple Tavily searches and return a citation-rich source pack."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.strip() in {"REPLACE_ME", "your-api-key-here"}:
        raise ValueError("TAVILY_API_KEY is missing or still a placeholder.")

    client = AsyncTavilyClient(api_key=api_key)
    core_query = clean_indents(
        f"""
        {question.question_text}

        Resolution criteria:
        {question.resolution_criteria}

        {question.fine_print}
        """
    )
    news_query = f"latest news updates: {question.question_text}"
    resolution_query = (
        f"evidence for or against resolution: {question.resolution_criteria}"
    )

    news_search, general_search, resolution_search = await asyncio.gather(
        client.search(
            news_query,
            search_depth="advanced",
            topic="news",
            time_range="month",
            max_results=6,
            include_raw_content="markdown",
        ),
        client.search(
            core_query,
            search_depth="advanced",
            topic="general",
            max_results=8,
            include_raw_content="markdown",
        ),
        client.search(
            resolution_query,
            search_depth="advanced",
            topic="general",
            max_results=5,
            include_answer="basic",
            include_raw_content="markdown",
        ),
    )

    sections = [
        _format_search_result_block("Latest news", news_search),
        _format_search_result_block("Background & context", general_search),
        _format_search_result_block("Resolution-relevant evidence", resolution_search),
    ]
    return "\n\n".join(sections)


async def synthesize_grounded_research(
    question: MetaculusQuestion,
    source_pack: str,
    summarizer: GeneralLlm,
) -> str:
    """Turn raw Tavily hits into a structured, citation-preserving research brief."""
    prompt = clean_indents(
        f"""
        You are a research assistant to a professional forecaster.
        Using ONLY the sourced material below, write a concise but detailed research brief.

        Rules:
        - Do NOT forecast probabilities or give a final Yes/No prediction.
        - Every important claim must cite its source as [n] matching the numbered sources.
        - Copy source URLs verbatim in a "Sources" section at the end.
        - Separate facts from speculation; label uncertainty explicitly.
        - Note whether current public information would lean toward Yes, No, or is unclear
          under the resolution criteria — but do not assign a probability.
        - If sources disagree, report the disagreement.

        Question:
        {question.question_text}

        Resolution criteria:
        {question.resolution_criteria}

        {question.fine_print}

        Sourced material:
        {source_pack}

        Format:
        ## Key facts (with [n] citations)
        ## Resolution-relevant signals
        ## Gaps / unknowns
        ## Sources (URL list)
        """
    )
    return await summarizer.invoke(prompt)


async def run_tavily_research(
    question: MetaculusQuestion,
    summarizer: GeneralLlm | None = None,
) -> str:
    """End-to-end Tavily research: search → synthesize → return grounded brief."""
    source_pack = await gather_tavily_sources(question)
    if summarizer is None:
        return source_pack

    try:
        return await synthesize_grounded_research(question, source_pack, summarizer)
    except Exception as exc:
        logger.warning(
            "Tavily synthesis failed (%s); returning raw source pack.", exc
        )
        return source_pack

"""Tests for parallel research behavior (no API keys required)."""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from bot_strategy import QuestionContext, TavilyBudget
from tavily_researcher import _fetch_tavily_source_pack


def _fake_question() -> MagicMock:
    q = MagicMock()
    q.question_text = "Will X happen by 2027?"
    q.resolution_criteria = "Resolves Yes if X happens."
    q.fine_print = ""
    q.page_url = "https://example.com/q/1"
    q.id_of_question = 1
    return q


def _fake_context() -> QuestionContext:
    return QuestionContext(
        resolution_analysis="## Resolution",
        decomposed_queries=["query one", "query two"],
        base_rate_analysis="## Base rate",
        time_horizon="short",
        difficulty="standard",
        community_anchor="none",
        type_guidance="binary",
    )


class ParallelTavilyTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_searches_use_gather(self) -> None:
        budget = TavilyBudget(max_per_question=2, max_per_run=10)
        question = _fake_question()
        context = _fake_context()
        call_times: list[float] = []

        async def fake_search(*_args, **_kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return {"results": [{"title": "t", "url": "u", "content": "c"}]}

        with patch.dict("os.environ", {"TAVILY_API_KEY": "tvly-test"}):
            with patch("tavily_researcher.AsyncTavilyClient") as client_cls:
                client = MagicMock()
                client.search = AsyncMock(side_effect=fake_search)
                client_cls.return_value = client

                await _fetch_tavily_source_pack(question, context, budget)

        self.assertEqual(len(call_times), 2)
        # Parallel: both searches start close together (< sequential 0.1s gap)
        self.assertLess(call_times[1] - call_times[0], 0.08)
        self.assertEqual(budget.searches_used, 2)

    async def test_cached_or_fetch_deduplicates_parallel_calls(self) -> None:
        budget = TavilyBudget(max_per_question=2, max_per_run=10)
        calls = 0

        async def slow_fetch() -> str:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return "research"

        results = await asyncio.gather(
            budget.cached_or_fetch("q1", slow_fetch),
            budget.cached_or_fetch("q1", slow_fetch),
        )
        self.assertEqual(results, ["research", "research"])
        self.assertEqual(calls, 1)


if __name__ == "__main__":
    unittest.main()

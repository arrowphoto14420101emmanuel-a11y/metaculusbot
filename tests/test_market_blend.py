"""Tests for median + market crowd blending."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from bot_strategy import MarketSignals, aggregate_binary_trimmed, aggregate_binary_with_markets


class MarketBlendTests(unittest.TestCase):
    def test_trimmed_median_is_core(self) -> None:
        median = aggregate_binary_trimmed([0.2, 0.25, 0.8, 0.82])
        self.assertAlmostEqual(median, 0.525, places=2)

    def test_blend_includes_kalshi_polymarket_metaculus(self) -> None:
        question = MagicMock()
        question.community_prediction_at_access_time = 0.60
        type(question).__name__ = "BinaryQuestion"
        # isinstance check needs BinaryQuestion - use real import
        from forecasting_tools import BinaryQuestion

        question = MagicMock(spec=BinaryQuestion)
        question.community_prediction_at_access_time = 0.60

        signals = MarketSignals(kalshi_yes=0.55, polymarket_yes=0.50)
        blended = aggregate_binary_with_markets([0.40, 0.42, 0.38], question, signals)
        # Should be between model ~0.40 and markets ~0.55-0.60
        self.assertGreater(blended, 0.40)
        self.assertLess(blended, 0.60)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from rag.highlight import highlight_key_spans
from rag.metrics import MetricFact


class HighlightKeySpansTests(unittest.TestCase):
    def test_bolds_metric_value_already_in_text(self) -> None:
        fact = MetricFact(
            "projects/asthma-air-pollution/v2/outputs/feature_analysis.json",
            "$.n_counties",
            67,
        )
        text = "The Version 2 analysis covers 67 Alabama counties."
        out = highlight_key_spans(text, metric_facts=[fact])
        self.assertIn("**67**", out)
        self.assertNotIn("****", out)

    def test_does_not_invent_missing_numbers(self) -> None:
        fact = MetricFact(
            "projects/asthma-air-pollution/v2/outputs/feature_analysis.json",
            "$.n_counties",
            67,
        )
        text = "Sample size is reported in the verified metric block."
        out = highlight_key_spans(text, metric_facts=[fact])
        self.assertEqual(out, text)
        self.assertNotIn("67", out)

    def test_avoids_double_bold_and_preserves_code(self) -> None:
        fact = MetricFact(
            "projects/asthma-air-pollution/v2/outputs/metrics.json",
            "$.pearson_r_pm25_asthma",
            -0.05726740504810031,
        )
        value = fact.rendered_value
        text = (
            f"Verified: **{value}** and inline `{value}` stay intact.\n"
            f"```\n{value}\n```\n"
            f"Also in prose: {value}"
        )
        out = highlight_key_spans(text, metric_facts=[fact])
        self.assertIn(f"**{value}**", out)
        self.assertNotIn("****", out)
        self.assertIn(f"`{value}`", out)
        self.assertIn(f"```\n{value}\n```", out)
        # Prose occurrence should be bolded once.
        self.assertIn(f"Also in prose: **{value}**", out)

    def test_does_not_bold_67_inside_1175(self) -> None:
        fact = MetricFact(
            "projects/asthma-air-pollution/v2/outputs/feature_analysis.json",
            "$.n_counties",
            67,
        )
        text = "Version 1 used 1175 census tracts; Version 2 uses 67 counties."
        out = highlight_key_spans(text, metric_facts=[fact])
        self.assertIn("1175", out)
        self.assertNotIn("**1175**", out)
        self.assertIn("**67**", out)

    def test_bolds_value_before_sentence_period(self) -> None:
        fact = MetricFact(
            "projects/asthma-air-pollution/v2/outputs/metrics.json",
            "$.pearson_r_pm25_asthma",
            -0.05726740504810031,
        )
        value = fact.rendered_value
        text = f"The Pearson correlation is {value}."
        out = highlight_key_spans(text, metric_facts=[fact])
        self.assertIn(f"**{value}**.", out)

    def test_light_query_term_emphasis(self) -> None:
        text = "The Pearson correlation between PM2.5 and asthma is weak."
        out = highlight_key_spans(
            text,
            question="What is the Pearson correlation between PM2.5 and asthma?",
        )
        self.assertIn("**Pearson**", out)
        self.assertIn("**PM2.5**", out)
        # Common filler words from the question must not be bolded.
        self.assertNotIn("**What**", out)
        self.assertNotIn("**the**", out)


if __name__ == "__main__":
    unittest.main()

"""Deterministic routing for questions about stored numeric results."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Chunk


@dataclass(frozen=True)
class MetricTarget:
    source_suffix: str
    locator: str


def _normalise(question: str) -> str:
    return re.sub(r"[^a-z0-9²]+", " ", question.lower()).strip()


def targets_for_question(question: str) -> list[MetricTarget]:
    """Return exact JSON leaves for well-defined metric intents."""
    q = _normalise(question)
    mentions_pm25 = "pm2 5" in q or "pm25" in q or "particulate" in q
    targets: list[MetricTarget] = []

    if mentions_pm25 and "pearson" in q and ("asthma" in q or "correlation" in q):
        targets.append(MetricTarget("v2/outputs/metrics.json", "$.pearson_r_pm25_asthma"))
    if mentions_pm25 and ("cross validated" in q or "cross validation" in q or "cv" in q) and ("r2" in q or "r²" in q):
        targets.extend(
            [
                MetricTarget("v2/outputs/multivariate_metrics.json", "$.models[0].r2_cv_mean"),
                MetricTarget("v2/outputs/multivariate_metrics.json", "$.models[0].r2_cv_std"),
            ]
        )
    if mentions_pm25 and ("partial" in q or "adjusted" in q) and "correlation" in q:
        targets.append(
            MetricTarget(
                "v2/outputs/multivariate_metrics.json",
                "$.partial_correlations_vs_asthma.partial_r_pm25_ug_m3_annual_mean_vs_asthma",
            )
        )
    if mentions_pm25 and ("coefficient of variation" in q or "exposure contrast" in q or "spread" in q):
        targets.append(
            MetricTarget("v2/outputs/feature_analysis.json", "$.variance_analysis.pm25_ug_m3_annual_mean.cv")
        )
    if mentions_pm25 and "permutation" in q and ("p value" in q or "significant" in q):
        targets.append(
            MetricTarget(
                "v2/outputs/robustness_report.json",
                "$.pm25_asthma_association.permutation_test.p_value_two_sided",
            )
        )
    if mentions_pm25 and "bootstrap" in q:
        targets.extend(
            [
                MetricTarget(
                    "v2/outputs/robustness_report.json",
                    "$.pm25_asthma_association.bootstrap_95ci.ci_low",
                ),
                MetricTarget(
                    "v2/outputs/robustness_report.json",
                    "$.pm25_asthma_association.bootstrap_95ci.ci_high",
                ),
            ]
        )
    if "obesity" in q and "correlation" in q:
        targets.append(
            MetricTarget(
                "v2/outputs/multivariate_metrics.json",
                "$.pearson_correlation_matrix.asthma_pct.obesity_pct",
            )
        )
    return targets


def exact_metric_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    targets = targets_for_question(question)
    return [
        chunk
        for target in targets
        for chunk in chunks
        if chunk.source.endswith(target.source_suffix) and chunk.locator == target.locator
    ]


def is_numeric_question(question: str) -> bool:
    q = _normalise(question)
    terms = {
        "correlation",
        "r2",
        "r²",
        "metric",
        "mean",
        "range",
        "coefficient",
        "p value",
        "confidence interval",
        "how many",
    }
    return any(term in q for term in terms)

"""Deterministic routing for questions about stored numeric results."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from .corpus import ASTHMA_ROOT, Chunk

StructuralIntent = Literal["n_counties", "tract_count", "not_patients"]
TopicIntent = Literal["predictive_gain", "outcome_definitions"]


@dataclass(frozen=True)
class MetricTarget:
    source_suffix: str
    locator: str


@dataclass(frozen=True)
class MetricFact:
    """An exact JSON value resolved independently of language generation."""

    source: str
    locator: str
    value: Any

    @property
    def citation(self) -> str:
        return f"{self.source} — {self.locator}"

    @property
    def rendered_value(self) -> str:
        return json.dumps(self.value, ensure_ascii=False, sort_keys=True)


def _normalise(question: str) -> str:
    return re.sub(r"[^a-z0-9²]+", " ", question.lower()).strip()


# Synonym families for structural study questions. Order in classify_structural_intent
# matters: patient clarification and tract count take precedence over county N.
_TRACT_TERMS = (
    r"\bcensus\s+tracts?\b",
    r"\btracts?\b",
    r"\bv1\s+rows?\b",
    r"\bversion\s*1\s+rows?\b",
    r"\b1\s*,?\s*175\b",
    r"\b1175\b",
)
_SAMPLE_SIZE_TERMS = (
    r"\bsample\s+size\b",
    r"\bdata\s+points?\b",
    r"\bdatapoints?\b",
    r"\bobservations?\b",
    r"\brows?\b",
    r"\bn_counties\b",
    # Allow adjectives between "many"/"of" and "counties" (e.g. "Alabama counties").
    r"\bhow\s+many\s+(?:\w+\s+){0,3}counties\b",
    r"\bnumber\s+of\s+(?:\w+\s+){0,3}counties\b",
    r"\bcounty\s+count\b",
    r"\bcounties\s+in\s+the\s+(?:study|analysis|dataset|data|version)\b",
    r"\balabama\s+counties\b",
    r"\bwhat(?:\s+is|\s*'?s)?\s+n\b",
    r"\bn\s*=\s*",
)
_PREDICTIVE_GAIN_TERMS = (
    r"\bimprove\s+prediction\b",
    r"\bpredictive\s+(?:value|gain|improvement)\b",
    r"\badds?\s+predictive\b",
    r"\bdid\s+(?:pm2\s*5|pm25)?\s*not\s+improve\b",
    r"\badding\s+pm2\s*5\b",
    r"\badding\s+pm25\b",
    r"\bpm2\s*5\s+improve\b",
    r"\bpm25\s+improve\b",
)
_OUTCOME_DEFINITION_TERMS = (
    r"\bprevalence\b",
    r"\bincidence\b",
    r"\bexacerbat",
)

# Ordered preferred passages (first match wins in answer ranking).
TRACT_PREFERRED: tuple[tuple[str, str], ...] = (
    (
        "projects/asthma-air-pollution/STUDY_FAQ.md",
        "Study FAQ (curated evidence snippets) > Sample size and geographic units",
    ),
    (
        "projects/asthma-air-pollution/v1/README_EXTENDED.md",
        "Version 1 retrospective correction > What I built in 2021",
    ),
    (
        "projects/asthma-air-pollution/v1/README.md",
        "Version 1: original Ironhack capstone > Reproducibility status",
    ),
    (
        "projects/asthma-air-pollution/v2/outputs/robustness_report.json",
        "$.v1_vs_v2_matching.v1_risk",
    ),
    (
        "projects/asthma-air-pollution/v2/VALIDATION.md",
        "Statistical validation and uncertainty > Version 1 comparison",
    ),
)

NOT_PATIENTS_PREFERRED: tuple[tuple[str, str], ...] = (
    (
        "projects/asthma-air-pollution/STUDY_FAQ.md",
        "Study FAQ (curated evidence snippets) > Sample size and geographic units",
    ),
    (
        "projects/asthma-air-pollution/v1/README_EXTENDED.md",
        "Version 1 retrospective correction > What I built in 2021",
    ),
    (
        "projects/asthma-air-pollution/v2/VALIDATION.md",
        "Statistical validation and uncertainty > Version 1 comparison",
    ),
)

PREDICTIVE_GAIN_PREFERRED: tuple[tuple[str, str], ...] = (
    (
        "projects/asthma-air-pollution/STUDY_FAQ.md",
        "Study FAQ (curated evidence snippets) > Does PM2.5 improve prediction after health indicators?",
    ),
    (
        "projects/asthma-air-pollution/v2/VALIDATION.md",
        "Statistical validation and uncertainty > Adjusted association",
    ),
    (
        "projects/asthma-air-pollution/v2/FEATURE_ANALYSIS.md",
        "Feature analysis methods and caveats > Conclusion",
    ),
)

OUTCOME_DEFINITIONS_PREFERRED: tuple[tuple[str, str], ...] = (
    (
        "projects/asthma-air-pollution/STUDY_FAQ.md",
        "Study FAQ (curated evidence snippets) > Prevalence, incidence, and acute exacerbations",
    ),
    (
        "projects/asthma-air-pollution/v2/VALIDATION.md",
        "Statistical validation and uncertainty > Interpretation boundaries",
    ),
)


def _matches_any(question: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, question) for pattern in patterns)


def _ordered_preferred(chunks: list[Chunk], preferred: tuple[tuple[str, str], ...]) -> list[Chunk]:
    """Return preferred passages in declared priority order."""
    by_key = {(chunk.source, chunk.locator): chunk for chunk in chunks}
    return [by_key[key] for key in preferred if key in by_key]


def classify_structural_intent(question: str) -> StructuralIntent | None:
    """Map natural-language structural questions onto application-owned intents."""
    q = _normalise(question)
    if not q:
        return None

    # Patient/participant wording must clarify ecological units, not answer with N=67.
    # Causal "prove/cause … individuals" questions keep the separate safety-chunk path.
    causal = bool(re.search(r"\b(caus\w*|causes|prove)\b", q))
    if not causal and (
        _matches_any(q, (r"\bpatients?\b", r"\bparticipants?\b"))
        or (
            _matches_any(q, (r"\bpeople\b", r"\bindividuals?\b", r"\bsubjects?\b"))
            and _matches_any(q, (r"\bhow\s+many\b", r"\bnumber\s+of\b", r"\bcount\b", r"\bsample\b"))
        )
    ):
        return "not_patients"

    if _matches_any(q, _TRACT_TERMS):
        return "tract_count"

    if _matches_any(q, _SAMPLE_SIZE_TERMS):
        return "n_counties"

    return None


def classify_topic_intent(question: str) -> TopicIntent | None:
    """Map FAQ-style interpretation questions onto curated topic passages."""
    q = _normalise(question)
    if not q:
        return None
    if _matches_any(q, _PREDICTIVE_GAIN_TERMS):
        return "predictive_gain"
    # Require at least two outcome vocabulary hits so metric questions stay unaffected.
    outcome_hits = sum(1 for pattern in _OUTCOME_DEFINITION_TERMS if re.search(pattern, q))
    if outcome_hits >= 2:
        return "outcome_definitions"
    return None


def synonym_examples_for_intent(intent: StructuralIntent) -> tuple[str, ...]:
    """Documented examples used by tests and guardrail notes."""
    if intent == "n_counties":
        return (
            "How many data points are in the study?",
            "What is the sample size?",
            "How many observations?",
            "How many rows?",
            "How many counties?",
            "How many Alabama counties are in the Version 2 analysis?",
            "What is n?",
        )
    if intent == "tract_count":
        return (
            "How many census tracts?",
            "How many tracts in Version 1?",
            "What are the v1 rows?",
        )
    return (
        "How many participants?",
        "How many patients are in the study?",
    )


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
    if classify_structural_intent(question) == "n_counties":
        targets.append(MetricTarget("v2/outputs/feature_analysis.json", "$.n_counties"))
    return targets


def exact_metric_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    targets = targets_for_question(question)
    return [
        chunk
        for target in targets
        for chunk in chunks
        if chunk.source.endswith(target.source_suffix) and chunk.locator == target.locator
    ]


def structural_preferred_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Boost documented passages for tract-count and not-patient clarifications."""
    intent = classify_structural_intent(question)
    if intent == "tract_count":
        return _ordered_preferred(chunks, TRACT_PREFERRED)
    if intent == "not_patients":
        return _ordered_preferred(chunks, NOT_PATIENTS_PREFERRED)
    return []


def topic_preferred_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Boost curated FAQ / methods passages for common interpretation questions."""
    intent = classify_topic_intent(question)
    if intent == "predictive_gain":
        return _ordered_preferred(chunks, PREDICTIVE_GAIN_PREFERRED)
    if intent == "outcome_definitions":
        return _ordered_preferred(chunks, OUTCOME_DEFINITIONS_PREFERRED)
    return []


def routed_evidence_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Union of deterministic metric leaves and structural/topic preferred passages."""
    seen: set[tuple[str, str]] = set()
    ordered: list[Chunk] = []
    for chunk in (
        exact_metric_chunks(question, chunks)
        + structural_preferred_chunks(question, chunks)
        + topic_preferred_chunks(question, chunks)
    ):
        key = (chunk.source, chunk.locator)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(chunk)
    return ordered


def _value_at_path(data: Any, locator: str) -> Any:
    """Resolve the restricted JSONPath syntax emitted by ``corpus.py``."""
    value = data
    tokens = re.findall(r"(?:^|\.)?([A-Za-z0-9_]+)|\[(\d+)\]", locator.removeprefix("$"))
    for key, index in tokens:
        value = value[int(index)] if index else value[key]
    return value


def exact_metric_facts(question: str, chunks: list[Chunk]) -> list[MetricFact]:
    """Load recognised metrics from their allowlisted JSON files by key path."""
    facts: list[MetricFact] = []
    for chunk in exact_metric_chunks(question, chunks):
        relative = chunk.source.removeprefix("projects/asthma-air-pollution/")
        data = json.loads((ASTHMA_ROOT / relative).read_text(encoding="utf-8"))
        facts.append(MetricFact(chunk.source, chunk.locator, _value_at_path(data, chunk.locator)))
    return facts


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
        "sample size",
        "data points",
        "observations",
    }
    return any(term in q for term in terms) or classify_structural_intent(question) is not None

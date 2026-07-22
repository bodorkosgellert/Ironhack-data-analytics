"""Lightweight markdown highlighting for scannable evidence answers.

Only bolds spans that already appear in the answer text. Does not invent values.
"""

from __future__ import annotations

import re
from typing import Iterable

from .metrics import MetricFact

# Placeholders avoid matching inside protected regions.
_PH = "\x00"
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_BOLD_RE = re.compile(r"\*\*[^*]+?\*\*")
_CITATION_RE = re.compile(r"\[[^\]]+\]")

# Domain terms worth light emphasis when the user asked about them.
_QUERY_FOCUS = (
    "pm2.5",
    "pm25",
    "pearson",
    "r²",
    "r2",
    "correlation",
    "bootstrap",
    "permutation",
    "cross-validated",
    "cross validated",
    "partial correlation",
    "obesity",
)


def _stash(text: str, pattern: re.Pattern[str], bucket: list[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        bucket.append(match.group(0))
        return f"{_PH}{len(bucket) - 1}{_PH}"

    return pattern.sub(repl, text)


def _protect(text: str) -> tuple[str, list[str]]:
    """Hide fences, inline code, existing bold, and citation brackets."""
    bucket: list[str] = []
    text = _stash(text, _FENCE_RE, bucket)
    text = _stash(text, _INLINE_CODE_RE, bucket)
    text = _stash(text, _BOLD_RE, bucket)
    text = _stash(text, _CITATION_RE, bucket)
    return text, bucket


def _restore(text: str, bucket: list[str]) -> str:
    for index, original in enumerate(bucket):
        text = text.replace(f"{_PH}{index}{_PH}", original, 1)
    return text


def _metric_literals(facts: Iterable[MetricFact]) -> list[str]:
    """Exact string forms already present in routed facts — never invented."""
    literals: list[str] = []
    seen: set[str] = set()
    for fact in facts:
        candidates = [fact.rendered_value]
        if isinstance(fact.value, (int, float, str, bool)):
            candidates.append(str(fact.value))
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            # Skip JSON literals that would bold nearly every sentence.
            if candidate in {"true", "false", "null"}:
                continue
            seen.add(candidate)
            literals.append(candidate)
    # Longest first so full precision wins over a short prefix.
    return sorted(literals, key=len, reverse=True)


def _bold_literal(text: str, literal: str) -> str:
    """Bold exact literal occurrences that are not glued inside larger numbers."""
    if not literal:
        return text
    escaped = re.escape(literal)
    # Left: not mid-token / mid-decimal. Right: allow a sentence-ending period
    # (so "... -0.057.") but not a longer number ("67.5" or "1175").
    pattern = re.compile(rf"(?<![\w.])({escaped})(?!\d)(?!\.\d)")
    return pattern.sub(r"**\1**", text)


def _query_focus_terms(question: str) -> list[str]:
    """Conservative query-term emphasis: only known focus phrases present in the question."""
    q = question.lower()
    q_compact = re.sub(r"\s+", " ", q)
    terms: list[str] = []
    for phrase in _QUERY_FOCUS:
        if phrase in q_compact or phrase.replace(" ", "") in q.replace(" ", ""):
            terms.append(phrase)
    # Prefer longer phrases first (e.g. "partial correlation" before "correlation").
    return sorted(set(terms), key=len, reverse=True)


def _bold_query_term(text: str, term: str) -> str:
    """Case-insensitive bold of a focus term; preserves the answer's original casing."""
    if not term:
        return text
    escaped = re.escape(term)
    # Allow flexible whitespace inside multi-word phrases.
    escaped = escaped.replace(r"\ ", r"\s+")
    pattern = re.compile(rf"(?<![\w*])({escaped})(?![\w*])", re.IGNORECASE)
    return pattern.sub(r"**\1**", text)


def highlight_key_spans(
    text: str,
    *,
    metric_facts: Iterable[MetricFact] | None = None,
    question: str | None = None,
) -> str:
    """Return markdown with key metric values (and light query terms) bolded.

    Safe for repeated application: existing ``**bold**``, code fences, inline code,
    and ``[citations]`` are left unchanged. Never inserts numbers that are not
    already in ``text``.
    """
    if not text:
        return text

    working, bucket = _protect(text)

    for literal in _metric_literals(metric_facts or ()):
        working = _bold_literal(working, literal)

    if question:
        for term in _query_focus_terms(question):
            working = _bold_query_term(working, term)

    return _restore(working, bucket)

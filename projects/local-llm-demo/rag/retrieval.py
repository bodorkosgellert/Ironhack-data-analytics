"""Deterministic term frequency-inverse document frequency retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Chunk, build_corpus
from .metrics import exact_metric_chunks

OTHER_GEOGRAPHIES = {
    "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana",
    "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
    "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
    "montana", "nebraska", "nevada", "new hampshire", "new jersey", "new mexico",
    "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee",
    "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming",
}


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation


@dataclass(frozen=True)
class Retrieval:
    passages: list[ScoredChunk]
    refused: bool
    reason: str | None = None


def _search_text(text: str) -> str:
    text = text.lower().replace("_", " ")
    text = re.sub(r"pm\s*2[.\s]?5", "pm25 particulate matter", text)
    text = text.replace("r²", "r2 coefficient determination")
    text = re.sub(r"\b(causes|causality|causally)\b", "causal", text)
    text = re.sub(r"\bindividuals\b", "individual", text)
    return re.sub(r"\s+", " ", text)


def _unsupported_geography(question: str) -> str | None:
    lowered = question.lower()
    return next((name for name in sorted(OTHER_GEOGRAPHIES, key=len, reverse=True) if re.search(rf"\b{re.escape(name)}\b", lowered)), None)


def _safety_chunks(question: str, chunks: list[Chunk]) -> list[Chunk]:
    """Prefer interpretation boundaries for causal phrasing."""
    q = question.lower()
    if not re.search(r"\b(caus\w*|causes|prove)\b", q):
        return []
    preferred = {
        (
            "projects/asthma-air-pollution/README.md",
            "Asthma prevalence and air pollution > Research question",
        ),
        (
            "projects/asthma-air-pollution/v2/VALIDATION.md",
            "Statistical validation and uncertainty > Interpretation boundaries",
        ),
    }
    return [chunk for chunk in chunks if (chunk.source, chunk.locator) in preferred]


class LexicalRetriever:
    """In-memory deterministic retriever over a fixed public corpus."""

    def __init__(self, chunks: list[Chunk] | None = None, threshold: float = 0.12):
        self.chunks = chunks if chunks is not None else build_corpus()
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            preprocessor=_search_text,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(
            f"{chunk.source} {chunk.locator} {chunk.locator} {chunk.locator} {chunk.text}"
            for chunk in self.chunks
        )

    def search(self, question: str, top_k: int = 5) -> Retrieval:
        geography = _unsupported_geography(question)
        if geography:
            return Retrieval([], True, f"The corpus contains Alabama results, not a specific {geography.title()} estimate.")

        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked_indices = scores.argsort(kind="stable")[::-1]
        ranked = [ScoredChunk(self.chunks[i], float(scores[i])) for i in ranked_indices if scores[i] > 0]

        # Exact metric routing is deterministic and takes precedence over prose
        # similarity; no value is calculated or parsed from natural language.
        exact = exact_metric_chunks(question, self.chunks) + _safety_chunks(question, self.chunks)
        exact_ids = {(chunk.source, chunk.locator) for chunk in exact}
        merged = [ScoredChunk(chunk, 1.0) for chunk in exact]
        merged.extend(item for item in ranked if (item.chunk.source, item.chunk.locator) not in exact_ids)
        passages = merged[: max(1, top_k)]

        if not passages or passages[0].score < self.threshold:
            score = passages[0].score if passages else 0.0
            return Retrieval(
                passages,
                True,
                f"No passage met the evidence threshold ({score:.3f} < {self.threshold:.3f}).",
            )
        return Retrieval(passages, False)

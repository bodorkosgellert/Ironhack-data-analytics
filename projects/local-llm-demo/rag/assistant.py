"""Orchestrate retrieval, refusal, and optional local generation."""

from __future__ import annotations

from dataclasses import dataclass

from .ollama import DEFAULT_MODEL, OllamaError, chat, setup_instructions, status
from .prompts import SYSTEM_PROMPT, user_prompt
from .retrieval import LexicalRetriever, ScoredChunk

REFUSAL = "The available evidence does not answer this question."


@dataclass(frozen=True)
class Answer:
    text: str
    passages: list[ScoredChunk]
    refused: bool
    generation_used: bool
    notice: str | None = None


def _context(passages: list[ScoredChunk]) -> str:
    return "\n\n".join(
        f"[{item.citation}]\n{item.chunk.text}" for item in passages
    )


def _retrieval_answer(passages: list[ScoredChunk]) -> str:
    lines = ["Retrieved evidence (values are quoted, not recomputed):"]
    lines.extend(f"- {item.chunk.text} [{item.citation}]" for item in passages)
    return "\n".join(lines)


class EvidenceAssistant:
    def __init__(self, threshold: float = 0.12, retriever: LexicalRetriever | None = None):
        self.retriever = retriever or LexicalRetriever(threshold=threshold)

    def ask(
        self,
        question: str,
        *,
        top_k: int = 5,
        retrieval_only: bool = False,
        model: str = DEFAULT_MODEL,
    ) -> Answer:
        result = self.retriever.search(question, top_k=top_k)
        if result.refused:
            detail = f" {result.reason}" if result.reason else ""
            return Answer(f"{REFUSAL}{detail}", result.passages, True, False)

        if retrieval_only:
            return Answer(_retrieval_answer(result.passages), result.passages, False, False)

        ollama_status = status()
        if not ollama_status.available:
            return Answer(
                _retrieval_answer(result.passages),
                result.passages,
                False,
                False,
                "Ollama was not detected; showing deterministic retrieval only.\n" + setup_instructions(model),
            )
        if model not in ollama_status.models:
            return Answer(
                _retrieval_answer(result.passages),
                result.passages,
                False,
                False,
                f"Model `{model}` is not installed. Run `ollama pull {model}`; showing retrieval only.",
            )

        try:
            generated = chat(model, SYSTEM_PROMPT, user_prompt(question, _context(result.passages)))
        except OllamaError as exc:
            return Answer(
                _retrieval_answer(result.passages),
                result.passages,
                False,
                False,
                f"{exc}\nShowing deterministic retrieval only.",
            )
        citations = "\n".join(f"- [{item.citation}]" for item in result.passages)
        return Answer(
            f"{generated}\n\nRetrieved citations:\n{citations}",
            result.passages,
            False,
            True,
        )

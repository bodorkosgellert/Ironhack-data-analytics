"""Local-first Streamlit interface for the cited asthma evidence assistant."""

from __future__ import annotations

import streamlit as st

from rag.assistant import EvidenceAssistant
from rag.ollama import DEFAULT_MODEL, setup_instructions, status

EXAMPLES = (
    "What is the Pearson correlation between PM2.5 and asthma?",
    "What is the cross-validated R² for the PM2.5-only model?",
    "Why can this ecological study not establish causality?",
    "How are prevalence, incidence, and acute exacerbations different?",
)


@st.cache_resource
def get_assistant(threshold: float) -> EvidenceAssistant:
    return EvidenceAssistant(threshold=threshold)


def main() -> None:
    st.set_page_config(page_title="Local cited evidence assistant", page_icon="📚", layout="wide")
    st.title("Local cited evidence assistant")
    st.caption(
        "This assistant explains existing public aggregate results from the Alabama asthma project. "
        "It does not modify or rerun the epidemiological analysis."
    )

    ollama = status()
    with st.sidebar:
        st.header("Local runtime")
        if ollama.available:
            st.success("Ollama is available")
            if ollama.models:
                st.caption("Installed models: " + ", ".join(ollama.models))
            else:
                st.warning("Ollama is running, but no models were reported.")
        else:
            st.warning("Ollama is unavailable; lexical retrieval remains available.")
            st.code(setup_instructions(), language="text")

        model = st.text_input("Ollama chat model", value=DEFAULT_MODEL)
        top_k = st.slider("Retrieved passages", min_value=1, max_value=10, value=5)
        threshold = st.slider(
            "Low-score refusal threshold",
            min_value=0.0,
            max_value=0.5,
            value=0.12,
            step=0.01,
        )
        retrieval_only = st.checkbox(
            "Lexical retrieval only",
            value=not ollama.available or model not in ollama.models,
            help="Returns cited passages without asking a language model to narrate them.",
        )

    st.subheader("Example questions")
    columns = st.columns(2)
    selected: str | None = None
    for index, example in enumerate(EXAMPLES):
        if columns[index % 2].button(example, key=f"example-{index}", use_container_width=True):
            selected = example

    question = st.chat_input("Ask about the public Alabama asthma analysis")
    question = question or selected
    if not question:
        st.info("Choose an example or enter a question. Unsupported evidence is refused.")
        return

    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Retrieving cited evidence…"):
        answer = get_assistant(threshold).ask(
            question,
            top_k=top_k,
            retrieval_only=retrieval_only,
            model=model,
        )
    with st.chat_message("assistant"):
        if answer.refused:
            st.warning(answer.text)
        else:
            st.markdown(answer.text)
        if answer.notice:
            st.info(answer.notice)

    st.subheader("Retrieved sources")
    if not answer.passages:
        st.caption("No source passage was returned.")
    for rank, passage in enumerate(answer.passages, start=1):
        with st.expander(f"{rank}. {passage.citation} · score {passage.score:.3f}"):
            st.write(passage.chunk.text)
            st.caption(f"Corpus type: {passage.chunk.kind}")

    st.divider()
    st.caption(
        "Safeguards: public allowlisted files only; exact JSON key paths for metrics; "
        "no hosted model or application programming interface keys; no causal inference."
    )


if __name__ == "__main__":
    main()

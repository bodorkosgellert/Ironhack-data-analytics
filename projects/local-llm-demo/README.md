# Local cited evidence assistant

I built this local retrieval-augmented generation (RAG) assistant to explain the existing public results of my Alabama asthma and fine particulate matter analysis. It retrieves traceable passages before answering and can use a local Ollama model without sending repository content to a hosted model.

The assistant is an explanatory layer over saved aggregate evidence. It does not modify data, rerun statistical scripts, produce new epidemiological estimates, or support causal conclusions.

## Architecture

The implementation deliberately uses the repository's existing scikit-learn dependency rather than adding a framework or vector database.

1. `rag/corpus.py` loads an explicit allowlist of public Markdown documentation and deterministic JavaScript Object Notation (JSON) outputs.
2. Markdown is divided along heading boundaries. Long sections use paragraph-aware chunks with a small overlap.
3. Every JSON leaf becomes stable text containing its exact key path, context label, and stored value.
4. `rag/retrieval.py` builds an in-memory term frequency-inverse document frequency index and ranks passages with cosine similarity.
5. `rag/metrics.py` routes recognised numeric intents directly to exact JSON leaves before lexical results are added.
6. `rag/assistant.py` applies a configurable low-score refusal threshold and optionally sends only the retrieved context to Ollama.
7. `streamlit_app.py` and `rag/ask.py` expose the same assistant through graphical and command-line interfaces.

The index is rebuilt quickly in memory. No embeddings, model data, or machine-specific index are required in Git.

## Corpus boundary

I index only these public asthma artifacts:

- the project overview, literature review, roadmap, and Version 2 documentation;
- `metrics.json`, `multivariate_metrics.json`, `robustness_report.json`, and `feature_analysis.json`.

The allowlist excludes `docs/internal/`, repository archives, Version 1 archive-style originals, notebooks, comma-separated value files, generated caches, and all unrelated portfolio projects. Every retrieved chunk retains a repository-relative source path plus a Markdown heading or JSON key path.

## Scientific safeguards

The system prompt requires the local model to:

- answer only from retrieved context and cite the source file and locator;
- state that evidence is unavailable when the context is insufficient;
- narrate stored statistics without inventing or recomputing values;
- distinguish asthma prevalence, incidence, and acute exacerbations;
- avoid causal and individual-level claims from this county ecological study;
- report holdout and cross-validated metrics with appropriate uncertainty.

Deterministic metric routing is important because language-model retrieval alone is not a reliable numeric lookup method. For example, a Pearson question retrieves the full stored value from `$.pearson_r_pm25_asthma`, while a PM2.5-only cross-validation question retrieves `$.models[0].r2_cv_mean`. Ollama may explain these values but is instructed not to calculate new ones.

## Windows setup

From the repository root in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Set-Location projects\local-llm-demo
```

The retrieval-only workflow does not need Ollama:

```powershell
python -m rag.ask --retrieval-only "What is the Pearson correlation between PM2.5 and asthma?"
python -m rag.ask --lexical-only --show-sources --top-k 5 "Why can this study not establish causality?"
```

I use [Ollama](https://ollama.com/download/windows) for optional local generation. Installation and model download are manual; the application never starts a multi-gigabyte download:

```powershell
ollama pull llama3.2:3b
ollama serve
ollama list
```

In a second PowerShell window:

```powershell
Set-Location projects\local-llm-demo
python -m rag.ask --model llama3.2:3b "What is the PM2.5-only cross-validated R²?"
streamlit run streamlit_app.py
```

If Ollama is stopped or the requested model is absent, both interfaces show setup guidance and continue with cited lexical retrieval.

## Corpus inspection, tests, and evaluation

From `projects/local-llm-demo`:

```powershell
python -m rag.index
python -m unittest discover -s tests -v
python evaluate.py
```

The benchmark contains fixed questions about metrics, methodology, limitations, literature, outcome definitions, and an unsupported California estimate. The evaluator runs without Ollama, records ranked paths and locators, and writes `outputs/benchmark.json`. This generated report is ignored by Git.

## Limitations

- Term frequency-inverse document frequency retrieval recognises wording overlap, not deep semantic equivalence. The metric router covers the most important stored numeric intents but is not a general query language.
- The refusal threshold is a practical safeguard, not a calibrated probability of answerability.
- A local language model can still produce poorly worded or incompletely cited narration. Retrieved passages remain the authoritative evidence shown below each answer.
- The corpus covers one small, cross-sectional Alabama county study. It cannot provide other-state estimates, patient advice, individual risks, or causal effects.
- The source analysis has 67 counties, model-based Centers for Disease Control and Prevention PLACES outcomes, narrow PM2.5 exposure contrast, correlated predictors, and uncertain split-specific metrics.
- This implementation does not persist embeddings or evaluate generative faithfulness automatically. A polished version would add a larger paraphrase benchmark, citation-entailment review, latency tracking across machines, and interface accessibility testing.

## Cloud limitation

Streamlit Community Cloud cannot reach Ollama running on my local Windows machine. A cloud deployment would therefore provide retrieval-only behavior unless it used a separately hosted model service. This project intentionally contains no hosted application programming interface key or fallback.

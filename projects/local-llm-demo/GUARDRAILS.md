# Guardrail levels for the local evidence assistant

This note describes practical guardrail levels for `projects/local-llm-demo`, what is
implemented today, and the main tradeoff between **false refusal** (withholding an
answer that exists in the corpus) and **hallucination** (fluent but unsupported claims).

## Levels

| Level | Name | Status | What it does |
|---|---|---|---|
| **L0** | No guardrails | Not used | Free-form model answers with no corpus boundary. High hallucination risk. |
| **L1** | Allowlisted corpus only | **Implemented** | Index only explicit public Markdown/JSON paths in `rag/corpus.py`. Private notes, notebooks, CSVs, and archives cannot enter retrieval by accident. |
| **L2** | Retrieval score threshold / low-confidence refusal | **Implemented** | If the best cosine score is below the threshold (default `0.12`) and no deterministic route matched, refuse. Reduces unsupported answers; can false-refuse paraphrases (see L3). |
| **L3** | Deterministic routing + synonym intent map | **Implemented** | Application code maps recognised intents (Pearson metrics, sample-size synonyms, tract-count / not-patient clarifications, FAQ topics such as predictive gain and outcome definitions) to JSON key paths or preferred documented passages (including `STUDY_FAQ.md`), bypassing weak lexical scores. When routing hits, weaker TF-IDF padding is omitted. |
| **L4** | Geography / out-of-scope topic refusals | **Implemented** (geography) | Unsupported US-state geography questions refuse before generation (e.g. California effect size). Broader topic ontology is only partial via L2. |
| **L5** | Conflict suppression | **Implemented** | When metric facts are routed, generated narration that invents other numbers is dropped; the JSON value remains authoritative. |
| **L6** | Optional hosted LLM with secrets | **Implemented** (optional) | Hosted OpenAI-compatible chat only via secrets; never commit keys. Suitable for this **public** corpus only — do not send confidential data. |
| **L7** | Human review / IRB patterns | **Future** | Required if the corpus ever included real patient-level or IRB-bound records: access control, audit, retention, and human sign-off. Not claimed for this public PLACES demo. |

## Tradeoffs

- **Tighten L2 alone** → fewer hallucinations, more false refusals on natural paraphrases (“sample size”, “data points”, “observations”).
- **Add L3 synonym / metric routing** → fewer false refusals for known structural and numeric facts, without asking the model to invent N.
- **Keep L4 geography refusals** → honest about Alabama-only evidence; do not weaken this to chase recall.
- **L5 vs fluent narration** → users may see retrieval-only or omitted narration when the model conflicts with JSON; that is intentional.
- **L6** trades operational convenience for data-residency risk; fine for public aggregate docs, wrong for confidential health text without institutional controls (L7).

## Presentation (UI)

- Streamlit bolds key numbers from routed/verified facts for scanning (`rag/highlight.py`); it does not invent values.
- Optional **Rephrase evidence** uses the same Ollama/hosted path with a rewrite-only prompt, then still shows the authoritative verified/retrieved block. L5 still drops the summary if numbers conflict. Rephrase requires generation (disables retrieval-only).

## Structural synonym examples (L3)

County sample size (`$.n_counties` = 67) accepts phrasings such as *data points*, *sample size*, *observations*, *rows*, *how many counties*, and *what is n*.

Version 1 tract structure routes to documented passages for *census tracts*, *tracts*, *v1 rows*, and *1175* — geographic rows, not patients.

Questions about *participants* / *patients* retrieve clarification that CDC PLACES estimates are model-based geographic prevalence units, not individual patient records. They do **not** answer with county N as if it were a patient count.

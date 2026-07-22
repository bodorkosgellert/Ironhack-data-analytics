# Evidence assistant eval set

Durable question set for the Alabama asthma / PM2.5 evidence assistant in this project.

## File

- [`evidence_assistant_eval.json`](evidence_assistant_eval.json) — ~17 cases with `id`, `question`, `expected_answer`, optional `must_include` / `must_not_include`, `category`, and notes.
- Categories: `metric` | `synonym` | `limitations` | `refusal` | `methods` | `insight`.

Values were taken from allowlisted study outputs and docs (not invented):

| Value | Source |
|---|---|
| Pearson r `-0.05726740504810031` | `v2/outputs/metrics.json` → `$.pearson_r_pm25_asthma` |
| `n_counties` `67` | `v2/outputs/feature_analysis.json` (also multivariate / metrics) |
| Partial r `0.1996` | `v2/outputs/multivariate_metrics.json` |
| PM2.5-only `r2_cv_mean` `-0.321694185275063` | same multivariate file, `models[0]` |
| PM2.5 CV `0.03469709687386365` | `feature_analysis.json` variance block |
| Bootstrap CI `[-0.2785, 0.1971]` | `robustness_report.json` |
| Obesity–asthma r `0.8646` | multivariate Pearson matrix |
| V1 tracts `1175` | `v1/README_EXTENDED.md`, `robustness_report.json` v1_risk |

Refusal / boundary behavior matches `GUARDRAILS.md` (L3 synonyms, L4 geography) and `evaluation/benchmarks.json`.

## Run checks

From `projects/local-llm-demo` (no Ollama required for the default path):

```bash
python evals/run_evidence_eval.py
```

Optional composed-assistant scoring (needs a running Ollama model):

```bash
python evals/run_evidence_eval.py --mode assistant --model qwen2.5-coder:3b
```

Results write to the ignored path `outputs/evidence_assistant_eval_results.json`.

Quality bar notes (retrieval-only):

- Metric / synonym cases should return `Verified stored metric` (deterministic JSON routing), not a long TF-IDF dump.
- Interpretation cases should prefer `STUDY_FAQ.md` / VALIDATION passages and must not surface markdown-table fragments such as `y resolution`.
- Geography refusals remain hard refusals; participant questions clarify ecological units without answering `$.n_counties` as a patient headcount.

## Relation to other harnesses

| Harness | Role |
|---|---|
| `evaluation/benchmarks.json` + `evaluate.py` | Retrieval source/locator smoke tests (16 cases) |
| `compare_models.py` | Five fixed model-comparison cases (raw vs assistant) |
| `evals/evidence_assistant_eval.json` + `run_evidence_eval.py` | Broader answer-content checks (`must_include` / refusals) |

This eval set does not replace `compare_models.py`; it extends coverage for synonyms, limitations, and extra refusals with the same assistant entry point (`EvidenceAssistant.ask`).

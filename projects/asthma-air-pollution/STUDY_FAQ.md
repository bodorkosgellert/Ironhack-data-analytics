# Study FAQ (curated evidence snippets)

Short, citable answers for the public evidence assistant. Values match Version 2 JSON outputs and project READMEs. This file does not replace the full methods documents.

## Sample size and geographic units

Version 2 analyses **67 Alabama counties** (`n_counties = 67`). Each observation is a county-level CDC PLACES model-based adult asthma prevalence estimate paired with Open-Meteo annual mean PM2.5 at the county centroid — not an individual patient or study participant.

Version 1 used **1,175** Alabama census-tract prevalence rows. Those were geographic estimates, not patient records, and the original Excel join is not treated as validated.

## Does PM2.5 improve prediction after health indicators?

No. After smoking, obesity, diabetes, physical inactivity, and binge drinking prevalence are included, **PM2.5 did not improve** holdout or cross-validated prediction in the current outputs. Confounders-only models reach holdout R² near 0.86 and cross-validated R² near 0.76; adding PM2.5 does not raise those scores. Higher R² here means better county-level prediction, not proof of causation.

## Ecological design and individual causation

This is an **ecological**, cross-sectional county study. County associations cannot be transferred to individuals and do not establish individual-level causation. The Modifiable Areal Unit Problem also means results can depend on geographic boundaries.

## Prevalence, incidence, and acute exacerbations

Chronic asthma **prevalence**, asthma **incidence**, and acute **exacerbations** are distinct outcomes. Version 2 uses current adult asthma prevalence from CDC PLACES. It does not measure incidence or daily acute exacerbations. Short-term PM2.5 spikes are better studied with daily outcomes and lagged exposure designs.

## Why local models matter for denser asthma data

Version 2 uses **public county aggregates**, not patient tracking. Many other asthma studies collect denser personal streams (symptoms, devices, location, diaries). Those corpora motivate keeping retrieval and optional narration **local or inside a private network**. See the evidence assistant note with example citations: [Local inference and confidential health data](../local-llm-demo/README.md#local-inference-and-confidential-health-data).

## Key stored metrics (Alabama 2023)

- Pearson correlation PM2.5 × asthma: about −0.057 (exact leaf in `v2/outputs/metrics.json`).
- Partial correlation PM2.5 × asthma after linear adjustment: 0.1996 (suppression-sensitive; not causal).
- PM2.5-only cross-validated R² mean: about −0.32 (worse than predicting the mean).
- PM2.5 coefficient of variation: about 0.035 (limited exposure contrast).

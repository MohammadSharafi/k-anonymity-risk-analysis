# k-anonymity-risk-analysis

MSCS 714 seminar project on k-anonymity, record linkage attacks, and privacy-utility trade-off analysis using the Adult Census Income dataset.

## Project Goal

This repository implements Project 4 for the MSCS 714 Data Privacy Seminar:

- build a k-anonymization pipeline using only generalization and suppression
- evaluate re-identification risk through exact-match linkage attacks
- measure how privacy protection changes data utility
- compare multiple `k` values, QI sets, and anonymization strategies

The current project uses the Adult Census Income dataset already included in the repo under `data/raw/adult`.

## Current Scope

The starter implementation already includes:

- dataset loading and cleaning
- two quasi-identifier sets
- explicit generalization hierarchies
- two anonymization strategies
- linkage attack evaluation
- utility metrics
- an experiment runner that saves releases, summary metrics, and plots
- a starter report in `report/project4_k_anonymity_report.tex`

## Repository Layout

```text
data/
  raw/adult/                  Raw Adult dataset files
report/
  project4_k_anonymity_report.tex
  project4_k_anonymity_report.pdf
results/
  figures/                    Generated plots
  metrics/                    CSV summaries
  releases/                   Anonymized dataset outputs
scripts/
  run_experiments.py          Main experiment entry point
src/
  k_anonymity_risk_analysis/  Project package
```

## Experiment Design

The repo is configured around the course requirements:

- `k = 2, 5, 10`
- two QI sets:
  - `set_a = [age, sex, education, marital_status]`
  - `set_b = [age, sex, occupation, native_country]`
- two anonymization strategies:
  - `generalization_first`
  - `targeted_suppression`

That produces 12 core experiment settings.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Experiments

```bash
python scripts/run_experiments.py
```

Optional flags:

```bash
python scripts/run_experiments.py --data-path data/raw/adult/adult.data --sample-fraction 0.2
```

Outputs are written to:

- `results/metrics/experiment_summary.csv`
- `results/figures/risk_vs_k.png`
- `results/figures/utility_vs_k.png`
- `results/releases/*.csv`

## Notes

- The project intentionally stays within the course scope of `k`-anonymity only.
- No differential privacy, cryptography, or `l`-diversity methods are used.
- The current heuristics are designed to be interpretable and easy to explain in the final report and demo.

# Project 4 Submission Checklist

## Final Files

- `report/project4_k_anonymity_report.pdf`
- `report/project4_k_anonymity_report.tex`
- `slides/project4_presentation.pdf`
- `results/metrics/experiment_summary.csv`
- `results/figures/risk_vs_k.png`
- `results/figures/utility_vs_k.png`
- `README.md`

## Technical Checks

- Run `python scripts/run_experiments.py`
- Run `python scripts/validate_project.py`
- Run `python -m unittest discover -s tests`
- Confirm all release files exist in `results/releases/`
- Confirm the report compiles from `report/project4_k_anonymity_report.tex`
- Confirm the slide deck compiles from `slides/project4_presentation.tex`

## Report Checks

- Confirm title page matches course expectations
- Confirm group member names are added if the instructor requires them
- Confirm repository link is correct in any submission portal
- Confirm plots in the PDF are readable
- Confirm appendix tables match `results/metrics/experiment_summary.csv`

## Presentation Checks

- Keep the talk within 10 to 12 minutes
- Assign speaking parts to group members
- Prepare a short demo: run experiments, open plots, open report PDF
- Be ready to explain:
  - why QI Set A and QI Set B differ
  - why unique match rate becomes zero
  - why information loss is a better utility metric than row count alone

## Remaining External Inputs

- Final group member names and IDs
- Any professor-specific formatting or upload instructions

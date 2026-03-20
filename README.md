# Safe Learning to Defer for Insurance Fraud Detection

This repository contains the LaTeX source and experiment code for the ICML 2025 paper:

> *Safe Learning to Defer with CVaR-Penalised Bayesian Uncertainty for Insurance Fraud Detection*

## Repository Structure

```
IX-Safety-Latex/
├── main.tex                    # Main LaTeX document
├── sections/                   # Paper sections
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── background.tex
│   ├── formulation.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── related.tex
│   └── discussion.tex
├── appendix/                   # Appendices
│   ├── appendix_math.tex
│   ├── appendix_data.tex
│   ├── appendix_algorithm.tex
│   └── appendix_hyperparams.tex
├── icml2025/                   # ICML 2025 style files
├── figures/                    # Generated figures (PDF + SVG)
├── code/                       # Experiment code
│   ├── run_all_experiments.ipynb       # Main notebook — runs all experiments
│   ├── fraud_model.py                  # Base utilities (data loading, preprocessing)
│   ├── fraud_model_advanced.py         # Bayesian inference, causal analysis, model comparison
│   ├── fraud_model_l2d.py             # CVaR-penalised Learning to Defer framework
│   ├── fraud_model_experiments.py      # Extended experiments (bootstrap, spectral risk, etc.)
│   ├── fraud_model_new_baselines.py    # Baseline comparisons (Mozannar-Sontag, cost-sensitive)
│   └── data/
│       └── insurance_fraud_claims.csv  # Dataset
└── requirements.txt            # Python dependencies
```

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda install --file requirements.txt
```

## Running Experiments

All experiments are consolidated in a single Jupyter notebook:

```bash
cd code
jupyter notebook run_all_experiments.ipynb
```

The notebook:
1. Runs all four experiment modules end-to-end
2. Generates 21 publication-quality figures (PDF + SVG) into `figures/`
3. Figures are referenced by the LaTeX source via `\includegraphics{figures/...}`

## Compiling the Paper

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

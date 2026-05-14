#!/usr/bin/env bash
# Reproducible pipeline template.
#
# Put ANYTHING you want here — this file is yours. The only requirement is
# that a grader can clone your repo and run `bash run.sh` to regenerate
# your final figure from scratch, with no manual steps.
#
# The `uvx ...` commands below are just one convenient way to get
# reproducibility: `uvx` spins up an isolated environment with the exact
# packages you ask for, so the grader doesn't need to install anything
# beyond `uv` itself. You're free to use plain `python`, a Makefile, a
# conda env, Docker, or whatever you prefer — as long as `bash run.sh`
# Just Works on a fresh clone.
#
# Conventions to keep:
#   - No manual steps.
#   - Data goes in `data/`, figures go in `figures/`.
#   - If you download/scrape data, do it here too (see the S&P 500
#     example below for a starting pattern).

set -euo pipefail

mkdir -p data figures

# ---------------------------------------------------------------------------
# Step 1 — (re)generate raw data.
#
# For the three provided datasets, the CSVs are already committed — you do
# not have to regenerate them. But if you want your pipeline to pull the
# freshest version (e.g., for S&P 500), call a fetcher here. Example
# (uvx is optional — plain `python scripts/fetch_sp500.py` works too if
# you've already installed the deps):
#
#   uvx --with pandas --with lxml python scripts/fetch_sp500.py
#
# If you bring your own data, put the download / scrape / assemble step
# here so a grader can reproduce it without any manual clicking.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Step 2 — run the analysis and save the figure.
#
# Adapt the filename to whatever you named your submission. Again, `uvx`
# is just for reproducibility — `marimo run submission.py ...` is fine
# too if marimo is already installed.
# ---------------------------------------------------------------------------

uvx marimo run --sandbox submission.py --output figs/uni_semaxis.png

uv run python semantic_axis.py # plotly viz saved as html


echo "Done. See plotly viz in figs/uni_semaxis.png"


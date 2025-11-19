# Hierarchical World Model Project

## Overview

This repository contains code and experiments for hierarchical world
model research

## Member
shiropa-uk, t-yamada02, ziwoo3244

## Structure

-   `docs/` -- proposals, reports, meeting notes\
-   `experiments/` -- configs, results, experiment scripts\
-   `src/` -- model implementations, training loops, evaluation code\
-   `data/` -- raw and processed datasets\
-   `notebooks/` -- exploratory analysis\
-   `tests/` -- unit and smoke tests\
-   `.github/` -- CI workflows, templates\
-   `docker/` -- environment setup\
-   `scripts/` -- utility scripts

## Usage

### Setup

    pip install -r requirements.txt

### Run an experiment

    python experiments/scripts/run_experiment.py --config experiments/configs/example.yaml

### Reproducibility

-   All configs stored in `experiments/configs/`
-   Seeds fixed in training code
-   Results stored under `experiments/results/<experiment>/`

## Contribution

-   Use feature branches
-   Submit PRs to `develop`
-   Follow templates under `.github/`

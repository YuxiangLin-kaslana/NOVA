# NOVA

NOVA is an experimental codebase for online open-vocabulary anomaly learning. It extends a conventional anomaly detector with novelty gating, semantic type naming, vocabulary growth, guarded updates, and reusable concept memory.

This repository contains code only. Frozen paper results and figure data live in the separate [NOVA_data repository](https://github.com/YuxiangLin-kaslana/NOVA_data).

## Code layout

- `code/sigla_exp/`: core data, profile, detector, early-warning, and online-memory components.
- `code/sota_compare/`: paper experiment runners, controlled baselines, ablations, and result collectors.
- `code/sigla_pipeline/`: end-to-end profile and pipeline entry points.
- `code/policy/`: cost-aware inspection and learned-policy experiments.
- `code/mimic/`: optional MIMIC fusion experiment entry points; data access is not included.
- `code/scripts/`: earlier experiment and training entry points.
- `code/tests/`: deterministic tests for online memory and many-type protocols.
- `code/eval/` and `code/train/`: evaluation and training utilities.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch installation can depend on the local CUDA version. If needed, install the appropriate PyTorch build first and then install the remaining requirements.

## Quick checks

Run from the repository root:

```bash
cd code
PYTHONPATH=. python -m unittest discover -s tests
python -m sigla_exp.train --dataset synthetic --task detector --epochs 2 --limit_batches 3
python -m sigla_exp.train --dataset synthetic --task pipeline --epochs 1 --limit_batches 1
```

Experiment launchers under `code/sota_compare/` write result JSON files to `code/runs/`, which is intentionally ignored by Git.

## Data and API credentials

Datasets, checkpoints, run outputs, W&B state, and Slurm logs are not included. Configure their paths locally. Experiments that use semantic naming read credentials from environment variables such as `OPENAI_API_KEY`; never commit `.env` or credentials.

This is a research codebase. Individual paper experiments may require dataset-specific preprocessing, a CUDA-enabled environment, or a cluster scheduler.

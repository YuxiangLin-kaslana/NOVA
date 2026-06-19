# Minimal SigLA Experiment Code

This directory contains a small PyTorch implementation for early experiments
around the SigLA paper draft and the CANDI codebase.

The code keeps the paper structure but makes it runnable:

- `Perceive`: sliding-window multivariate signal loading.
- `Profile`: lightweight concept profile extraction for spike, level shift,
  seasonal break, contextual deviation, and correlation break.
- `Decide`: a small Signal-Profile-Action policy trained by behavior cloning
  from weak precursor-window labels.
- CANDI-style baseline: an MLP reconstruction autoencoder for anomaly scoring.

## Files

- `sigla_exp/data.py`: SMD, SWaT, `.npz`, and synthetic data loaders.
- `sigla_exp/profiles.py`: simple calibrated concept-profile features.
- `sigla_exp/actions.py`: precursor-window weak action labels.
- `sigla_exp/model/`: current detector, concept extractor, policy, and fallback components.
- `sigla_exp/train/`: training entry point and framework training logic.
- `sigla_exp/train.py`: compatibility wrapper for the training entry point.

## Quick Smoke Tests

Run from `/u/ylin30/sigLA/code`:

```bash
python -m sigla_exp.train --dataset synthetic --task detector --epochs 2 --limit_batches 3
python -m sigla_exp.train --dataset synthetic --task policy --epochs 2 --limit_batches 3
python -m sigla_exp.train --dataset synthetic --task pipeline --epochs 1 --limit_batches 1
```

## CANDI SMD Data

The default data directory points to the shared SigLA data directory:

```bash
python -m sigla_exp.train \
  --dataset SMD_1-7 \
  --task detector \
  --epochs 5
```

Train the minimal SigLA action policy from the labeled test split. The current
trainer always builds its train/validation windows from `bundle.test`, so the
dataset train split is not used as the training source:

```bash
python -m sigla_exp.train \
  --dataset SMD_1-7 \
  --task policy \
  --epochs 5
```

Outputs are saved under `runs/<run_name>/` with `config.json`,
`metrics.json`, and `checkpoint_best.pt`.

## Expected Input Format

For `.npz` data, provide:

- `train`: array shaped `[time, variables]`
- `test`: array shaped `[time, variables]`
- `test_label` or `test_labels`: binary point labels shaped `[time]`
- optional `train_label`: binary train labels

The current action labels are weak labels derived from event onsets:

- stable region: `wait`
- valid precursor window: `alarm`
- late pre-onset region: `request_evidence`
- on-event/post-onset anomalous window: `escalate`

This is a minimal experimental scaffold, not the full RL/preference-training
system described in the paper.

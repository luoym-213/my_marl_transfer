# Chapter 3: KF and CommDrop state estimation

This implementation keeps the observation-radius-0.5 HIG-SAR `ep400.pt`
frozen.  Only teammate motion state `[px, py, vx, vy]` is estimated.  Ego-local
belief maps, landmarks, claims, options, targets, task progress, and active
flags continue to come from the communication-limited cache.

## Runtime information boundary

`EstimatorCommunicationEnv` exposes only packets that actually arrived on the
current control step.  Each event contains sender/receiver IDs, source and
arrival steps/times, motion state, option, target, task progress, active state,
and the receiver's own historical motion at the same source step.  Out-of-order
packets remain visible for causal auditing but are rejected by the local cache
and all estimators.  Estimator predictions are never written to either the
communication cache or ego history.

All methods implement:

```text
reset(initial t=0 synchronization)
  -> ingest_deliveries(real arrivals only)
  -> predict(query step, live ego state)
```

The result holds a mean, full covariance, AoI in seconds, and source step for
every directed ego/sender relation.  Query propagation is non-persistent.

KF uses a constant-velocity transition with `dt=0.1s`, position-only updates,
and the Joseph covariance form.  Base Q is the one-step CV residual covariance
on training episodes; R is `1e-4` times training position variance.  Each
communication level selects independent Q/R multipliers from `{0.25, 1, 4}` on
validation spatial-edge RMSE, with position RMSE as the tie-breaker.

CommDrop implements the Chapter-3 event/history/query graph: 16-dimensional
Fourier time features (eight geometrically spaced time scales ω from 0.1 to 5.0 seconds), a
shared GRU with independent directed histories, AoI queries, live ego and task
semantic encoders, directed fully connected dependency gates without
self-loops, `0.30s` source-freshness filtering, normalized multi-source
residuals, AoI residual gating, and diagonal Gaussian mean/variance decoding.
The training loss is normalized heteroscedastic Gaussian NLL plus the weak
dependency-gate sparsity regularizer in equations (3.33)--(3.35).

## Reproduction

Use the repository environment and local packages:

```bash
export PYTHONPATH=src/multiagent/mape:src/gym-vecenv
PY=/home/robo/anaconda3/envs/mpe_trans/bin/python
```

1. Collect the 1000 fixed-policy ideal trajectories.  Episode `k` uses seed
   `7378+k`; the saved split is strictly `700/150/150`.

```bash
$PY collect_estimator_trajectories.py --no-cuda
```

2. Calibrate three KFs and train three CommDrop models.  Channel replay for
   every level/episode uses seed `20260905+k` and the paper profile.

```bash
$PY train_estimators.py --estimator all --device cuda --require-formal-dataset
```

For a CPU run use `--device cpu`.  Training defaults are AdamW (`lr=1e-3`,
`weight_decay=1e-5`), episode batch 16, 100 epochs, gradient clipping at 1, and
validation early stopping after 15 epochs.  If the default model does not beat
both validation baselines, the script automatically runs only the allowed
fallback grid.  `--no-fallback-search` exists for smoke tests, not the formal
experiment.

3. Run validation-gated offline and paired closed-loop evaluation:

```bash
$PY evaluate_estimators.py --mode both --no-cuda \
  --require-formal-dataset --num-closed-loop-episodes 150
```

The evaluator first reads only the validation split.  It opens the test split
and runs closed-loop evaluation only if CommDrop beats Stale on spatial-edge
RMSE at every level and its macro spatial-edge RMSE beats KF.  JSON and
Markdown results include per-episode mean, standard deviation, and 95%
confidence interval.  Checkpoints include model state, normalization,
configuration, communication profile, split, seeds, best epoch, validation
metrics, and the frozen-policy SHA-256.

All large data and checkpoints are written outside the Git repository to:

```text
../marlsave/save_new/chapter3_95_estimators_20260905/
```

## Fast verification

```bash
$PY -m unittest \
  tests.test_estimators \
  tests.test_estimator_data \
  tests.test_estimator_integration -v
```

The integration smoke test constructs a small deterministic trajectory set,
calibrates KF, trains CommDrop for one epoch, reloads checkpoints, and checks
the policy adapter.  Formal 1000-episode collection/training is intentionally
separate from unit tests.

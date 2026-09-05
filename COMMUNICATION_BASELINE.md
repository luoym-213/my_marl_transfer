# Chapter 3 communication degradation baseline

This branch implements Parts 1 and 2 of the Chapter 3 work.  The trained
HIG-SAR policy remains frozen; CommDrop itself is intentionally not included in
this baseline.

## Information model

- Every UAV maintains its own belief map and updates it only from its own
  observation footprint.
- A successful directed packet carries the sender's timestamped position,
  velocity, high-level goal/task state, active flag, and belief-map snapshot.
- Received belief maps are fused cell-wise by retaining the posterior farther
  from the common `0.5` prior.  This gives an idempotent union of static-scene
  evidence and avoids counting the same relayed map repeatedly.
- Unknown landmarks are derived independently from each local belief map.  A
  landmark becomes visible to another UAV only through a delivered map
  snapshot.
- At `t=0` all UAV states are synchronized, as assumed in Chapter 3.  The stale
  estimator then holds the newest source-timestamped packet for each teammate;
  late packets older than the cache are discarded.
- Both high-level task-graph construction and the low-level teammate-position
  observation consume the ego UAV's local cache.  Ground truth remains inside
  the simulator only for physics and reward/metric computation.
- The degraded environment removes synchronized global map fields from the
  policy-facing `info`; its legacy reset-only landmark heatmap is initialized
  to zeros. New episodes also clear prior goals before the `t=0` sync.

## Communication profiles

| Level | Radius | Nominal geometric delay | Maximum delay | Packet loss |
| --- | ---: | ---: | ---: | ---: |
| Mild | 1.5 | 1 | 2 | 0.05 |
| Medium | 0.9 | 2 | 5 | 0.15 |
| Severe | 0.6 | 4 | 10 | 0.30 |

The delay sample is `min(Geometric(p=1/m), M)` simulation steps.  Packets are
directed and loss is sampled independently per sender/receiver pair.

## Frozen-policy baseline

The requested observation-radius-0.5 checkpoint was evaluated for 50 paired
episodes per condition. Communication degradation produces the expected
monotonic decline:

| Condition | Success rate | Mean successful steps | Mean final distance |
| --- | ---: | ---: | ---: |
| Ideal | 94% | 30.79 | 0.0763 |
| Mild | 82% | 34.05 | 0.1166 |
| Medium | 68% | 39.62 | 0.1506 |
| Severe | 44% | 41.91 | 0.2195 |

The complete machine-readable metrics are stored in
`communication_baseline_results.json`.

## Reproduce the baseline

Use the repository's existing `mpe_trans` environment and local packages:

```bash
PYTHONPATH=src/multiagent/mape:src/gym-vecenv \
  /home/robo/anaconda3/envs/mpe_trans/bin/python eval_comm.py --no-cuda
```

By default this evaluates the ideal, Mild, Medium, and Severe conditions for 50
episodes using:

- checkpoint: `../marlsave/save_new/chapter2_94_attn_fallback_20260905_05/ep400.pt`
- observation radius: `0.5`
- trajectory seed: `7378`
- communication seed: `20260905`

Every condition resets episode `k` with trajectory seed `7378 + k`; degraded
conditions independently reset their channel RNG with `20260905 + k`. This
keeps initial UAV/landmark configurations matched even when earlier closed-loop
trajectories consume different numbers of RRT random samples.

Use `--comm-results-path <path>` to save the printed JSON summary.  Other
evaluation flags such as `--num-eval-episodes`, `--seed`, and `--render` are
passed through to the existing evaluator.

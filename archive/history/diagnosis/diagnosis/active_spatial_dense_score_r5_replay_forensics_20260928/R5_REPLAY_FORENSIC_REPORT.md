# r5 historical replay forensic report

Final status: `REAL_HISTORICAL_REPLAY_PASS`.

Scope is historical r5 replay only. No S0/S1/S5 reward intervention, model
forward, backward, optimizer step, checkpoint mutation, renderer call, or new
rollout was performed.

## First numerical divergence

The first semantic divergence was Stage D, trajectory construction, in the
custom offline replayer. It stacked raw `group_idx`, `traj_idx`, and `turn_idx`
with NumPy. Because r5 group IDs are UUID strings, NumPy promoted all three
columns to strings. Sorting `turn_idx` then became lexicographic:

```text
production: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
old custom: 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9
```

The first affected trajectory is group
`269c599c-df79-4c88-af1e-25216667624c`, trajectory 2, task
`0014_841007/bed+wardrobe/left_of/6`. The legacy path reached a maximum raw-GAE
error of `0.4765625`; downstream whitening made the original final advantage
error much larger. The complete per-turn reward/value/delta/raw trace is in
`first_mismatch_trace.json`.

The minimal fix mirrors production: factorize only non-numeric group IDs, cast
trajectory and turn IDs to `int64`, then build the key matrix. Reward, GAE,
terminal handling, whitening, and snapshot values were not changed.

## Stage results

| Stage | Result | Evidence |
| --- | --- | --- |
| A row identity/order | PASS | 300 snapshot rows, 297 unique triples, 3 padded duplicates; duplicate tensor/trace content is exact. |
| B reward reconstruction | PASS | Saved score vs actual trace total max error `0`; component reconstruction max error `2.48e-19`. |
| C turn value extraction | PASS | Production and replay use first valid response token; all first positions are 0; values are BF16. |
| D trajectory construction | PASS after fix | Numeric turn order now matches production; legacy string ordering reproduces the bug. |
| E TD residual | PASS | Production-formula vs fixed custom delta max error `0`. |
| F raw GAE | PASS | Production reconstruction vs fixed custom raw GAE max error `0`; legacy error `0.4765625`. |
| G whitening | PASS | 297 unique rows, 17,801 valid tokens, unbiased variance, epsilon `1e-8`; final valid-token error `0`. |
| H token broadcast/duplicate expansion | PASS | 297 unique results expand to 300 rows; all-token advantage error `0`. |
| I final snapshot parity | PASS | Advantages, returns, value mask, and token rewards all have max error `0`. |

## Production versus custom replay

Directly calling production
`compute_gae_no_concat_advantage_return_firsttok` on the frozen snapshot gives:

- production replay vs online saved advantages: max error `0`;
- production replay vs online saved returns: max error `0`;
- fixed custom replay vs online saved advantages: max error `0`;
- fixed custom replay vs online saved returns: max error `0`.

This is case A from the requested decision table: online equals production
replay, and the pre-fix custom replayer was wrong.

## Whitening and BF16 note

Whitening is computed on unique semantic turns, not on all 300 padded rows:

- unique rows: 297;
- unique valid-token population: 17,801;
- post-expansion valid-token count: 18,000;
- mean: `-2.3017001152038574`;
- variance: `0.07959889620542526`;
- std: `0.2821327745914459`;
- unbiased variance with epsilon `1e-8`.

Subtracting a saved BF16 value from a saved BF16 return is not an exact raw-GAE
oracle because both operands have already been rounded. That audit-only
calculation differs by up to `0.015625`. Raw parity is instead checked inside
the production/custom recursion before return construction and is exactly 0;
the final saved production tensors are also exactly reproduced. No tolerance
was widened.

## Snapshot sufficiency

r5 does not store the original pre-balance physical row index, but it is not
needed for historical replay. Production itself defines padding duplicates by
the `(group_idx, traj_idx, turn_idx)` triple. The three duplicate groups have
identical model inputs, masks, scores, values, and reward traces, so the unique
set and inverse expansion are deterministically recoverable. r5 is therefore
not `SNAPSHOT_SCHEMA_INSUFFICIENT` for this analysis.

## Verification

Forensic command:

```bash
/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python \
  scripts/active_spatial_r5_replay_forensics.py \
  --snapshot exps/vagen_active_spatial/active_spatial_dense_score_diagnostic_real_batch_20260916_r5/snapshot/global_step_000001_pre_update \
  --output-dir docs/diagnosis/active_spatial_dense_score_r5_replay_forensics_20260928
```

Regression command:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/pytest -q \
  tests/active_spatial/test_ppo_snapshot_replay.py \
  --junitxml=/tmp/active_spatial_r5_replay_pytest.xml
```

Result: `12 passed`, `0 failed`, `0 errors`, `0 skipped` in 28.980 seconds.
The added padding fixture starts with five mixed-length turns, applies the same
prefix-copy rule needed for four DP ranks (`5 -> 8`), reorders the rows, and
checks unique-population whitening plus production/custom tensor equality.

## Next state

`REAL_HISTORICAL_REPLAY_PASS`.

r5 can be used in a later, separately authorized round for S0/S1/S5
reward/advantage/gradient intervention. This round stops here.

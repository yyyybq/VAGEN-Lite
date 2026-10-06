# Commands and lifecycle ledger

## Commands actually run

Read-only inspection used `git status`, `git rev-parse`, `find`, `rg`,
`sed`, `stat`, `wc`, `sha256sum`, and Python JSON parsing against the repository
and frozen artifacts.

r5 replay regression:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  /mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/pytest -q \
  -p no:cacheprovider tests/active_spatial/test_ppo_snapshot_replay.py
```

Result: `12 passed`, `0 failed`, with two CUDA/pynvml environment warnings.

## Commands deliberately not run

- Zoetrope renderer submission: `NOT_RUN`
- Zoetrope 8xH800 trainer submission: `NOT_RUN`
- renderer health/render request: `NOT_RUN`
- COMMON_STEP0 creation: `NOT_RUN`
- S0/S1/S5 config materialization and launch: `NOT_RUN`
- r5 gradient backward: `NOT_RUN`
- optimizer step: `NOT_RUN`
- canonical model evaluation: `NOT_RUN`

Reason: the first formal GPU launch gate failed. No speculative submission
command or fake job identity is recorded.

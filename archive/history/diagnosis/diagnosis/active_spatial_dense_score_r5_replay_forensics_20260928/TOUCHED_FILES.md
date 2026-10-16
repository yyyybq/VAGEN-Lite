# Touched files and diff scope

Files changed for the r5 forensic task:

- `vagen/utils/active_spatial_ppo_replay.py`
  - normalize group/trajectory/turn IDs exactly like production;
  - expose deterministic raw-turn/delta/unique-index diagnostics;
  - preserve explicit `events=0` potential components as unapplied.
- `vagen/utils/active_spatial_ppo_snapshot.py`
  - load large immutable payloads with `mmap=True` when supported;
  - earlier capture fixes for NumPy scalars and resolved OmegaConf remain in the
    same uncommitted snapshot module.
- `tests/active_spatial/test_ppo_snapshot_replay.py`
  - regression for UUID group IDs with numeric turns 1/2/10;
  - snapshot scalar/config and skipped-potential regressions.
- `scripts/active_spatial_r5_replay_forensics.py`
  - historical-only Stage A-I comparator and artifact generator.
- `docs/diagnosis/active_spatial_dense_score_r5_replay_forensics_20260928/`
  - reports, per-row/per-stage evidence, trace, whitening, comparison, and
    provenance ledger.

The repository already contained many unrelated dirty and untracked files from
other windows. They were neither reset, stashed, nor modified for this forensic
task. No R1/canonical/donor files were edited.

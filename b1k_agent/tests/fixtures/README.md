Test fixtures.

- `turning_on_radio/problem0.bddl`, `picking_up_trash/problem0.bddl`: verbatim copies of
  BEHAVIOR-1K v3.9.3-post2 `bddl3/bddl/activity_definitions/<task>/problem0.bddl` (a8247a8c).
- Observations used by the tests are SYNTHETIC (`tests/synth.py`): official keys, shapes and dtypes,
  random pixels. They check wiring only. Recorded real observations (scripts/record_obs_fixture.py on
  server 105) will be added as `real_obs_*.npz` once the simulator has run.

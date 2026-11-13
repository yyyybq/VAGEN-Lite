"""Training-data interfaces (task spec section 7): skill segments, memory experiences, distillation
records and the training-round registry.

    data/schema.py    dataclasses + JSON (de)serialization + validation: SkillSegment (7.1),
                      ExperienceRecord (M-memory), DistillRecord (7.3, M-distill), the field-category
                      map (7.2) and the source-isolated train/dev split
    data/distill.py   python -m data.distill export RUN_DIR --out FILE.jsonl
                      (provider_call events of agent/recorder.py -> DistillRecords, no truth fields)
    data/rounds.py    python -m data.rounds ...: the round registry (7.4): one trainable component per
                      round, freeze, Dev results, keep / rollback, "the config of round k"

Isolation: nothing here imports diagnostics/ or a simulator package, so runtime code (e.g. a future
M-memory impl reading ExperienceRecords) may import data/. Truth labels are written into
DistillRecords only by diagnostics/label_distill.py, offline.

Status: written and unit-tested on SYNTHETIC inputs and on the recorder output of a mock episode
(tests/test_data_diag_*.py). No real episode has been exported yet (nothing has run on server 105).
"""

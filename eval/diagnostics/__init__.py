"""Diagnostics next to the InteriorGS v10 evaluation (data_gen/interiorgs_v10/evaluation.py), docs/transition_qa.

failure_breakdown.py - what each episode did: reached a view that meets the task, stopped early, used up its
                       actions (best state one gate short, or not), reached and drifted off; per task family.
qa_probe.py          - accuracy of a model on the P1 probe questions (p1_probe.jsonl), per question type, and the
                       curve over a training run's checkpoints.
"""

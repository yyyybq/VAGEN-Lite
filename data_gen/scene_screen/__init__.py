"""The scene screen of 2026-10-28/29 (docs/transition_qa/SCENE_SCREEN.md): features.py and classify.py type the 1,000
InteriorGS scenes from their object labels, page.py builds the review page, final.py reads the user's choices,
exclusions.py writes configs/scene_screen/scene_exclusions_20261029.json, matrix_inputs.py the screened RL rows and
eval sets; c0_restore.py, c0_eval.py and c0_combine.py bring C0 back from AOSS and evaluate it on the added test tasks.
One-off scripts with the cluster paths of that run; the bank and the mixture builder read the list."""

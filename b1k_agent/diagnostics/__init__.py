"""Offline diagnostics: the ONLY package allowed to read simulator truth (task spec section 4).

Runtime packages (agent, modules, executor, providers, eval) never import this package
(tests/test_import_guards.py). Truth is read only in diagnostics/replay_truth.py, inside main(), on
server 105, after the episodes; everything it produces is TRAINING_ANNOTATION data for offline
analysis and labels. See diagnostics/README.md.
"""

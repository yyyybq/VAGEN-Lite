"""Operational scripts for server 105 (Step 1 inventory, Step 4 throughput, Step 5 skill diagnosis, fixtures).

Not a runtime package: nothing in agent/, modules/, executor/, providers/ or eval/ imports from here. The file
exists so that ``from scripts import ...`` resolves to this directory (tests, and scripts that share helpers)
and never to an unrelated top-level ``scripts`` package installed in a venv.

Status of every script: written and unit-tested with mocks / synthetic inputs (tests/test_ops_*.py); none
has been run on server 105 yet.
"""

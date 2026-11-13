"""Pluggable modules, one package per trick. Each package exposes ``IMPLS: dict[str, ImplSpec]``
with at least a "null" implementation. Packages never import each other (tests/test_import_guards.py);
they exchange information only through EpisodeState fields and interface return values."""

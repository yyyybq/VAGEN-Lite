"""M-distill: a small model trained on the strong model's legal-input judgments. Not a loop slot. See README.md.

When it exists it will provide a cheaper implementation for the "check" slot, configured as a cross-package impl:

    modules:
      check: {enabled: true, impl: "distill:small_vlm", params: {...}}

agent/registry.py resolves "distill:<name>" to ``modules.distill.IMPLS[<name>]`` and requires its ImplSpec.slot to be
"check". Training data are data/schema.py DistillRecord rows exported from the provider-call logs by data/distill.py
and labelled offline by diagnostics/label_distill.py (truth only as a label, never as an input).

``IMPLS`` is empty: no impl is written, so "distill:<name>" fails with a KeyError that lists the available names.
"""

IMPLS: dict = {}

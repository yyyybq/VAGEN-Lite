"""Stand-in for ``open_clip``, which the Cambrian source's vision-tower builder imports at module level (for its CLIP
towers) and the StarVLA env does not have.  Cambrian-S uses the SigLIP2 tower, never these; calling them fails.  Only
model_matrix/downstream/cambrian_entry.py puts the stand-ins on the path, at its end, so an installed package would
win."""


def create_model_from_pretrained(*args, **kwargs):
    raise ImportError("open_clip is not installed in this env (model_matrix/downstream/shims stand-in)")


def get_tokenizer(*args, **kwargs):
    raise ImportError("open_clip is not installed in this env (model_matrix/downstream/shims stand-in)")

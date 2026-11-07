"""Group G of matrix v3: the actor entropy over the action segment (vagen/segment_entropy.py)."""

import importlib.util
from pathlib import Path

import pytest

from vagen import segment_entropy as SE

QWEN3 = Path("/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
             "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
HAVE_TORCH = importlib.util.find_spec("torch") is not None


def test_action_token_start_with_pieces():
    pieces = ["<targets>A: (", "469", ",352)</targets", "><", "action>move_forward|", "</action>"]
    ids = list(range(len(pieces)))
    decode = lambda xs: "".join(pieces[i] for i in xs)  # noqa: E731
    assert SE.action_token_start(ids, decode) == 3  # the merged "><" holds the "<" of "<action>"
    d = ["<action>", "move", "_forward", "|", "</action>"]
    assert SE.action_token_start(list(range(5)), lambda xs: "".join(d[i] for i in xs)) == 0
    assert SE.action_token_start([0, 1], lambda xs: "".join(["no", " action"][i] for i in xs)) is None
    assert SE.enabled({"entropy_by_segment": True}) and not SE.enabled({}) and not SE.enabled(None)


@pytest.mark.skipif(not QWEN3.exists(), reason="the Qwen3 tokenizer (cluster)")
def test_action_token_start_with_the_qwen3_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(QWEN3))
    decode = lambda xs: tok.decode(list(xs), skip_special_tokens=True)  # noqa: E731
    g = "<targets>A: (469,352); B: none</targets><action>move_forward|turn_left|</action>"
    ids = tok(g, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
    k = SE.action_token_start(ids, decode)
    assert decode(ids[:k]).startswith("<targets>") and not decode(ids[:k]).endswith("<")
    assert decode(ids[k:]).lstrip(">").startswith("<action>") and decode(ids[k:]).endswith("</action>")
    d = tok("<action>move_forward|</action>", add_special_tokens=False)["input_ids"]
    assert SE.action_token_start(d, decode) == 0


@pytest.mark.skipif(not HAVE_TORCH, reason="torch")
def test_segment_metrics():
    import torch

    def agg_loss(loss_mat, loss_mask, loss_agg_mode):  # token-mean, as verl's
        return (loss_mat * loss_mask).sum() / loss_mask.sum()

    ent = torch.tensor([[0.5, 0.5, 0.1, 0.1, 9.0], [0.2, 0.2, 0.2, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    nt = {"action_spans": [{"spans": [[2, 5]]}, {"spans": []}], "targets_spans": [{"spans": [[0, 2]]}, {"spans": [[0, 3]]}]}
    m = SE.segment_metrics(ent, mask, nt, "token-mean", agg_loss)
    assert m["actor/entropy_action"] == pytest.approx(0.1)  # position 4 is outside the response mask
    assert m["actor/entropy_targets"] == pytest.approx((0.5 + 0.5 + 0.2 * 3) / 5)
    assert SE.segment_metrics(ent, mask, {}, "token-mean", agg_loss) == {}
    assert SE.segment_metrics(ent, mask, {"action_spans": [{"spans": []}, {"spans": []}]}, "token-mean", agg_loss) == {}

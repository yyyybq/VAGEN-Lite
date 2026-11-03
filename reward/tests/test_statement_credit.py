"""Tests of direction 3's statement term (vagen/turn_credit.py turn_term "statement") and the statement's token span
(vagen/statement_truth.statement_token_count)."""

import numpy as np
import pytest
import torch

from vagen.statement_truth import statement_token_count
from vagen.turn_credit import episode_advantages, episode_record, settings, token_advantages, turn_record

CFG = settings({"turn_credit": {"turn_term": "statement", "episode_norm": "none", "turn_weight": 0.5}})


class CharTokenizer:
    """One token per vocabulary piece; '><' is a single piece, as BPE may merge the statement's end with the action."""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)


def test_token_count_takes_the_merged_token_with_the_statement():
    pieces = ["<observe>", "where=A:left", "; dist=far", "</observe", "><", "action>", "turn_left|", "</action>"]
    tok = CharTokenizer(pieces)
    assert statement_token_count(tok, list(range(len(pieces)))) == 5  # up to and with "><"
    plain = ["<observe>", "x", "</observe>", "<action>", "a|", "</action>"]
    assert statement_token_count(CharTokenizer(plain), list(range(len(plain)))) == 3
    assert statement_token_count(CharTokenizer(["<action>", "a|", "</action>"]), [0, 1, 2]) == 0


def test_turn_record_statement_fields():
    info = {"statement": {"readable": True, "accuracy": 0.75}}
    r = turn_record([10, 30], -0.01, info, False, True, [10, 18])
    assert r["statement_accuracy"] == 0.75 and r["statement_span"] == [10, 18]
    r = turn_record([10, 30], -0.01, {"statement": {"readable": False, "accuracy": 0.0}}, False, True, None)
    assert r["statement_span"] == [10, 30]  # no readable statement: the whole answer
    assert "statement_span" not in turn_record([10, 30], 0.0, {}, False, True)
    rec = episode_record(0.1, [turn_record([90, 120], 0.0, info, False, True, [90, 110])], 100)
    assert rec["turns"][0]["statement_span"] == [90, 100] and rec["turns"][0]["span"] == [90, 100]


def test_settings_check_the_term():
    with pytest.raises(ValueError):
        settings({"turn_credit": {"turn_term": "words"}})
    with pytest.raises(ValueError):
        settings({"turn_credit": {"turn_term": "statement", "turn_reward": True}})
    assert settings({"turn_credit": {}})["turn_term"] == "shaping"


def two_groups():
    """Two tasks x 2 episodes; each turn's answer at [s, e), its statement at [s, s + 4)."""
    recs, uids, returns = [], [], []
    plan = {"g1": [(5.0, [1.0, 0.5]), (-0.2, [0.25])], "g2": [(-0.1, [0.0, 0.75]), (-0.1, [1.0])]}
    for uid, eps in plan.items():
        for ret, accs in eps:
            turns, pos = [], 0
            for a in accs:
                turns.append(turn_record([pos, pos + 10], 0.0, {"statement": {"readable": True, "accuracy": a}},
                                         False, True, [pos, pos + 4]))
                pos += 30  # an observation between the answers
            recs.append(episode_record(0.0, turns, 64))
            uids.append(uid)
            returns.append(ret)
    return recs, uids, returns


def test_statement_term_only_on_statement_tokens_and_grpo_elsewhere():
    recs, uids, returns = two_groups()
    mask = torch.zeros(len(recs), 64)
    for i, r in enumerate(recs):
        for t in r["turns"]:
            mask[i, t["span"][0]:t["span"][1]] = 1
    adv, metrics = token_advantages(recs, uids, mask, returns, CFG)
    grpo = episode_advantages(returns, uids)
    accs = np.array([t["statement_accuracy"] for r in recs for t in r["turns"]])
    z = (accs - accs.mean()) / (accs.std() + CFG["eps"])
    k = 0
    for i, r in enumerate(recs):
        for t in r["turns"]:
            s, e = t["span"]
            ss, se = t["statement_span"]
            assert torch.allclose(adv[i, ss:se], torch.full((se - ss,), grpo[i] + 0.5 * z[k], dtype=torch.float32), atol=1e-5)
            assert torch.allclose(adv[i, se:e], torch.full((e - se,), grpo[i], dtype=torch.float32), atol=1e-6)
            k += 1
        assert float(adv[i][mask[i] == 0].abs().sum()) == 0.0  # observations keep 0
    assert metrics["turn_credit/tied_groups"] == 1.0  # g2's returns are equal: actions get 0 there
    assert abs(grpo[2]) < 1e-9 and abs(grpo[3]) < 1e-9

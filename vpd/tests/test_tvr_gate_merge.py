"""G0 item 7 of gate_tvr since PREREG_TVR_PILOT.md section 17 (row 2026-11-11, DECISIONS G16): merge correctness only.

For every merged pilot eval (S, tier): exactly one entry of the registered merge-correctness reports (mp_diag.json,
P/merge_correctness.json pins their sha256) for start S and adapter dir EVAL_ADAPTER[tier] (100s1002 -> 100_s1002),
with the served adapter's sha256, gate_reproduction.ok (and its gate block == the eval job's parity json), the
diagnostic's own U-M == that parity json exactly (mean |dCE|, agreeing and reply counts), served_ok, T-MF mean
|dCE| <= 1e-4 and equal module sets.  The bf16 U-M (gates.merge_parity) is reported, not gated.
Fake pilot roots of test_tvr_jobs.make_gate_root only (8 eval tasks); never the default roots.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from vpd.tests import test_tvr_jobs as TJ
from vpd.tvr import gate_tvr as G
from vpd.tvr import pilot_cfg
from vpd.tvr import pilot_tvr as PT


@pytest.fixture
def gate_root(tmp_path, monkeypatch):
    return TJ.make_gate_root(tmp_path, monkeypatch)


def run_gate(gate_root):
    return TJ.run_gate(gate_root)


def report_paths(L):
    return [p for p in sorted((L.runs / "explore" / "mpdiag").glob("r*/mp_diag.json"))]


def edit_entry(L, key, fn, reregister=True):
    """Edit adapter `key` (e.g. "E2_100") in whichever report holds it; re-pin the registry (as the orchestrator
    would after a new diagnostic) unless reregister is False."""
    for p in report_paths(L):
        doc = json.loads(p.read_text())
        if key in doc["adapters"]:
            fn(doc["adapters"][key])
            p.write_text(json.dumps(doc))
            if reregister:
                TJ.write_registry(L, report_paths(L))
            return p
    raise KeyError(key)


def fails_alone_with(rep, needle):
    TJ._only_fail(rep, "merge_parity")
    probs = TJ._item(rep, "merge_parity")["detail"]["problems"]
    assert any(needle in b for b in probs), (needle, probs)
    return probs


def test_pass_and_bf16_reported_not_gated(gate_root):
    L, cfg, sv, _ = gate_root
    rep = run_gate(gate_root)
    it = TJ._item(rep, "merge_parity")
    assert it["ok"] is True and rep["decision"] == "ruler", it
    facts = it["detail"]["facts"]
    assert facts["max_t_mf_mean_abs_dce"] == 1e-4 and facts["registry_sha256"] == TJ.fsha(L.pilot_root /
                                                                                            G.MC_REGISTRY)
    # one judged entry per merged eval: 4 per start, 5 for Q3B (the replicate), the replicate from adapter 100_s1002
    per = facts["per_eval"]
    assert set(per) == {f"{s}/{t}" for s in pilot_cfg.STARTS for t in ("0W", "100", "ALL")} | {"Q3B/100s1002"}
    assert per["Q3B/100s1002"]["key"] == "Q3B_100_s1002" and per["Q3B/100s1002"]["served_ok"] is True
    # every parity json is outside the old bf16 thresholds (pass False): reported, not gated
    b16 = rep["reported_not_gated"]["merge_bf16"]
    assert b16["note"].startswith("reported, not gated")
    assert set(b16["per_eval"]) == set(per)
    r = b16["per_eval"]["Q3B/100s1002"]
    assert r == {"mean_abs_dce": 0.0133, "reply_agreement": 254 / 267, "replies_agreeing": 254, "n_replies": 267,
                 "max_mean_abs_dce": 0.005, "min_reply_agreement": 0.99, "within_bf16_thresholds": False,
                 "parity_json_pass": False, "gated": False}
    md = G.render_md(rep)
    assert "reported, not gated" in md and "| Q3B/100s1002 | 0.01330 | 0.9513 | 254/267 | 0.005 / 0.99 | False |" in md
    assert "合并是否正确" in md and "Q3B_100_s1002" in md
    # still no between-start numbers anywhere in GATE.json
    keys = set(TJ._keys(rep))
    assert not [k for k in keys if re.search(r"diff|delta|minus|gap|vs_", str(k), re.I)], keys
    assert rep["note"] == "no between-start differences are computed"


def test_bf16_numbers_and_thresholds_never_change_the_verdict(gate_root):
    L, cfg, sv, _ = gate_root
    # reply_agreement is not one of the reproduction keys: a far worse one is reported, nothing else
    TJ._edit(L.parity("Q3S", "ALL"), lambda r: r.update(reply_agreement=0.5, **{"pass": False}))
    cfg["gates"]["merge_parity"]["max_mean_abs_dce"] = 1e-9
    rep = run_gate(gate_root)
    # (another pilot_cfg makes POWER.json stale, so A4 is pending: G0 is what this test is about)
    assert TJ._item(rep, "merge_parity")["ok"] is True and rep["G0"]["status"] == "pass"
    r = rep["reported_not_gated"]["merge_bf16"]["per_eval"]["Q3S/ALL"]
    assert r["reply_agreement"] == 0.5 and r["max_mean_abs_dce"] == 1e-9 and r["within_bf16_thresholds"] is False
    # a parity json inside the old thresholds is reported as such (and still only reported)
    for p in L.pilot_root.glob("*/parity/merge_*.json"):
        TJ._edit(p, lambda r: r.update(reply_agreement=1.0))
    cfg["gates"]["merge_parity"]["max_mean_abs_dce"] = 0.5
    rep = run_gate(gate_root)
    assert rep["reported_not_gated"]["merge_bf16"]["per_eval"]["E2/0W"]["within_bf16_thresholds"] is True


def test_the_constant_and_its_registered_source():
    assert G.MERGE_MF_MAX_MEAN_ABS_DCE == 1e-4
    src = inspect.getsource(G)
    i = src.index("MERGE_MF_MAX_MEAN_ABS_DCE = 1e-4")
    assert "section 17" in src[max(0, i - 600):i] and "2026-11-11" in src[max(0, i - 600):i]
    # the threshold is not a pilot_cfg value (pilot_cfg.py is in the run_tvr harness fingerprint)
    assert "1e-4" not in json.dumps(pilot_cfg.get("gates.merge_parity"))


def test_missing_registry(gate_root):
    L, cfg, sv, _ = gate_root
    (L.pilot_root / G.MC_REGISTRY).unlink()
    probs = fails_alone_with(run_gate(gate_root), "no merge-correctness registry")
    assert any("no merge-correctness report entry" in b for b in probs)
    (L.pilot_root / G.MC_REGISTRY).write_text("not json")
    fails_alone_with(run_gate(gate_root), "no merge-correctness registry")


def test_malformed_registry(gate_root):
    L, cfg, sv, _ = gate_root
    reg = L.pilot_root / G.MC_REGISTRY
    TJ._edit(reg, lambda r: r.update(schema="other/1"))
    fails_alone_with(run_gate(gate_root), "schema 'other/1'")
    TJ._edit(reg, lambda r: r.update(schema=G.MC_REGISTRY_SCHEMA, reports=[]))
    fails_alone_with(run_gate(gate_root), "no reports listed")
    TJ.write_registry(L, report_paths(L))
    TJ._edit(reg, lambda r: r["reports"].append({"path": str(L.root / "nowhere.json"), "sha256": "0" * 64}))
    fails_alone_with(run_gate(gate_root), "nowhere.json missing")
    TJ._edit(reg, lambda r: r.update(reports=r["reports"][:2] + [r["reports"][0]]))
    fails_alone_with(run_gate(gate_root), "listed twice")


def test_report_sha_mismatch(gate_root):
    L, cfg, sv, _ = gate_root
    p = edit_entry(L, "E2_100", lambda a: a["diag"].update(note="edited after registration"), reregister=False)
    probs = fails_alone_with(run_gate(gate_root), f"report {p} sha256")
    # the changed report contributes no entry: its adapters have no evidence
    assert any("E2/100: no merge-correctness report entry" in b for b in probs)
    TJ.write_registry(L, report_paths(L))
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True


def test_no_entry(gate_root):
    L, cfg, sv, _ = gate_root
    p = edit_entry(L, "E2_100", lambda a: None)
    doc = json.loads(p.read_text())
    doc["adapters"].pop("E2_100")
    p.write_text(json.dumps(doc))
    TJ.write_registry(L, report_paths(L))
    fails_alone_with(run_gate(gate_root), "E2/100: no merge-correctness report entry with start E2, tier 100")


def test_two_entries(gate_root):
    L, cfg, sv, _ = gate_root
    doc = json.loads(report_paths(L)[0].read_text())
    third = TJ.wj(L.runs / "explore" / "mpdiag" / "r3" / "mp_diag.json",
                  {"schema": G.MP_DIAG_REPORT_SCHEMA, "order": ["Q3B_0W"], "adapters": {"Q3B_0W": doc["adapters"][
                      "Q3B_0W"]}})
    TJ.write_registry(L, report_paths(L))
    assert third in report_paths(L)
    fails_alone_with(run_gate(gate_root), "Q3B/0W: 2 merge-correctness report entries")


def test_adapter_sha_mismatch(gate_root):
    L, cfg, sv, _ = gate_root
    edit_entry(L, "Q3S_100", lambda a: a["diag"].update(adapter_sha256="1" * 64))
    fails_alone_with(run_gate(gate_root), "Q3S/100: report adapter_sha256 111111111111")
    edit_entry(L, "Q3S_100", lambda a: (a["diag"].update(adapter_sha256=TJ.fsha(L.adapter("Q3S", "100") /
                                                                                 "adapter.pt")),
                                        a["weights"].update(adapter_sha256="2" * 64)))
    fails_alone_with(run_gate(gate_root), "Q3S/100: report weights adapter_sha256 222222222222")


def test_reproduction_false_or_of_another_parity_json(gate_root):
    L, cfg, sv, _ = gate_root
    edit_entry(L, "E2_0W", lambda a: a["diag"]["gate_reproduction"].update(ok=False))
    fails_alone_with(run_gate(gate_root), "E2/0W: gate_reproduction.ok False")
    edit_entry(L, "E2_0W", lambda a: a["diag"].update(gate_reproduction={"ok": None, "reason": "no gate json"}))
    fails_alone_with(run_gate(gate_root), "E2/0W: gate_reproduction.ok None")
    edit_entry(L, "E2_0W", lambda a: a["diag"]["gate_reproduction"].update(
        ok=True, gate=dict(a["diag"]["pairs"]["U-M"])))
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True
    # the eval job's parity json is not the one the diagnostic reproduced (rewritten since): mean and agreeing count
    TJ._edit(L.parity("E2", "0W"), lambda r: r.update(mean_abs_dce=0.0134))
    fails_alone_with(run_gate(gate_root), "E2/0W: gate_reproduction.gate.mean_abs_dce 0.0133 != merge_0W.json 0.0134")
    TJ._edit(L.parity("E2", "0W"), lambda r: r.update(mean_abs_dce=0.0133, replies_agreeing=253))
    fails_alone_with(run_gate(gate_root), "gate_reproduction.gate.replies_agreeing 254 != merge_0W.json 253")


def test_u_m_must_equal_the_parity_json_exactly(gate_root):
    """DECISIONS G16: the diagnostic's own U-M equals the eval job's parity json exactly.  mp_diag's verdict
    gate_reproduction.ok allows |d mean| <= 1e-6, so ok True with gate == the parity json is not enough."""
    L, cfg, sv, _ = gate_root
    par = json.loads(L.parity("Q3S", "100").read_text())

    def um(a):
        return a["diag"]["pairs"]["U-M"]

    # ok True and gate == the parity json, but the diagnostic's U-M is 5e-7 off (inside mp_diag's tolerance)
    edit_entry(L, "Q3S_100", lambda a: um(a).update(mean_abs_dce=par["mean_abs_dce"] + 5e-7))
    probs = fails_alone_with(run_gate(gate_root), "Q3S/100: U-M mean_abs_dce 0.01330")
    assert any("!= merge_100.json 0.0133 (" in b for b in probs), probs
    assert not [b for b in probs if "gate_reproduction" in b or not b.startswith("Q3S/100:")], probs
    f = TJ._item(run_gate(gate_root), "merge_parity")["detail"]["facts"]["per_eval"]["Q3S/100"]
    assert f["u_m_equals_parity_json"] is False and f["gate_reproduction_ok"] is True
    assert "| Q3S/100 | r2#Q3S_100 | True / False |" in G.render_md(run_gate(gate_root))
    edit_entry(L, "Q3S_100", lambda a: um(a).update(mean_abs_dce=par["mean_abs_dce"]))
    rep = run_gate(gate_root)
    assert TJ._item(rep, "merge_parity")["ok"] is True
    assert TJ._item(rep, "merge_parity")["detail"]["facts"]["per_eval"]["Q3S/100"]["u_m_equals_parity_json"] is True
    # the agreeing count and the reply count; a missing value or a bool never counts as equal
    for upd, needle in (({"replies_agreeing": 253}, "Q3S/100: U-M replies_agreeing 253 != merge_100.json 254"),
                        ({"n_replies": 266}, "Q3S/100: U-M n_replies 266 != merge_100.json 267"),
                        ({"replies_agreeing": None}, "Q3S/100: U-M replies_agreeing None != merge_100.json 254"),
                        ({"mean_abs_dce": True}, "Q3S/100: U-M mean_abs_dce True")):
        edit_entry(L, "Q3S_100", lambda a: um(a).update(**upd))
        fails_alone_with(run_gate(gate_root), needle)
        edit_entry(L, "Q3S_100", lambda a: um(a).update(**{k: par[k] for k in upd}))
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True
    # no U-M pair at all
    edit_entry(L, "Q3S_100", lambda a: a["diag"]["pairs"].pop("U-M"))
    fails_alone_with(run_gate(gate_root), "Q3S/100: no U-M pair in the report")


def test_served_ok_false(gate_root):
    L, cfg, sv, _ = gate_root
    edit_entry(L, "Q3B_ALL", lambda a: a["weights"].update(served_ok=False))
    fails_alone_with(run_gate(gate_root), "Q3B/ALL: served_ok False")
    edit_entry(L, "Q3B_ALL", lambda a: a.update(weights=None))
    probs = fails_alone_with(run_gate(gate_root), "Q3B/ALL: served_ok None")
    assert any("weights adapter_sha256 None" in b for b in probs)


def test_t_mf_threshold(gate_root):
    L, cfg, sv, _ = gate_root
    edit_entry(L, "E2_ALL", lambda a: a["diag"]["pairs"]["T-MF"].update(mean_abs_dce=1.5e-4))
    fails_alone_with(run_gate(gate_root), "E2/ALL: T-MF mean |dCE| 0.00015 > 0.0001")
    edit_entry(L, "E2_ALL", lambda a: a["diag"]["pairs"]["T-MF"].update(mean_abs_dce=1e-4))  # <= passes
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True
    edit_entry(L, "E2_ALL", lambda a: a["diag"]["pairs"].pop("T-MF"))
    fails_alone_with(run_gate(gate_root), "E2/ALL: no T-MF mean_abs_dce")
    # on the parity json's 267 replies, no demo dropped
    edit_entry(L, "E2_ALL", lambda a: a["diag"]["pairs"].update(**{"T-MF": {
        "mean_abs_dce": 1e-6, "n_replies": 250, "demos_dropped": 1}}))
    fails_alone_with(run_gate(gate_root), "E2/ALL: T-MF on 250 replies, 1 demos dropped")


def test_module_mismatch(gate_root):
    L, cfg, sv, _ = gate_root
    for upd, needle in (({"n_wrapped_not_merged": 1}, "wrapped-not-merged 1"),
                        ({"n_merged_not_wrapped": 2}, "merged-not-wrapped 2"),
                        ({"merge_keys": 251}, "merge keys 251"),
                        ({"wrapped": None}, "wrapped None")):
        edit_entry(L, "Q3S_0W", lambda a: a["diag"]["facts"]["modules"].update(**upd))
        fails_alone_with(run_gate(gate_root), "Q3S/0W: module sets")
        assert any(needle in b for b in TJ._item(run_gate(gate_root), "merge_parity")["detail"]["problems"])
        edit_entry(L, "Q3S_0W", lambda a: a["diag"]["facts"]["modules"].update(
            wrapped=252, merge_keys=252, n_wrapped_not_merged=0, n_merged_not_wrapped=0))
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True
    edit_entry(L, "Q3S_0W", lambda a: a["diag"].pop("facts"))
    fails_alone_with(run_gate(gate_root), "Q3S/0W: module sets wrapped None")


def test_replicate_name_mapping(gate_root):
    """The replicate's eval / parity name is 100s1002, its adapter dir (the diagnostic's tier) 100_s1002."""
    L, cfg, sv, _ = gate_root
    assert PT.EVAL_ADAPTER["100s1002"] == "100_s1002" and L.parity("Q3B", "100s1002").name == "merge_100s1002.json"
    edit_entry(L, "Q3B_100_s1002", lambda a: a["diag"].update(tier="100s1002"))
    fails_alone_with(run_gate(gate_root), "Q3B/100s1002 (adapter 100_s1002): no merge-correctness report entry with "
                                         "start Q3B, tier 100_s1002")
    # the seed-1001 100 entry is never taken for the replicate (another adapter sha anyway)
    edit_entry(L, "Q3B_100_s1002", lambda a: a["diag"].update(tier="100_s1002"))
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True
    edit_entry(L, "Q3B_100", lambda a: a["diag"].update(tier="100_s1002"))
    fails_alone_with(run_gate(gate_root), "Q3B/100s1002 (adapter 100_s1002): 2 merge-correctness report entries")


def test_missing_parity_json_is_incomplete_and_other_adapter_fails(gate_root):
    L, cfg, sv, _ = gate_root
    p = L.parity("E2", "ALL")
    orig = p.read_text()
    TJ._edit(p, lambda r: r.update(adapter_sha256="3" * 64))
    fails_alone_with(run_gate(gate_root), "merge_ALL.json adapter sha 333333333333")
    p.unlink()
    rep = run_gate(gate_root)
    it = TJ._item(rep, "merge_parity")
    assert it["ok"] is None and "E2: merge_ALL.json" in it["detail"]["missing"] and rep["decision"] == "incomplete"
    assert rep["reported_not_gated"]["merge_bf16"]["per_eval"]["E2/ALL"]["within_bf16_thresholds"] is None
    p.write_text(orig)
    assert TJ._item(run_gate(gate_root), "merge_parity")["ok"] is True


def test_a_report_of_another_schema_or_shape_never_crashes(gate_root):
    L, cfg, sv, _ = gate_root
    p = report_paths(L)[1]
    p.write_text(json.dumps({"schema": "vpd.tvr.mp_diag/1", "adapters": {}}))
    TJ.write_registry(L, report_paths(L))
    fails_alone_with(run_gate(gate_root), "is not a vpd.tvr.mp_diag/1:report file")
    p.write_text(json.dumps({"schema": G.MP_DIAG_REPORT_SCHEMA, "adapters": {"E2_100": {"diag": "x", "weights": 3},
                                                                           "junk": None}}))
    TJ.write_registry(L, report_paths(L))
    fails_alone_with(run_gate(gate_root), "E2/100: no merge-correctness report entry")

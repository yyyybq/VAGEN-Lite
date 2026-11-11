"""vpd/tvr/compare_tvr.py: the report-only comparison registered in PREREG_TVR_PILOT.md section 17 (row 2026-11-11).

Synthetic pilot roots only (N_TASKS registered tasks, EVAL500_N monkeypatched); never the default pilot root.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from vpd.tvr import compare_tvr as C
from vpd.tvr import pilot_cfg
from vpd.tvr import pilot_tvr as PT

N_TASKS = 24
CATS = (("ithor", "easy"), ("ithor", "hard"), ("procthor", "easy"), ("procthor", "hard"))
IDS = [f"t{i:03d}" for i in range(N_TASKS)]
# successes (task indices) per eval; every other eval of the 16 succeeds nowhere
SUCC = {("Q3S", "0W", "c"): set(range(12)), ("Q3B", "0W", "c"): {10, 11, 12, 13}, ("E2", "0W", "c"): {0, 1, 2},
        ("Q3S", "100", "c"): set(range(6)), ("E2", "100", "c"): {6},
        ("Q3S", "ALL", "c"): set(range(9)), ("Q3B", "ALL", "c"): {8, 9},
        ("Q3S", "ALL", "u"): {0, 1}, ("Q3B", "ALL", "u"): {0, 1}, ("E2", "ALL", "u"): {0, 1, 2},
        ("Q3S", "ALL", "blind"): set(range(5)),
        ("Q3B", "100s1002", "c"): {0, 1, 2}}


def wj(p: Path, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return p


def all_evals():
    out = [(s, t, a) for s in pilot_cfg.STARTS for t, a in C.cells()]
    return out + [("Q3B", "100s1002", "c")]


def write_eval(root: Path, key, succ: set, ids=IDS) -> Path:
    s, tier, arm = key
    d = C.eval_dir(root, s, tier, arm)
    for i, tid in enumerate(ids):
        ds, diff = CATS[i % 4]
        wj(d / "tasks" / tid / "trajectory.json", {"task_id": tid, "dataset": ds, "difficulty": diff,
                                                    "result": {"success": i in succ, "stopped": True}})
    (d / "per_task.jsonl").write_text("".join(json.dumps({"task_id": tid, "success": int(i in succ)}) + "\n"
                                              for i, tid in enumerate(ids)))
    wj(d / "results.json", {"valid": True, "n_tasks": len(ids), "sr": {"k": len([i for i in succ if i < len(ids)]),
                                                                      "n": len(ids)}})
    return d


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(PT, "EVAL500_N", N_TASKS)
    monkeypatch.setattr(PT, "prereg_state", lambda: {"commit": "c0ffee"})
    root = tmp_path / "pilot"
    tasks_dir = tmp_path / "repo" / "data" / "tasks"
    wj(tasks_dir / "eval.json", [{"task_id": t, "dataset": CATS[i % 4][0], "difficulty": CATS[i % 4][1]}
                                 for i, t in enumerate(IDS)])
    for key in all_evals():
        write_eval(root, key, SUCC.get(key, set()))
    wj(root / "GATE.json", {"decision": "ruler", "G0": {"status": "pass"}, "A4": {"status": "pass"},
                            "note": "no between-start differences are computed"})
    wj(root / "POWER.json", {"pass": True, "fallback_pass": False})
    return root, tasks_dir


def run(world, *extra):
    root, tasks_dir = world
    return C.main(["--pilot-root", str(root), "--tasks-dir", str(tasks_dir), *extra])


def load(world):
    return json.loads((world[0] / "COMPARE.json").read_text(encoding="utf-8"))


# --- statistics -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("c,b_min", [(0, 6), (1, 8), (2, 10)])
def test_exact_p_thresholds_of_prereg_12_2(c, b_min):
    """PREREG 12.2: c = 0 needs b >= 6, c = 1 needs b >= 8, c = 2 needs b >= 10 for p < 0.05."""
    assert C.exact_p(b_min, c) < Fraction(1, 20) <= C.exact_p(b_min - 1, c)


def test_exact_p_values():
    assert C.exact_p(6, 0) == Fraction(1, 32) and C.exact_p(8, 1) == Fraction(20, 512)
    assert C.exact_p(10, 2) == Fraction(2 * (1 + 12 + 66), 4096) and C.exact_p(2, 10) == C.exact_p(10, 2)
    assert C.exact_p(0, 0) == 1 and C.exact_p(3, 3) == 1 and C.exact_p(1, 0) == 1


def test_wilson_examples_of_prereg_12_1():
    lb3, lb2 = C.wilson(3, 500, 1.96)[0], C.wilson(2, 500, 1.96)[0]
    assert round(lb3, 5) == 0.00204 and round(lb2, 4) == 0.0011
    assert C.wilson(0, 500, 1.96)[0] == pytest.approx(0.0, abs=1e-15)


# --- the report -------------------------------------------------------------------------------------------------

def test_report_cells_reference_and_files(world):
    assert run(world) == 0
    rep = load(world)
    assert rep["schema"] == "vpd.tvr.compare/1" and rep["multiplicity_correction"] == "none" and rep["z"] == 1.96
    assert rep["cells"] == ["0W__c", "100__c", "ALL__c", "ALL__u", "ALL__blind"]
    assert C.ROW_NEEDLE in rep["prereg"]["row_text"] and rep["prereg"]["row_text"].startswith("| 2026-11-11 |")
    root = world[0]
    assert rep["gate"]["sha256"] == C.sha256_file(root / "GATE.json") and rep["gate"]["decision"] == "ruler"
    assert rep["power"]["sha256"] == C.sha256_file(root / "POWER.json") and rep["power"]["pass"] is True
    assert rep["registered_tasks"]["n"] == N_TASKS and rep["crosscheck"] == {
        "ok": True, "mismatches": [], "not_checked": [], "used_sides_all_checked": True}
    assert {r["results_file"] for r in rep["per_start"].values()} == {"results.json"}
    assert "cats" not in rep["registered_tasks"] and "ids" not in rep["registered_tasks"]
    pri, sup = rep["comparisons"]["Q3S-Q3B"], rep["comparisons"]["Q3S-E2"]
    assert pri["role"] == "primary" and sup["role"] == "supplement" and pri["b_is"] == "only Q3S succeeds"
    want = {"Q3S-Q3B": {"0W__c": (12, 4, 10, 2), "100__c": (6, 0, 6, 0), "ALL__c": (9, 2, 8, 1), "ALL__u": (2, 2, 0, 0),
                        "ALL__blind": (5, 0, 5, 0)},
            "Q3S-E2": {"0W__c": (12, 3, 9, 0), "100__c": (6, 1, 6, 1), "ALL__c": (9, 0, 9, 0), "ALL__u": (2, 3, 0, 1),
                       "ALL__blind": (5, 0, 5, 0)}}
    for name, cells in want.items():
        for cell, (k1, k2, b, c) in cells.items():
            x = rep["comparisons"][name]["cells"][cell]
            p = C.exact_p(b, c)
            assert (x["k_first"], x["k_second"], x["b"], x["c"], x["n"]) == (k1, k2, b, c, N_TASKS), (name, cell)
            assert x["p"] == float(p) and x["p_exact"] == f"{p.numerator}/{p.denominator}" and not x["missing"]
            assert x["rate_difference"] == (k1 - k2) / N_TASKS
            assert x["wilson95_first"] == C.wilson(k1, N_TASKS, 1.96)
            assert x["both"] + x["neither"] + b + c == N_TASKS
    assert rep["comparisons"]["Q3S-Q3B"]["cells"]["0W__c"]["p"] == 158 / 4096
    ref = rep["reference"]
    assert (ref["first"], ref["second"]) == ({"tier": "100", "seed": 1001}, {"tier": "100s1002", "seed": 1002})
    assert (ref["pair"]["b"], ref["pair"]["c"], ref["pair"]["p"]) == (0, 3, 0.25)
    assert rep["per_start"]["Q3B/100s1002__c"]["k"] == 3 and len(rep["per_start"]) == 16
    d = str(C.eval_dir(root, "E2", "ALL", "blind"))
    assert rep["inputs"][d]["n_trajectory_files"] == N_TASKS and rep["inputs"][d]["results_json_sha256"]
    md = (root / "COMPARE.md").read_text(encoding="utf-8")
    assert "## Q3S 对 Q3B（主报告）" in md and "## Q3S 对 E2（补充）" in md and "参照行" in md
    assert "| 0W | c | 12/24" in md and "偏乐观" in md and "推断" in md and "全部一致" in md
    assert "_success" not in json.dumps(rep)


def test_out_option(world, tmp_path):
    out = tmp_path / "elsewhere" / "C.json"
    assert run(world, "--out", str(out)) == 0
    assert out.is_file() and out.with_suffix(".md").is_file() and not (world[0] / "COMPARE.json").exists()


@pytest.mark.parametrize("name", ["GATE.json", "POWER.json"])
def test_refuses_without_gate_or_power(world, name, capsys):
    (world[0] / name).unlink()
    assert run(world) == 2
    assert "refused" in capsys.readouterr().err and not (world[0] / "COMPARE.json").exists()


def test_refuses_without_the_registered_row_or_task_set(world, tmp_path):
    pre = tmp_path / "PREREG.md"
    pre.write_text("## 17. 偏差日志\n| 2026-11-10 | something else |\n", encoding="utf-8")
    assert run(world, "--prereg", str(pre)) == 2
    wj(world[1] / "eval.json", [{"task_id": t} for t in IDS[:-1]])
    assert run(world) == 2


def test_missing_cells_are_never_imputed(world):
    root = world[0]
    import shutil

    shutil.rmtree(C.eval_dir(root, "E2", "100", "c"))
    # a side with one registered task missing, one with an unregistered extra task, one with success null
    # (each side self-consistent: its per_task.jsonl / results.json agree with its trajectories)
    shutil.rmtree(C.eval_dir(root, "Q3B", "ALL", "u"))
    write_eval(root, ("Q3B", "ALL", "u"), {0, 1}, ids=IDS[1:])
    write_eval(root, ("Q3B", "ALL", "blind"), set(), ids=IDS + ["x999"])
    p = C.eval_dir(root, "E2", "ALL", "c") / "tasks" / IDS[3] / "trajectory.json"
    t = json.loads(p.read_text())
    t["result"]["success"] = None
    p.write_text(json.dumps(t))
    assert run(world) == 3
    rep = load(world)
    cells = {(n, k): c for n, comp in rep["comparisons"].items() for k, c in comp["cells"].items()}
    for key in (("Q3S-E2", "100__c"), ("Q3S-Q3B", "ALL__u"), ("Q3S-Q3B", "ALL__blind"), ("Q3S-E2", "ALL__c")):
        c = cells[key]
        assert c["missing"] is True and "b" not in c and "p" not in c and "k_first" not in c, key
    assert any("no trajectories" in w for w in cells[("Q3S-E2", "100__c")]["why"])
    assert any("1 registered missing" in w for w in cells[("Q3S-Q3B", "ALL__u")]["why"])
    assert any("1 unregistered" in w for w in cells[("Q3S-Q3B", "ALL__blind")]["why"])
    assert any("bool result.success" in w for w in cells[("Q3S-E2", "ALL__c")]["why"])
    assert cells[("Q3S-Q3B", "0W__c")]["missing"] is False and len(rep["missing_cells"]) == 4
    assert rep["per_start"]["E2/100__c"]["usable"] is False and "k" not in rep["per_start"]["E2/100__c"]
    md = (root / "COMPARE.md").read_text(encoding="utf-8")
    assert "缺的格" in md and "没有结果文件" not in md  # an unusable side is not in the 'numbers from trajectories' note
    assert rep["crosscheck"]["used_sides_all_checked"] is True and any(
        u.startswith("E2/100__c: results_json: no results.json (pilot__E2__100__c)") for u in rep["crosscheck"][
            "not_checked"])


def test_duplicate_trajectory_makes_the_cell_missing(world):
    root = world[0]
    d = C.eval_dir(root, "Q3S", "0W", "c")
    t = json.loads((d / "tasks" / IDS[0] / "trajectory.json").read_text())
    wj(d / "tasks" / "copy" / "trajectory.json", t)
    # exit 1, not 3: the copy also sits in a dir that is not its task_id (the task_dirs cross-check)
    assert run(world) == 1
    rep = load(world)
    c = rep["comparisons"]["Q3S-Q3B"]["cells"]["0W__c"]
    assert c["missing"] and any("two trajectories" in w for w in c["why"])
    (m,) = rep["crosscheck"]["mismatches"]
    assert (m["eval_dir"], m["check"], m["n_misnamed"]) == ("pilot__Q3S__0W__c", "task_dirs", 1)
    assert "dirs ['copy']" in m["problems"][0] and len(rep["missing_cells"]) == 2


def test_trajectory_vs_per_task_and_results_mismatch(world):
    """Numbers come from the trajectories; a per_task.jsonl or results.json that disagrees is reported (exit 1)."""
    root = world[0]
    d = C.eval_dir(root, "Q3S", "ALL", "c")
    rows = [json.loads(x) for x in (d / "per_task.jsonl").read_text().splitlines()]
    rows[20]["success"] = 1
    (d / "per_task.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert run(world) == 1
    rep = load(world)
    assert rep["crosscheck"]["ok"] is False
    (m,) = rep["crosscheck"]["mismatches"]
    assert m["eval"] == "Q3S/ALL__c" and m["check"] == "per_task" and m["first"] == [IDS[20]]
    assert rep["comparisons"]["Q3S-Q3B"]["cells"]["ALL__c"]["k_first"] == 9  # from the trajectories
    assert "Q3S/ALL__c per_task 不一致" in (root / "COMPARE.md").read_text(encoding="utf-8")
    write_eval(root, ("Q3S", "ALL", "c"), SUCC[("Q3S", "ALL", "c")])
    r = json.loads((d / "results.json").read_text())
    r["sr"]["k"] = 10
    (d / "results.json").write_text(json.dumps(r))
    assert run(world) == 1
    (m,) = load(world)["crosscheck"]["mismatches"]
    assert (m["check"], m["k"], m["trajectories"]) == ("results_json", 10, [9, N_TASKS])
    # a per_task row missing for one task is a mismatch too
    write_eval(root, ("Q3S", "ALL", "c"), SUCC[("Q3S", "ALL", "c")])
    lines = (d / "per_task.jsonl").read_text().splitlines()
    (d / "per_task.jsonl").write_text("\n".join(lines[1:]) + "\n")
    assert run(world) == 1 and load(world)["crosscheck"]["mismatches"][0]["first"] == [IDS[0]]


def test_duplicate_per_task_rows_are_reported(world):
    """A duplicate per_task.jsonl row is a mismatch whatever its value; a copy that contradicts result.success is
    named in first even when the last copy agrees (a later row never hides an earlier one)."""
    root = world[0]
    d = C.eval_dir(root, "E2", "ALL", "c")
    rows = (d / "per_task.jsonl").read_text().splitlines()
    wrong = dict(json.loads(rows[5]), success=1)  # E2 ALL c succeeds nowhere
    (d / "per_task.jsonl").write_text("\n".join([json.dumps(wrong)] + rows) + "\n")
    assert run(world) == 1
    rep = load(world)
    (m,) = rep["crosscheck"]["mismatches"]
    assert (m["eval"], m["eval_dir"], m["check"]) == ("E2/ALL__c", "pilot__E2__ALL__c", "per_task")
    assert (m["n_rows"], m["n_task_ids"], m["n_duplicate_rows"], m["duplicate_ids"]) == (25, 24, 1, [IDS[5]])
    assert m["first"] == [IDS[5]] and m["n_mismatch"] == 1 and len(m["problems"]) == 2
    assert rep["comparisons"]["Q3S-E2"]["cells"]["ALL__c"]["k_second"] == 0  # from the trajectories
    # a plain duplicate (same value) is reported too
    (d / "per_task.jsonl").write_text("\n".join(rows + [rows[7]]) + "\n")
    assert run(world) == 1
    (m,) = load(world)["crosscheck"]["mismatches"]
    assert (m["n_duplicate_rows"], m["duplicate_ids"], m["n_mismatch"], m["first"]) == (1, [IDS[7]], 0, [])
    assert m["problems"] == [f"1 duplicate per_task.jsonl row(s), task ids e.g. {[IDS[7]]}"]
    assert "E2/ALL__c per_task 不一致（1 duplicate per_task.jsonl row(s)" in (root / "COMPARE.md").read_text(
        encoding="utf-8")


def test_a_used_side_without_results_json_exits_1(world):
    """An eval with all its trajectories but only results_invalid.json (score_tvr judged it invalid): its cells are
    computed from the trajectories, its results.json cross-check is listed as not made, and the exit is 1."""
    root = world[0]
    d = C.eval_dir(root, "E2", "ALL", "blind")
    (d / "results.json").rename(d / "results_invalid.json")
    assert run(world) == 1
    rep = load(world)
    cc = rep["crosscheck"]
    assert cc["ok"] is True and cc["mismatches"] == [] and cc["used_sides_all_checked"] is False
    assert cc["not_checked"] == ["E2/ALL__blind: results_json: no results.json (results_invalid.json present: "
                                 "score_tvr did not score this eval as valid) (pilot__E2__ALL__blind)"]
    assert rep["per_start"]["E2/ALL__blind"]["results_file"] == "results_invalid.json"
    assert rep["comparisons"]["Q3S-E2"]["cells"]["ALL__blind"]["missing"] is False
    ins = rep["inputs"][str(d)]
    assert ins["results_json_sha256"] is None and ins["results_invalid_json_sha256"] == C.sha256_file(
        d / "results_invalid.json")
    md = (root / "COMPARE.md").read_text(encoding="utf-8")
    assert "没核对：E2/ALL__blind: results_json" in md and "E2/ALL__blind（results_invalid.json）" in md
    # a side whose cells are missing anyway does not turn exit 3 into 1 (test_missing_cells_are_never_imputed)


def test_task_dirs_are_cross_checked(world):
    root = world[0]
    d = C.eval_dir(root, "Q3S", "100", "c")
    (d / "tasks" / "stray").mkdir()
    assert run(world) == 1
    (m,) = load(world)["crosscheck"]["mismatches"]
    assert (m["eval_dir"], m["check"], m["n_dirs_without_trajectory"]) == ("pilot__Q3S__100__c", "task_dirs", 1)
    (d / "tasks" / "stray").rmdir()
    # a trajectory in a dir not named by its task_id: the same task ids, so the cell is still computed
    (d / "tasks" / IDS[2]).rename(d / "tasks" / f"{IDS[2]}__retry1")
    assert run(world) == 1
    rep = load(world)
    (m,) = rep["crosscheck"]["mismatches"]
    assert m["check"] == "task_dirs" and m["n_misnamed"] == 1 and f"{IDS[2]}__retry1" in m["problems"][0]
    assert rep["comparisons"]["Q3S-Q3B"]["cells"]["100__c"]["missing"] is False


def test_results_json_by_category_is_cross_checked(world):
    root = world[0]
    d = C.eval_dir(root, "Q3S", "0W", "c")
    succ = SUCC[("Q3S", "0W", "c")]
    byc = {}
    for i, _ in enumerate(IDS):
        c = pilot_cfg.category_of({"dataset": CATS[i % 4][0], "difficulty": CATS[i % 4][1]})
        e = byc.setdefault(c, {"sr": {"k": 0, "n": 0}})
        e["sr"]["k"] += int(i in succ)
        e["sr"]["n"] += 1
    r = json.loads((d / "results.json").read_text())
    wj(d / "results.json", dict(r, by_category=byc))
    assert run(world) == 0
    # the overall sr unchanged, two categories off by one
    a, b = sorted(byc)[:2]
    byc[a]["sr"]["k"] += 1
    byc[b]["sr"]["k"] -= 1
    wj(d / "results.json", dict(r, by_category=byc))
    assert run(world) == 1
    (m,) = load(world)["crosscheck"]["mismatches"]
    assert (m["check"], m["k"], m["trajectories"], m["by_category_checked"]) == ("results_json", 12, [12, N_TASKS],
                                                                                True)
    assert [p.split(" sr ")[0] for p in m["problems"]] == [f"by_category {a}", f"by_category {b}"]


def test_never_reads_the_default_pilot_root(world, monkeypatch):
    """The default root is pilot_tvr.LAYOUT.pilot_root; tests always pass --pilot-root (this one checks the default
    resolution without reading it: a missing GATE.json there refuses before anything else is read)."""
    fake = world[0].parent / "no_pilot_here"
    monkeypatch.setattr(PT.LAYOUT, "pilot_root", fake)
    with pytest.raises(C.Refused):
        C.compare(tasks_dir=world[1])

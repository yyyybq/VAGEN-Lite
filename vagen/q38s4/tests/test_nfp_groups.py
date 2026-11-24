"""q38s4 tools.nfp_groups (PREREG_v1 v1.2): the fixed groups (against the PREREG text too), the check against
summary.json, distinct windows, the group numbers by hand, the code guard on a real git repo, the command line's
refusals and that a failed check computes nothing."""
from __future__ import annotations

import importlib.machinery
import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vagen.q38s4.lf_video import nfp_indices
from vagen.q38s4.tools import nfp_groups as G
from vagen.q38s4.tools import nfp_heldout as H

V = "/d/VSI-590K"
U = (0.125, 0.375, 0.625, 0.875)
LONG, SHORT = (9000, 30.0), (1800, 30.0)  # 300 s and 60 s at 30 fps: 600 and 120 grid frames at 2 fps
MEDIA = {"video_fps": 2.0, "video_maxlen": 256}
PREREG = G.ROOT / "docs/q38s4/PREREG_v1.md"


def _names():
    return {
        "scannet": ["scene0433_00", "scene0617_00"] + [f"scene{1000 + i}_00" for i in range(12)],
        "procthor": ["001082/raw_navigation_camera__1", "001224/raw_navigation_camera__0",
                     "001082/raw_navigation_camera__0", "000001/raw_navigation_camera__0",
                     "000002/raw_navigation_camera__1"],  # another camera of house 001082 is not the listed one
        "s3dis": ["Area_3_hallway_6", "Area_4_conferenceRoom_2", "Area_4_hallway_11"],
        "scannetppv2": ["05d3c8c6f0", "0caa1ae59a", "192ab15daf", "3c8d535d49", "4e2db5058c", "52599ae063",
                        "5b38982d25", "5c21213f16", "9ad3639fbe", "c842edbdf5", "eee7eef8e5"],
        "adt": ["ADT_a_preview_rgb", "ADT_b_preview_rgb"],
        "arkitscenes": [f"4{i:07d}" for i in range(25)],
    }


def _videos(names=None):
    names = names or _names()
    return [f"{V}/{src}/{n}.mp4" for src in sorted(names) for n in names[src]]


def _header_of(videos):
    """Every other video long (4 distinct windows for the 4 u), the rest short (one window: span 0)."""
    table = {v: (LONG if i % 2 == 0 else SHORT) for i, v in enumerate(sorted(videos))}
    return lambda v: table[v]


def _rows(videos, header):
    recs = []
    for i, v in enumerate(sorted(videos)):
        total, avg = header(v)
        for j, u in enumerate(U):
            pairs = 100 + 10 * (i % 3) + j
            copy, head, rand = 0.3 + 0.01 * (i % 7), 0.3 * (0.5 + 0.05 * (i % 5)) + 0.001 * j, 0.35
            recs.append({"row": len(recs), "video": v, "source": H.source_of(v), "u": u,
                         "units": len(nfp_indices(total, avg, u, MEDIA["video_fps"], MEDIA["video_maxlen"])) // 2,
                         "tokens_per_unit": 1, "pairs": pairs,
                         "copy": {"cos_sum": copy * pairs, "mse_sum": 1.0 * pairs, "unit_cos": []},
                         "head": {"cos_sum": head * pairs, "mse_sum": 0.5 * pairs, "unit_cos": []},
                         "random": {"cos_sum": rand * pairs, "mse_sum": 2.0 * pairs, "unit_cos": []}})
    return recs


@pytest.fixture
def data():
    vids = _videos()
    header = _header_of(vids)
    return vids, header, _rows(vids, header)


def _hand_r(sub):
    P = sum(r["pairs"] for r in sub)
    c, h = sum(r["copy"]["cos_sum"] for r in sub) / P, sum(r["head"]["cos_sum"] for r in sub) / P
    return (c - h) / c


# ------------------------------------------------------------------------------------------------ the fixed list
def test_the_fixed_lists_are_v12s():
    assert {k: len(v) for k, v in G.NONE_FOUND.items()} == {"scannet": 2, "procthor": 2, "s3dis": 3, "scannetppv2": 11}
    names = [n for v in G.NONE_FOUND.values() for n in v]
    assert len(names) == len(set(names)) == 18 == G.EXPECT_GROUPS["none_found"]
    assert sum(G.EXPECT_GROUPS.values()) == H.EXPECT_VIDEOS and sum(G.EXPECT_OVERLAP.values()) == 17
    assert G.UNDETERMINED_SOURCE == "arkitscenes" and G.EXPECT_WINDOWS == 128 and G.SPLITS == ("scannet", "procthor")
    assert G.ROWS_PER_VIDEO == 4 and G.CODE[0] == G.THIS and G.U_VALUES == U


@pytest.mark.skipif(not PREREG.is_file(), reason="PREREG_v1.md not in this checkout")
def test_the_names_are_the_ones_written_in_prereg_v12():
    lines = PREREG.read_text(encoding="utf-8").splitlines()
    v12 = next(i for i, l in enumerate(lines) if l.startswith("## v1.2"))
    at = next(i for i in range(v12, len(lines)) if "**没查到组**" in lines[i])
    got = {}
    for l in lines[at + 1:at + 5]:
        src, names = l.strip().lstrip("- ").split("：", 1)
        got[src] = tuple(n.strip() for n in names.rstrip("；。").split("、"))
    assert got == G.NONE_FOUND


# ------------------------------------------------------------------------------------------------ groups
def test_groups_of(data):
    vids, _, _ = data
    g = G.groups_of(vids)
    assert {k: sum(1 for v in g.values() if v == k) for k in G.GROUPS} == G.EXPECT_GROUPS
    assert g[f"{V}/scannet/scene0433_00.mp4"] == "none_found" and g[f"{V}/scannet/scene1003_00.mp4"] == "overlap"
    assert g[f"{V}/procthor/001082/raw_navigation_camera__1.mp4"] == "none_found"
    assert g[f"{V}/procthor/001082/raw_navigation_camera__0.mp4"] == "overlap"
    assert g[f"{V}/arkitscenes/40000003.mp4"] == "undetermined" and g[f"{V}/adt/ADT_a_preview_rgb.mp4"] == "overlap"
    assert all(g[v] == "none_found" for v in vids if H.source_of(v) in ("s3dis", "scannetppv2"))


def test_groups_refuse_a_missing_name_a_double_match_and_wrong_counts():
    n = _names()
    n["scannet"][0] = "scene0433_01"
    with pytest.raises(SystemExit, match="scannet scene0433_00: 0 held-out videos match"):
        G.groups_of(_videos(n))
    with pytest.raises(SystemExit, match="2 held-out videos match"):
        G.groups_of(_videos() + [f"{V}/procthor/x/001082/raw_navigation_camera__1.mp4"])
    n = _names()
    n["scannet"].pop()
    n["adt"].append("ADT_c_preview_rgb")  # sizes still 18/17/25
    with pytest.raises(SystemExit, match="overlap group by source"):
        G.groups_of(_videos(n))
    with pytest.raises(SystemExit, match="group sizes"):
        G.groups_of(_videos() + [f"{V}/arkitscenes/49999999.mp4"])
    n = _names()
    n["scannet"][0] = "scene0433_00_extra"  # a longer name is not the listed one
    with pytest.raises(SystemExit, match="0 held-out videos match"):
        G.groups_of(_videos(n))
    n = _names()
    n["scannet"][0], n["scannetppv2"][0] = "05d3c8c6f0", "scene0433_00"  # a name under another source
    with pytest.raises(SystemExit, match="0 held-out videos match"):
        G.groups_of(_videos(n))


# ------------------------------------------------------------------------------------------------ windows
def test_windows_and_the_first_row_of_each(data):
    vids, header, recs = data
    keys = G.window_keys(recs, MEDIA, header)
    assert len(keys) == 240 and all(k[0] == r["video"] for k, r in zip(keys, recs))
    long_v, short_v = sorted(vids)[0], sorted(vids)[1]
    assert len({k for k in keys if k[0] == long_v}) == 4 and len({k for k in keys if k[0] == short_v}) == 1
    assert len(keys[0][1]) == 256 and len(keys[4][1]) == 120  # 128 and 60 units
    first = G.first_per_key(keys)
    assert len(first) == 30 * 4 + 30 and first[:6] == [0, 1, 2, 3, 4, 8]
    calls = []
    G.window_keys(recs, MEDIA, lambda v: calls.append(v) or header(v))
    assert len(calls) == 60  # one header read per video
    recs[5]["units"] += 1
    with pytest.raises(SystemExit, match="row 5: 120 frames = 60 units, the row has 61"):
        G.window_keys(recs, MEDIA, header)


# ------------------------------------------------------------------------------------------------ the check
def _summary(recs, step=1000):
    s = json.loads(json.dumps(H.summarize(recs)))
    s["meta"] = {"model": f"/runs/formal/c_nfp/ckpt_bf16/checkpoint-{step}", "media": dict(MEDIA)}
    return s


def test_the_check_passes_on_its_own_numbers_and_stops_on_any_difference(data):
    _, _, recs = data
    s = _summary(recs)
    c = G.check_summary(recs, s)
    assert c["ok"] and c["max_abs_diff"] == 0.0 and c["compared"] == 1 + 2 + 2 + 6
    assert set(c) == {"ok", "compared", "max_abs_diff", "tolerance"}  # verdict 1's numbers are read from summary.json
    s2 = json.loads(json.dumps(s))
    s2["r"]["head"] += 1e-13  # within the tolerance
    assert 0 < G.check_summary(recs, s2)["max_abs_diff"] <= G.TOL
    for path, msg in ((("r", "head"), r"r\.head differs"), (("bootstrap", "head", "lo"), r"bootstrap\.head\.lo"),
                      (("by_source", "scannet", "r", "head"), r"by_source\.scannet\.r\.head"),
                      (("copy", "cos"), r"copy\.cos")):
        bad = json.loads(json.dumps(s))
        d = bad
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] += 1e-9
        with pytest.raises(SystemExit, match=msg):
            G.check_summary(recs, bad)
    for edit, msg in ((lambda b: b["bootstrap"].update(draws=1000), "draws 1000"),
                      (lambda b: b.update(pairs=b["pairs"] + 1), "rows/videos/pairs"),
                      (lambda b: b["by_source"].pop("adt"), "sources"),
                      (lambda b: b["r"].update(head=None), r"r\.head differs by inf"),
                      (lambda b: b["r"].update(head=float("nan")), r"r\.head differs by nan"),
                      (lambda b: (b["r"].update(head=float("nan")), b["copy"].update(cos=b["copy"]["cos"] + 1)),
                       r"r\.head differs by nan \(and 1 more\)")):
        bad = json.loads(json.dumps(s))
        edit(bad)
        with pytest.raises(SystemExit, match=msg):
            G.check_summary(recs, bad)


# ------------------------------------------------------------------------------------------------ numbers
def test_group_numbers_by_hand(data):
    vids, header, recs = data
    keys = G.window_keys(recs, MEDIA, header)
    res = G.compute(recs, keys, expect_windows=150)
    g = G.groups_of(vids)
    a = res["all_rows"]
    assert (a["videos"], a["rows"], a["windows"], a["pairs"]) == (60, 240, 150, sum(r["pairs"] for r in recs))
    assert a["r"] == H.pooled(recs)["r"]["head"] == pytest.approx(_hand_r(recs))
    assert (a["interval"]["lo"], a["interval"]["hi"]) == tuple(H.bootstrap(recs)["head"][k] for k in ("lo", "hi"))
    assert set(a) == {"videos", "rows", "windows", "pairs", "r", "nfp_cos", "copy_cos", "interval"}
    for name, n_vid in G.EXPECT_GROUPS.items():
        sub = [r for r in recs if g[r["video"]] == name]
        got = res["groups"][name]
        b = H.bootstrap(sub, draws=H.DRAWS, seed=H.SEED)
        assert (got["videos"], got["rows"], got["pairs"]) == (n_vid, 4 * n_vid, sum(r["pairs"] for r in sub))
        assert got["windows"] == len({k for k, r in zip(keys, recs) if g[r["video"]] == name})
        assert got["r"] == pytest.approx(_hand_r(sub)) and got["copy_cos"] == pytest.approx(
            sum(r["copy"]["cos_sum"] for r in sub) / got["pairs"])
        assert (got["interval"]["lo"], got["interval"]["hi"]) == (b["head"]["lo"], b["head"]["hi"])
        assert got["interval"]["videos"] == n_vid and got["interval"]["draws"] == 2000
        assert len(res["members"][name]) == n_vid
    assert res["groups"]["overlap"]["videos_by_source"] == G.EXPECT_OVERLAP
    assert "scannet/scene0433_00.mp4" in res["members"]["none_found"]
    first = G.first_per_key(keys)
    dw = res["distinct_windows"]
    assert (dw["rows"], dw["windows"], dw["videos"]) == (150, 150, 60)
    assert dw["pairs"] == sum(recs[i]["pairs"] for i in first) and dw["r"] == pytest.approx(
        _hand_r([recs[i] for i in first]))
    sc = res["splits"]["scannet"]
    assert (sc["overlap"]["videos"], sc["none_found"]["videos"]) == (12, 2)
    assert (res["splits"]["procthor"]["overlap"]["videos"], res["splits"]["procthor"]["none_found"]["videos"]) == (3, 2)
    assert "interval" not in sc["overlap"] and sc["none_found"]["r"] == pytest.approx(_hand_r(
        [r for r in recs if H.source_of(r["video"]) == "scannet" and g[r["video"]] == "none_found"]))
    assert "no conclusion" in res["splits"]["note"]
    with pytest.raises(SystemExit, match="150 distinct windows, expected 128"):
        G.compute(recs, keys)


# ------------------------------------------------------------------------------------------------ the code guard
def _run_git(d, *a):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *a], cwd=d,
                          check=True, capture_output=True, text=True).stdout.strip()


@pytest.mark.skipif(shutil.which("git") is None, reason="no git")
def test_provenance_and_the_code_check_on_a_real_repo(tmp_path):
    files = ("a.py", "b/c.py")
    d = tmp_path / "repo"
    (d / "b").mkdir(parents=True)
    _run_git(d, "init", "-q")
    (d / "README").write_text("x\n")
    _run_git(d, "add", "README")
    _run_git(d, "commit", "-q", "-m", "init")
    (d / "a.py").write_text("1\n")
    (d / "b/c.py").write_text("2\n")
    p = G.provenance(None, d, files)
    assert not p["committed"] and p["files"]["a.py"]["status"] == "?? a.py" and not p["files"]["a.py"]["tracked"]
    with pytest.raises(SystemExit, match=r"not committed and clean .*a\.py \(\?\? a\.py\)"):
        G.check_code(p, 2000, False)
    G.check_code(p, 1000, True)  # the check before the commit
    with pytest.raises(SystemExit, match="step 1000 only"):
        G.check_code(p, 2000, True)
    _run_git(d, "add", "a.py", "b/c.py")
    assert not G.provenance(None, d, files)["committed"]  # staged is not committed
    _run_git(d, "commit", "-q", "-m", "code")
    c1 = _run_git(d, "rev-parse", "HEAD")  # a.py's first commit: the one every committed run is tied to
    p = G.provenance(c1, d, files)
    assert p["committed"] and p["expect_commit"] == c1 and p["differ"] == [] and p["head"] == c1
    assert p["first_commit"] == c1 and p["files"]["b/c.py"]["blob_at_first"] == _run_git(d, "rev-parse", "HEAD:b/c.py")
    assert p["files"]["a.py"]["blob"] == _run_git(d, "rev-parse", "HEAD:a.py") and p["files"]["b/c.py"]["last_commit"] == c1
    assert p["files"]["a.py"]["md5_at_expect"] == p["files"]["a.py"]["md5"] and p["expect_commit_arg"] == c1
    G.check_code(p, 2000, False)
    for name in ("HEAD", "HEAD~0", "master", "main", c1.upper(), c1[:10], c1[:39]):  # a name follows HEAD; a short
        with pytest.raises(SystemExit, match="give the commit's full hash"):  # hash can lose to a branch of that name
            G.provenance(name, d, files)
    with pytest.raises(SystemExit, match="--expect-commit COMMIT is needed"):
        G.check_code(G.provenance(None, d, files), 2000, False)
    (d / "a.py").write_text("1 changed\n")
    with pytest.raises(SystemExit, match=r"a\.py \( M a\.py\)"):
        G.check_code(G.provenance(c1, d, files), 2000, False)
    _run_git(d, "commit", "-q", "-am", "change")
    c2 = _run_git(d, "rev-parse", "HEAD")
    with pytest.raises(SystemExit, match=re.escape(f"a.py differ from {c1[:8]} (a.py: its blob at HEAD is not the one "
                                                   f"there): run with the commit's own files; a change after the "
                                                   f"commit needs a new v1.x first")):
        G.check_code(G.provenance(c1, d, files), 2000, False)
    p = G.provenance(c2, d, files)  # the change's own commit cannot start another chain, not even at step 1000
    assert p["first_commit"] == c1 and p["differ"] == ["a.py"] and p["files"]["b/c.py"]["blob_at_first"]
    with pytest.raises(SystemExit, match=re.escape(f"a.py differ from {c2[:8]} (a.py: its blob there is not the one at "
                                                   f"a.py's first commit {c1[:8]}, the commit v1.2 fixes)")):
        G.check_code(p, 1000, False)
    (d / "a.py").write_text("1\n")
    _run_git(d, "commit", "-q", "-am", "back")  # the first commit's bytes again
    c3 = _run_git(d, "rev-parse", "HEAD")
    (d / "README").write_text("y\n")
    _run_git(d, "commit", "-q", "-am", "unrelated")  # HEAD moves, the files do not
    p = G.provenance(c3, d, files)
    assert p["head"] != c3 and p["differ"] == [] and p["first_commit"] == c1
    G.check_code(p, 3441, False)
    _run_git(d, "update-index", "--skip-worktree", "a.py")  # git status no longer looks at a.py on disk
    (d / "a.py").write_text("1 changed, hidden\n")
    p = G.provenance(c3, d, files)
    assert p["files"]["a.py"]["status"] == "clean" and p["differ"] == ["a.py"]  # the bytes on disk are compared
    with pytest.raises(SystemExit, match=re.escape(f"a.py differ from {c3[:8]} (a.py: the bytes on disk are not its "
                                                   f"blob there")):
        G.check_code(p, 3441, False)
    _run_git(d, "update-index", "--no-skip-worktree", "a.py")
    _run_git(d, "checkout", "--", "a.py")
    root_commit = _run_git(d, "rev-list", "--max-parents=0", "HEAD")
    with pytest.raises(SystemExit, match=re.escape("b/c.py: not in that commit")):  # a commit before the files
        G.check_code(G.provenance(root_commit, d, files), 3441, False)
    with pytest.raises(SystemExit, match="never.py: no commit in HEAD's history adds it"):
        G.provenance(c3, d, ("never.py",) + files)
    with pytest.raises(SystemExit, match="no such commit"):
        G.provenance("0" * 40, d, files)
    (d / "b/c.py").write_text("2 changed\n")  # a later commit that changes only another file of the three
    _run_git(d, "commit", "-q", "-am", "c only")
    p = G.provenance(_run_git(d, "rev-parse", "HEAD"), d, files)
    assert p["differ"] == ["b/c.py"] and p["differ_why"]["b/c.py"].startswith("its blob there is not the one at a.py's")
    with pytest.raises(SystemExit, match="is not the top of its git checkout"):
        G.provenance(None, d / "b", files)
    (tmp_path / "norepo").mkdir()
    with pytest.raises(SystemExit, match="git rev-parse --show-toplevel failed"):
        G.provenance(None, tmp_path / "norepo", files)


# ------------------------------------------------------------------------------------------------ command line
def _write(tmp_path, recs, step=1000, summary=None):
    d = tmp_path / f"heldout_{step}"
    d.mkdir()
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "summary.json").write_text(json.dumps(summary or _summary(recs, step)))
    lst = tmp_path / "list.txt"
    lst.write_text("".join(G.relative(v) + "\n" for v in sorted({r["video"] for r in recs})))
    return d, lst


def _code(committed, expect=None, differ=()):
    st = "clean" if committed else "?? " + G.THIS
    return {"head": "1" * 40, "committed": committed, "expect_commit": expect, "differ": list(differ),
            "files": {G.THIS: {"tracked": committed, "status": st, "md5": "x"}}}


UNCOMMITTED = lambda c=None: _code(False)  # noqa: E731
COMMITTED = lambda c=None: _code(True, "2" * 40 if c else None)  # noqa: E731
CHANGED = lambda c=None: _code(True, "2" * 40, [G.THIS])  # noqa: E731


def _args(d, lst, out, *extra):
    return ["--heldout", str(d), "--out", str(out), "--heldout-list", str(lst), *extra]


def test_main_writes_groups_json(tmp_path, data):
    _, header, recs = data
    d, lst = _write(tmp_path, recs)
    out = tmp_path / "out"
    res = G.main(_args(d, lst, out, "--allow-uncommitted"), header=header, prov=UNCOMMITTED, expect_windows=150)
    assert json.loads((out / "groups.json").read_text(encoding="utf-8")) == json.loads(json.dumps(res))
    m = res["meta"]
    assert m["step"] == 1000 and m["partly_seen"] is True and "partly_seen_note" in m and m["code"]["committed"] is False
    assert m["rows_md5"] == G.md5(d / "rows.jsonl") and m["heldout_list_md5"] == G.md5(lst) and res["check"]["ok"]
    assert (m["note"], m["reading"], m["expect_windows"], m["allow_uncommitted"]) == (G.NOTE, G.READING, 150, True)
    assert m["same_code_as"] is None and sorted(m["imports"]) == sorted(f[:-3].replace("/", ".") for f in G.CODE)
    assert m["env"]["threads"] == {k: "1" for k in G.THREAD_VARS} and m["env"]["torch_threads"] == 1
    assert m["env"]["numpy"] and m["env"]["torch"] and m["env"]["python"]
    with pytest.raises(SystemExit, match="exists"):
        G.main(_args(d, lst, out, "--allow-uncommitted"), header=header, prov=UNCOMMITTED, expect_windows=150)


def test_main_needs_the_committed_file_and_the_same_commit_after_step_1000(tmp_path, data):
    _, header, recs = data
    d1, lst = _write(tmp_path, recs, step=1000)
    d2, _ = _write(tmp_path, recs, step=2000)
    run = lambda d, out, prov, *extra: G.main(_args(d, lst, tmp_path / out, *extra), header=header, prov=prov,  # noqa: E731
                                              expect_windows=150)
    p1 = run(d1, "p1000", COMMITTED, "--expect-commit", "abc")  # the committed step-1000 run
    assert p1["meta"]["partly_seen"] is True and p1["meta"]["code"]["committed"] and p1["meta"]["same_code_as"] is None
    prev = tmp_path / "p1000" / "groups.json"
    for prov, extra, msg in ((UNCOMMITTED, ["--allow-uncommitted"], "step 1000 only"),
                             (UNCOMMITTED, [], "not committed and clean"),
                             (COMMITTED, [], "--expect-commit COMMIT is needed"),
                             (CHANGED, ["--expect-commit", "abc", "--same-code-as", str(prev)], "differ from 22222222"),
                             (COMMITTED, ["--expect-commit", "abc"], "--same-code-as PREV/groups.json is needed")):
        with pytest.raises(SystemExit, match=msg):
            run(d2, "o1", prov, *extra)
    assert not (tmp_path / "o1").exists()
    res = run(d2, "o2", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(prev))
    m = res["meta"]
    assert m["step"] == 2000 and m["partly_seen"] is False and "partly_seen_note" not in m
    assert m["same_code_as"] == {"path": str(prev.resolve()), "md5": G.md5(prev), "step": 1000,
                                 "expect_commit": "2" * 40}
    p2 = tmp_path / "o2" / "groups.json"
    with pytest.raises(SystemExit, match="PREV of step 2000 must be the committed step-1000 run"):
        run(d2, "o3", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(p2))
    with pytest.raises(SystemExit, match="step 1000 takes no --same-code-as: its committed run starts the chain"):
        run(d1, "o3", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(prev))
    with pytest.raises(SystemExit, match="--allow-uncommitted takes no --same-code-as"):
        run(d1, "o3", UNCOMMITTED, "--allow-uncommitted", "--same-code-as", str(prev))
    run(d1, "u1000", UNCOMMITTED, "--allow-uncommitted")

    def edited(src, edit, name):
        g = json.loads(src.read_text(encoding="utf-8"))
        edit(g)
        bad = tmp_path / name
        bad.write_text(json.dumps(g), encoding="utf-8")
        return bad

    for bad, msg in ((tmp_path / "u1000" / "groups.json", "not a committed run"),
                     (edited(prev, lambda g: g["meta"].update(allow_uncommitted=True), "b1.json"), "not a committed run"),
                     (edited(prev, lambda g: g["meta"]["code"].update(expect_commit="3" * 40), "b2.json"),
                      "used commit 33333333"),
                     (edited(prev, lambda g: g["meta"]["code"]["files"][G.THIS].update(md5="y"), "b3.json"),
                      "md5s are not this run's"),
                     (edited(prev, lambda g: g["meta"]["code"].update(first_commit="4" * 40), "b5.json"),
                      "its first commit 44444444 is not this run's"),
                     (edited(prev, lambda g: g["meta"].update(heldout_list_md5="0" * 32), "b6.json"),
                      "another held-out list"),
                     (edited(prev, lambda g: g["meta"]["media"].update(video_maxlen=128), "b7.json"),
                      "other media settings"),
                     (edited(prev, lambda g: g["meta"].update(model="/runs/formal/c_nonfp/ckpt_bf16/checkpoint-1000"),
                             "b8.json"), "a copy of another run"),
                     (edited(prev, lambda g: g.pop("meta"), "b9.json"), "not a groups.json of this tool"),
                     (tmp_path / "missing.json", "not a groups.json of this tool")):
        with pytest.raises(SystemExit, match=msg):
            run(d2, "o4", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(bad))
    # step 3441 follows step 2000's run, which follows step 1000's with the same commit
    d3, _ = _write(tmp_path, recs, step=3441)
    for bad, msg in ((prev, "PREV of step 3441 must be the committed step-2000 run"),
                     (edited(p2, lambda g: g["meta"].update(same_code_as=None), "b4.json"),
                      "does not follow a run with the same commit")):
        with pytest.raises(SystemExit, match=msg):
            run(d3, "o5", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(bad))
    r3 = run(d3, "o6", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(p2))
    assert r3["meta"]["step"] == 3441 and r3["meta"]["same_code_as"]["step"] == 2000
    d4, _ = _write(tmp_path, recs, step=3000)  # the groups are not registered at other steps
    with pytest.raises(SystemExit, match="step 3000: v1.2 registers"):
        run(d4, "o7", COMMITTED, "--expect-commit", "abc", "--same-code-as", str(p2))
    assert not any((tmp_path / o).exists() for o in ("o3", "o4", "o5", "o7"))


def test_main_stops_before_any_number(tmp_path, data):
    vids, header, recs = data
    calls = []
    spy = lambda v: calls.append(v) or header(v)  # noqa: E731
    s = _summary(recs)
    s["by_source"]["scannet"]["r"]["head"] += 1e-6
    d, lst = _write(tmp_path, recs, summary=s)
    args = _args(d, lst, tmp_path / "o", "--expect-commit", "abc")
    with pytest.raises(SystemExit, match="does not give summary.json"):
        G.main(args, header=spy, prov=COMMITTED, expect_windows=150)
    lst.write_text("".join(G.relative(v) + "\n" for v in sorted(vids)[1:]))
    with pytest.raises(SystemExit, match="not the held-out list: 1 extra"):
        G.main(args, header=spy, prov=COMMITTED, expect_windows=150)
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs[1:]))
    lst.write_text("".join(G.relative(v) + "\n" for v in sorted(vids)))
    with pytest.raises(SystemExit, match="1 videos without 4 rows"):
        G.main(args, header=spy, prov=COMMITTED, expect_windows=150)
    recs[1]["u"] = recs[0]["u"]  # 4 rows, but two windows drawn at the same u
    (d / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match=r"1 videos without the u values \(0\.125, 0\.375, 0\.625, 0\.875\)"):
        G.main(args, header=spy, prov=COMMITTED, expect_windows=150)
    assert calls == [] and not (tmp_path / "o").exists()  # no window, no group number, no output


@pytest.mark.parametrize("edit,msg", [(lambda r: r.pop("u"), r"missing \['u'\].*ask the user"),
                                      (lambda r: r["head"].pop("cos_sum"), r"missing \['head\.cos_sum'\]"),
                                      (lambda r: r.update(source="scannet"), "the path says")])
def test_rows_without_a_field_or_with_another_source_stop_the_run(tmp_path, data, edit, msg):
    _, _, recs = data
    edit(recs[7])
    p = tmp_path / "rows.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(SystemExit, match=msg):
        G.read_rows(p)


def test_the_modules_run_are_this_checkouts(tmp_path, monkeypatch):
    got = G.check_imports()
    assert sorted(got.values()) == sorted(str((G.ROOT / f).resolve()) for f in G.CODE)
    with pytest.raises(SystemExit, match="is imported from"):
        G.check_imports(tmp_path)
    name = "vagen.q38s4.tools.nfp_heldout"  # a module loaded from elsewhere
    shadow = types.ModuleType(name)
    shadow.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=str(tmp_path / "nfp_heldout.py"))
    monkeypatch.setitem(sys.modules, name, shadow)
    with pytest.raises(SystemExit, match=re.escape(f"{name} is imported from {tmp_path.resolve()}")):
        G.check_imports()


def test_a_copy_under_another_name_does_not_run(tmp_path):
    """A copy next to the registered file (here: the same layout under tmp_path) would pass every check made on the
    registered file; it stops first."""
    assert G.check_this_file() == str((G.ROOT / G.THIS).resolve())
    copy = tmp_path / "vagen/q38s4/tools/nfp_groups_v13.py"
    copy.parent.mkdir(parents=True)
    shutil.copy(Path(G.__file__), copy)
    g = runpy.run_path(str(copy), run_name="copy")
    with pytest.raises(SystemExit, match=re.escape(f"this is {copy.resolve()}, not ") + ".*only the registered file"):
        g["main"](["--heldout", str(tmp_path), "--out", str(tmp_path / "o"), "--allow-uncommitted"])
    assert not (tmp_path / "o").exists()


def test_the_import_check_as_python_m_runs_it():
    """Under -m the tool is __main__, not in sys.modules under its name (run_module does the same); thread counts
    the shell exports are replaced by 1 before numpy loads."""
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
           "NUMEXPR_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "true", "PYTHONDONTWRITEBYTECODE": "1"}
    code = ("import json, os, runpy, sys; g = runpy.run_module('vagen.q38s4.tools.nfp_groups', run_name='x'); "
            "assert 'vagen.q38s4.tools.nfp_groups' not in sys.modules; "
            "print(json.dumps([g['check_imports'](), {k: os.environ[k] for k in g['THREAD_VARS']}, "
            "os.environ['TOKENIZERS_PARALLELISM']]))")
    p = subprocess.run([sys.executable, "-c", code], cwd=G.ROOT, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    imports, threads, tok = json.loads(p.stdout.strip().splitlines()[-1])
    assert sorted(imports.values()) == sorted(str((G.ROOT / f).resolve()) for f in G.CODE)
    assert threads == {k: "1" for k in G.THREAD_VARS} and tok == "false"


def test_small_helpers():
    assert G.step_of({"meta": {"model": "/r/ckpt_bf16/checkpoint-3441"}}) == 3441 and G.step_of({"meta": {}}) is None
    assert G.step_of({"meta": {"model": "/runs/c_nfp_from_checkpoint-1000/ckpt_bf16/checkpoint-3441"}}) == 3441
    assert G.step_of({"meta": {"model": "/r/checkpoint-1000/bf16"}}) is None
    assert G.relative("/mnt/x/VSI-590K/adt/a.mp4") == "adt/a.mp4" and G.stem("a/b.mp4") == "a/b"
    assert G.THIS == str(Path(G.__file__).resolve().relative_to(G.ROOT).as_posix())
    assert all((G.ROOT / f).is_file() for f in G.CODE)


# ------------------------------------------------------------------------------------------------ a real video
REAL = Path("/mnt/umm/users/yinbaiqiao/q38/s4/data/VSI-590K")


@pytest.mark.skipif(not (G.HELDOUT_LIST.is_file() and REAL.is_dir()), reason="held-out videos not readable")
def test_video_header_reads_a_held_out_video_as_the_training_plugin_does():
    P = pytest.importorskip("vagen.q38s4.lf_plugin")
    from vagen.q38s4.lf_video import stream_info

    rel = next(l.strip() for l in G.HELDOUT_LIST.read_text().splitlines() if l.strip())
    path = str(REAL / rel)
    c, s = P._open_video_stream(path)
    try:
        info = stream_info(c, s)
    finally:
        c.close()
    assert G.video_header(path) == (info["total_frames"], info["average_fps"])

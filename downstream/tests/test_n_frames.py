"""The frame rules of 2026-11-05 (the user's plan 甲 after the 11-04 rescan, downstream/n/n_frames.py): near-black,
upside-down and striped tiles, broken or only hints by kind of strip; scan's new fields and what n_report.py --exclude
takes from them; the closed-loop example-strip scan (thumbs) and its place in run_eval_ckpt.sh."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402
import n_frames as F  # noqa: E402
import n_report as R  # noqa: E402

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

S = 16


def grey(a):
    return np.repeat(np.asarray(a, dtype=np.uint8)[..., None], 3, axis=2)


ACROSS = grey(np.tile(np.arange(S) * 12 + 30, (S, 1)))  # a left-to-right ramp: trips nothing
DOWN = grey(np.tile((np.arange(S) * 12 + 30)[:, None], (1, S)))  # a top-to-bottom ramp
UP = DOWN[::-1]  # DOWN upside down
DARK = grey(np.full((S, S), 4))
STRIPES = grey((np.arange(S)[:, None] % 2) * 180 + np.arange(S)[None, :] + 20)  # rows alternate, a slight ramp across


def test_each_rule_trips_on_its_own_kind_of_tile():
    assert F.tile_rules([ACROSS, DOWN, UP, DARK, STRIPES]) == [[], [], ["flip"], ["near_black"], ["stripes"]]
    assert F.tile_rules([UP, ACROSS, ACROSS, DARK, DARK]) == [[], [], [], ["near_black"], ["near_black"]]
    assert all(F.chroma(t) == 0 for t in (ACROSS, DOWN, DARK, STRIPES))  # grey: never colour noise


def test_broken_and_hints_depend_on_the_kind_of_strip():
    ts = [ACROSS, DOWN, UP, DARK, STRIPES]
    assert F.defects(ts, "png") == {"broken": {"near_black": [3]}, "hints": {"flip": [2], "stripes": [4]}}
    assert F.defects(ts, "jpg") == {"broken": {"near_black": [3], "flip": [2]}, "hints": {}}
    assert F.defects([ACROSS, ACROSS], "png") == {"broken": {}, "hints": {}}


def strip(path, ts):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(ts, axis=1)).save(path)


def test_scan_adds_broken_and_hints_and_only_broken_instances_are_left_out(tmp_path, capsys):
    ev = tmp_path / "ev"
    strip(ev / "S/frames/A_s1_equidistance.png", [ACROSS, DARK, ACROSS])
    strip(ev / "RS/frames/A_s1_equidistance.png", [ACROSS, DOWN, UP])
    strip(ev / "S/frames/B_s1_fov_inclusion.png", [ACROSS, STRIPES])
    strip(ev / "RS/frames/B_s1_fov_inclusion.png", [ACROSS, ACROSS])
    rep = F.scan([ev])
    r = rep["evals"][str(ev)]
    assert rep["rules"] == {"broken": ["near_black"], "hints": ["flip", "stripes"]}
    assert r["tiles"] == 10 and r["corrupted_tiles"] == 0 and r["instances"] == {} and r["episodes"] == 0
    assert r["broken"] == {"A_s1_equidistance": {"S": {"near_black": [1], "tiles": 3}}}
    assert r["hints"] == {"A_s1_equidistance": {"RS": {"flip": [2], "tiles": 3}},
                          "B_s1_fov_inclusion": {"S": {"stripes": [1], "tiles": 2}}}
    out = tmp_path / "scan.json"
    assert F.main(["scan", "--eval", str(ev), "--out", str(out)]) == 0
    assert "broken {'near_black': 1} in 1 instances, hints {'flip': 1, 'stripes': 1} in 2 instances" in \
        capsys.readouterr().out
    assert R.excluded([out]) == {"A_s1_equidistance"}  # B has hints only: looked at before it is left out
    old = tmp_path / "old.json"  # a scan report from before 11-05: corrupted instances only
    old.write_text(json.dumps({"evals": {"x": {"instances": {"C_s1_equidistance": {"S": {}}}}}}))
    assert R.excluded([old, out]) == {"A_s1_equidistance", "C_s1_equidistance"}


def thumb(folder, name, frames, row=None):
    """An example strip written by closed_loop.FrameRecorder itself (32 px frames, JPEG), and optionally its row."""
    rec = closed_loop.FrameRecorder(folder / "frames" / f"{name}.jpg", size=32, keep=len(frames))
    for k, f in enumerate(frames):
        rec.add(16 * k, {closed_loop.FRAME_CAMERA: f})
    saved = rec.save(bool(row and row.get("success")))
    if row is not None:
        with open(folder / "episodes.jsonl", "a") as fh:
            fh.write(json.dumps(dict(row, **saved)) + "\n")


def closed_loop_folder(tmp_path):
    ev = tmp_path / "out/episodes"
    thumb(ev, "OpenDrawer_s1000", [ACROSS, STRIPES, ACROSS, ACROSS],
          {"task": "OpenDrawer", "seed": 1000, "status": "complete", "success": True})
    thumb(ev, "OpenDrawer_s1001", [ACROSS, DOWN, UP, ACROSS],
          {"task": "OpenDrawer", "seed": 1001, "status": "complete", "success": False})
    with open(ev / "episodes.jsonl", "a") as fh:  # a later error row of the same episode: the complete one counts
        fh.write(json.dumps({"task": "OpenDrawer", "seed": 1001, "status": "error"}) + "\n")
    thumb(ev, "PickPlaceCounterToSink_s1002", [ACROSS, DARK, DARK, ACROSS])  # no row: its worker died
    thumb(ev, "LoadDishwasher_s1002", [DARK, ACROSS], {"task": "LoadDishwasher", "seed": 1002, "status": "error"})
    (ev / "frames/Cut_s1003.jpg").write_bytes(b"\xff\xd8 cut short")
    Image.new("RGB", (70, 50)).save(ev / "frames/Odd_s1004.jpg")
    excl = tmp_path / "excluded.json"
    excl.write_text(json.dumps({"episodes": [["LoadDishwasher", 1002]]}))
    return ev, excl


def test_thumbs_records_broken_episodes_with_their_steps_and_results(tmp_path):
    ev, excl = closed_loop_folder(tmp_path)
    rep = F.thumbs([ev], excluded_path=excl)
    assert rep["rules"] == ["near_black", "flip"]
    r = rep["folders"][str(ev)]
    assert r["strips"] == 6 and r["tiles"] == 14 and r["unreadable"] == ["Cut_s1003.jpg", "Odd_s1004.jpg"]
    by = {f["episode"]: f for f in r["flagged"]}
    assert set(by) == {"OpenDrawer_s1001", "PickPlaceCounterToSink_s1002", "LoadDishwasher_s1002"}  # stripes: unused
    assert by["OpenDrawer_s1001"] == {"episode": "OpenDrawer_s1001", "task": "OpenDrawer", "seed": 1001, "tiles": 4,
                                      "broken": {"flip": [2]}, "steps": [32], "status": "complete",
                                      "success": False, "excluded": False}
    assert by["PickPlaceCounterToSink_s1002"]["broken"] == {"near_black": [1, 2]}
    assert by["PickPlaceCounterToSink_s1002"]["status"] is None and by["PickPlaceCounterToSink_s1002"]["steps"] == []
    assert by["LoadDishwasher_s1002"]["excluded"] and by["LoadDishwasher_s1002"]["status"] == "error"
    assert F.thumbs([tmp_path / "empty"])["folders"][str(tmp_path / "empty")] == {
        "strips": 0, "tiles": 0, "unreadable": [], "flagged": []}


def test_thumbs_command_writes_the_report_and_lists_the_episodes(tmp_path, capsys):
    ev, _ = closed_loop_folder(tmp_path)
    out = tmp_path / "out/frames_scan.json"
    assert F.main(["thumbs", "--eval", str(ev), "--out", str(out)]) == 0
    assert len(json.loads(out.read_text())["folders"][str(ev)]["flagged"]) == 3
    text = capsys.readouterr().out
    assert f"{ev}: 3 of 6 episodes with a broken frame (near_black, flip), 2 strips unreadable" in text
    assert "  OpenDrawer_s1001: flip 1 of 4 tiles at steps [32]; complete, success False" in text
    assert "  PickPlaceCounterToSink_s1002: near_black 2 of 4 tiles at steps []; no result" in text


def test_every_evaluation_scans_its_example_strips_without_changing_its_exit_code():
    script = (ROOT / "eval/run_eval_ckpt.sh").read_text()
    start = script.index('closed_loop.py" --ckpt')
    loop = script[start:script.index("\ndone", start)]
    scan = loop[loop.index('n_frames.py" thumbs'):]
    assert '--eval "$OUT/episodes$SUFFIX"' in scan and '--out "$OUT/frames_scan$SUFFIX.json"' in scan
    assert "rc=" not in scan and "exit" not in scan

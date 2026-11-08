"""WP-WBDEMO tests: RM expert demos (real 640x400 renders), the 20-dialogue warm-up set, the wb_demos.sh payload.

Login node, 1 core, no GPU, PY_EVAL (WalkerBench env), about 1-2 min (three real-render subprocess runs):
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
      nice -n 19 $PY_EVAL -m pytest -q -p no:cacheprovider vpd/tests/test_wbdemo.py
Scratch goes under eval_external_assets/vpd/scratch/wbdemo/pytest_<rand>/ (env VPD_WBDEMO_TEST_DIR overrides), never
/tmp, and is removed at the end.  The frozen split and the survey prototypes under OUT are only read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vpd import paths  # noqa: E402
from vpd.walkerbench import action_regex as AR  # noqa: E402
from vpd.walkerbench import demos as D  # noqa: E402
from vpd.walkerbench import split as S  # noqa: E402
from vpd.walkerbench import warmup as W  # noqa: E402

WORKTREE = Path(__file__).resolve().parents[2]
SPLIT_JSON = paths.WB_SPLIT / "split_s2.json"
PROTO_DEMOS = paths.PLAN_DIR / "login_tmp" / "vpd_demos" / "demo_proto"
PROTO_STEMS = ("nav_0003_target_20260124_0233_3", "vis_0001_target_20260127_1940_06")
need_split = pytest.mark.skipif(not SPLIT_JSON.exists(), reason="frozen split_s2.json missing")


@pytest.fixture(scope="module")
def workdir():
    base = Path(os.environ.get("VPD_WBDEMO_TEST_DIR", paths.scratch("wbdemo")))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d)  # only what this test created


@pytest.fixture(scope="module")
def split_doc():
    return json.loads(SPLIT_JSON.read_text())


def _env() -> dict:
    env = dict(os.environ, PYTHONPATH=str(WORKTREE), PYTHONDONTWRITEBYTECODE="1", VPD_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    return env


def _py(*args, timeout=900) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=WORKTREE, env=_env(), capture_output=True, text=True,
                          timeout=timeout)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jpeg(path: Path, size=(640, 400)) -> str:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 120, 150)).save(path, quality=90)
    return str(path)


def _fake_record(stem: str, img_dir: Path, n: int = 3, split: str = "train_rest", size=(640, 400)) -> dict:
    msgs = [{"role": "system", "content": "SYS"}]
    imgs = []
    for j in range(n):
        msgs.append({"role": "user", "content": f"**Step {j + 1}**\n...\nAnalyze the image." + D.IMAGE})
        act = AR.STOP_ACTION if j == n - 1 else ("ROTATION(244.4, 0)" if j == 0 else "MOVE(1)")
        msgs.append({"role": "assistant", "content": AR.reply(act)})
        imgs.append(_jpeg(img_dir / stem / ("turn_%02d.jpg" % j), size))
    return {"id": stem, "task_type": S.kind_of(stem), "split": split, "messages": msgs, "images": imgs,
            "n_turns": n, "size": list(size), "expert": {"rotations": 1}}


# ---------------------------------------------------------------- pure helpers -------------------------------------

def test_action_text_and_reply():
    cases = [({"type": "move", "move_id": 3}, "MOVE(3)"), ({"type": "rotation", "heading": 244.4, "pitch": 0},
             "ROTATION(244.4, 0)"), ({"type": "stop", "answer": "ARRIVED"}, 'STOP("ARRIVED")')]
    for act, want in cases:
        assert D.action_text(act) == want
        txt = D.reply_text(act)
        assert txt == json.dumps({"THOUGHT": "", "ACTION": want})
        assert D.regex_ok(txt, 4)
        assert D.same_action(D._fallback_parse(txt), dict(act, answer=act.get("answer"), pitch=act.get("pitch")))
    assert D.STOP_REPLY == '{"THOUGHT": "", "ACTION": "STOP(\\"ARRIVED\\")"}'
    assert not D.regex_ok(D.reply_text({"type": "move", "move_id": 5}), 4)
    with pytest.raises(ValueError):
        D.action_text({"type": "jump"})


def test_to_messages():
    raw = [{"role": "system", "content": "S"},
           {"role": "user", "content": [{"type": "text", "text": "T1"}, {"type": "image_url", "image_url": {}}]},
           {"role": "assistant", "content": "R"},
           {"role": "user", "content": [{"type": "text", "text": "T2"}]}]
    assert D.to_messages(raw) == [{"role": "system", "content": "S"}, {"role": "user", "content": "T1<image>"},
                                  {"role": "assistant", "content": "R"}, {"role": "user", "content": "T2"}]
    with pytest.raises(ValueError):
        D.to_messages([{"role": "user", "content": [{"type": "image_url"}, {"type": "text", "text": "x"}]}])


def test_shards_and_tokens():
    stems = [f"nav_{i:04d}" for i in range(23)]
    parts = [D.shard_stems(stems, i, 8) for i in range(8)]
    assert sorted(s for p in parts for s in p) == sorted(stems)
    assert sum(len(p) for p in parts) == len(stems)
    assert parts[0] == sorted(stems)[0::8]
    assert D.parse_shard("3/8") == (3, 8)
    for bad in ("8/8", "1", "-1/2", "a/b"):
        with pytest.raises(ValueError):
            D.parse_shard(bad)
    assert D.image_tokens((640, 400)) == 242 and D.image_tokens((1280, 800)) == 1002


def test_check_record(workdir):
    good = _fake_record("nav_0001_x", workdir / "cr")
    assert D.check_record(good, (640, 400)) == []
    bad = json.loads(json.dumps(good))
    bad["messages"][1]["content"] = D.IMAGE + "text"
    assert any("at the end" in p for p in D.check_record(bad, (640, 400)))
    bad = json.loads(json.dumps(good))
    bad["images"] = bad["images"][:-1]
    assert any("images for" in p for p in D.check_record(bad, (640, 400)))
    bad = json.loads(json.dumps(good))
    bad["messages"][-1]["content"] = AR.reply("MOVE(1)")
    assert any("last reply" in p for p in D.check_record(bad, (640, 400)))
    bad = json.loads(json.dumps(good))
    bad["messages"][2]["content"] = '{"ACTION": "MOVE(1)", "THOUGHT": ""}'
    assert any("not canonical" in p for p in D.check_record(bad, (640, 400)))
    bad = json.loads(json.dumps(good))
    bad["images"][0] = _jpeg(workdir / "cr" / "small.jpg", (320, 200))
    assert any("not (640, 400)" in p for p in D.check_record(bad, (640, 400)))
    bad = json.loads(json.dumps(good))
    bad["n_turns"] = 2
    assert any("n_turns" in p for p in D.check_record(bad, (640, 400)))
    # a parser that maps every reply to the stock fallback rotation fails the round trip
    fallback = lambda t, o: {"type": "rotation", "heading": o["heading"], "pitch": o["pitch"]}  # noqa: E731
    assert any("round-trip" in p for p in D.check_record(good, (640, 400), parse=fallback))


@need_split
def test_split_tags(split_doc):
    tags = D.split_tags(split_doc)
    assert len(tags) == 540
    assert sum(v == "val" for v in tags.values()) == 57
    assert sum(v == "train_rest" for v in tags.values()) == 483


@need_split
def test_merge_fake(workdir, split_doc):
    sp = split_doc["splits"]
    stems = sorted(sp["s2val"][:2] + sp["train_rest"][:3])
    tags = D.split_tags(split_doc)
    root = workdir / "merge"
    tasks = root / "tasks"
    tasks.mkdir(parents=True)
    for s in stems:
        (tasks / f"{s}.json").write_text("{}")
    out = root / "out"
    sha = S.sha256_file(SPLIT_JSON)

    def write_shards(n=2, partial=False, drop=None):
        sd = out / "shards"
        if sd.exists():
            shutil.rmtree(sd)
        sd.mkdir(parents=True)
        for i in range(n):
            mine = D.shard_stems(stems, i, n)
            recs = [_fake_record(s, out / "images", split=tags[s]) for s in mine if s != drop]
            text = "".join(json.dumps(r) + "\n" for r in recs)
            (sd / f"shard{i}of{n}.jsonl").write_text(text)
            man = {"shard": [i, n], "size": [640, 400], "assigned": mine, "partial": partial, "n_records": len(recs),
                   "dropped": [{"id": drop, "reasons": ["test"]}] if drop in mine else [],
                   "jsonl": f"shard{i}of{n}.jsonl", "jsonl_sha256": D.sha256_text(text), "split_s2_sha256": sha,
                   "sandbox_settings_diff": "DIFF", "walkerbench_commit": "c", "seconds": 1.0,
                   "sandbox_seconds": 0.5, "unconsumed_temp_images": 0, "logs": "L"}
            (sd / f"shard{i}of{n}.json").write_text(json.dumps(man))

    write_shards()
    man = D.merge(out, (640, 400), tasks, paths.WB_SPLIT)
    lines = (out / "demos_all.jsonl").read_text().splitlines()
    assert [json.loads(x)["id"] for x in lines] == stems
    assert man["counts"]["records"] == 5 and man["counts"]["by_split"] == {"train_rest": 3, "val": 2}
    assert man["n_images"] == 15 and man["tokens_estimate"]["image_tokens"] == 15 * 242
    assert man["demos_all_sha256"] == _sha(out / "demos_all.jsonl")
    # same content again is fine; different content is refused without --force
    D.merge(out, (640, 400), tasks, paths.WB_SPLIT)
    write_shards(drop=stems[1])
    with pytest.raises(FileExistsError):
        D.merge(out, (640, 400), tasks, paths.WB_SPLIT)
    man = D.merge(out, (640, 400), tasks, paths.WB_SPLIT, force=True)
    assert man["counts"]["dropped"] == 1 and man["counts"]["records"] == 4
    # a missing shard, a partial shard, a tampered shard file
    write_shards()
    (out / "shards" / "shard1of2.json").unlink()
    with pytest.raises(RuntimeError, match="shards present"):
        D.merge(out, (640, 400), tasks, paths.WB_SPLIT, force=True)
    write_shards(partial=True)
    with pytest.raises(RuntimeError, match="partial"):
        D.merge(out, (640, 400), tasks, paths.WB_SPLIT, force=True)
    D.merge(out, (640, 400), tasks, paths.WB_SPLIT, allow_partial=True, force=True)
    write_shards()
    with open(out / "shards" / "shard0of2.jsonl", "a") as f:
        f.write("\n")
    with pytest.raises(RuntimeError, match="changed"):
        D.merge(out, (640, 400), tasks, paths.WB_SPLIT, force=True)
    write_shards()
    with pytest.raises(RuntimeError, match="other size"):
        D.merge(out, (1280, 800), tasks, paths.WB_SPLIT, force=True)


def test_frozen_dir_guards(capsys):
    with pytest.raises(SystemExit):
        D.main(["--limit", "1"])
    with pytest.raises(SystemExit):
        D.main(["--merge", "--allow-partial"])
    with pytest.raises(SystemExit):
        W.main(["--limit", "3"])


# ---------------------------------------------------------------- warm-up (pure) -----------------------------------

def test_warmup_templates_frozen():
    assert W.T_MOVE == "Format practice, not a navigation task: take available move %d (%s)."
    assert W.T_ROT == "Format practice, not a navigation task: without moving, look %s (heading=%s, pitch=0)."
    assert W.T_STOP == "Format practice, not a navigation task: you are already at the destination. Stop now."
    prereg = (paths.VPD_PKG / "PREREG_WB_PILOT.md").read_text(encoding="utf-8")
    assert W.T_MOVE.replace("%d", "{id}").replace("%s", "{direction}") in prereg
    assert W.T_ROT.replace("%s", "{LEFT|RIGHT|BACK}", 1).replace("%s", "{h}") in prereg
    assert W.T_STOP in prereg
    assert W.PROTO_SHA256 in prereg
    assert W.PLAN == ["move"] * 10 + ["rotation"] * 5 + ["stop"] * 5


def test_warmup_instruction_rules():
    obs = {"heading": 300.0, "available_moves": [{"id": 1, "direction": "front"}, {"id": 2, "direction": "left"},
                                                 {"id": 3, "direction": "back"}]}
    assert W.instruction(0, obs) == (W.T_MOVE % (1, "front"), "MOVE(1)")
    assert W.instruction(4, obs) == (W.T_MOVE % (2, "left"), "MOVE(2)")  # 4 % 3 = 1
    sides = [W.instruction(i, obs) for i in range(10, 15)]
    assert [s[1] for s in sides] == ["ROTATION(210.0, 0)", "ROTATION(30.0, 0)", "ROTATION(120.0, 0)",
                                     "ROTATION(30.0, 0)", "ROTATION(210.0, 0)"]
    assert "look LEFT (heading=210.0, pitch=0)" in sides[0][0]
    assert W.instruction(17, obs) == (W.T_STOP, AR.STOP_ACTION)
    h = W.md5("pano")
    assert W.spawn_heading(h) == round(int(h[:6], 16) % 3600 / 10, 1)
    assert 0 <= W.spawn_heading("ffffff00") < 360


def test_warmup_select_rules():
    fences = {"f1": {"a", "b", "c", "d"}, "f2": {"e", "f", "g"}, "f3": {"h", "i"}, "fX": {"z", "y", "x"}}
    links = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"], "d": ["a"], "e": ["f", "g"], "f": ["e"], "g": ["e"],
             "h": ["i"], "i": ["h"], "z": ["y", "x"], "y": ["z"], "x": ["z"]}
    imgs = set("abcdefghixyz") - {"g"}
    used = {"a"}
    picks, counts = W.select(["f1", "f2", "f3"], fences, links, imgs, used)
    # f1: b and c qualify (a used, d has 1 neighbour); f2: e has f (with image) and g (no image) -> 1 -> none;
    # f3: each has 1 neighbour -> none; fX is not a train-rest fence
    assert counts == {"f1": 2, "f2": 0, "f3": 0}
    assert picks == [(W.md5(min(("b", "c"), key=W.md5)), "f1", min(("b", "c"), key=W.md5))]
    tj = W.task_json("nav_warm_00", "f1", "b", 12.3)
    assert tj["spawn_point"] == "b" and tj["visual_path"] == [{"step_index": 0, "pano_id": "b"}]


@need_split
def test_warmup_fences_disjoint(split_doc):
    rest, other = W.fence_sets(split_doc)
    assert len(rest) == 47 and not set(rest) & other


# ---------------------------------------------------------------- real renders (subprocess) ------------------------

@pytest.fixture(scope="module")
def demos_limit3(workdir):
    out = workdir / "demos"
    r = _py("-m", "vpd.walkerbench.demos", "--limit", "3", "--out", str(out), "--work", str(workdir / "work"),
            "--strict")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out


@need_split
def test_demos_limit3(demos_limit3, split_doc):
    out = demos_limit3
    man = json.loads((out / "shards" / "shard0of1.json").read_text())
    assert man["partial"] and man["limit"] == 3 and man["dropped"] == []
    assert man["assigned"] == sorted(split_doc["splits"]["s2train"])[:3]
    assert "(640, 400)" in man["sandbox_settings_diff"] and "vpd_s2train" in man["sandbox_settings_diff"]
    recs = [json.loads(x) for x in (out / "shards" / "shard0of1.jsonl").read_text().splitlines()]
    assert [r["id"] for r in recs] == man["assigned"]
    sys_prompt = (paths.REPO_WB / "config" / "system_prompt_nav.txt").read_text(encoding="utf-8").strip()
    tags = D.split_tags(split_doc)
    for r in recs:
        assert sorted(r) == ["expert", "id", "images", "messages", "n_turns", "size", "source", "split", "task_type"]
        assert D.check_record(r, (640, 400)) == []
        assert r["task_type"] == S.kind_of(r["id"]) and r["split"] == tags[r["id"]]
        assert r["messages"][0] == {"role": "system", "content": sys_prompt}
        users = [m["content"] for m in r["messages"] if m["role"] == "user"]
        replies = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
        assert len(users) == len(replies) == len(r["images"]) == r["n_turns"] >= 2
        assert all(u.endswith("<image>") and u.count("<image>") == 1 for u in users)
        assert all(u.startswith("**Step %d**" % (j + 1)) for j, u in enumerate(users))
        assert replies[-1] == D.STOP_REPLY
        assert all(D.image_size(p) == (640, 400) for p in r["images"])
        assert [Path(p).name for p in r["images"]] == ["turn_%02d.jpg" % j for j in range(r["n_turns"])]
        assert all(Path(p).parent == out / "images" / r["id"] for p in r["images"])
        assert r["expert"]["success"] == 1 and r["expert"]["steps"] == r["n_turns"]
        assert r["source"]["sha256"] == split_doc["tasks"][r["id"]]["sha256"]
        # the expert walks the visual path: at most path length - 1 moves (fewer only if the path revisits a pano)
        assert 1 <= r["expert"]["moves"] <= r["source"]["n_path"] - 1
        assert r["expert"]["moves"] + r["expert"]["rotations"] + 1 == r["n_turns"]
    # partial merge into the scratch dir
    r = _py("-m", "vpd.walkerbench.demos", "--merge", "--allow-partial", "--out", str(out))
    assert r.returncode == 0, r.stderr[-3000:]
    m = json.loads((out / "manifest.json").read_text())
    assert m["partial"] and m["counts"]["records"] == 3 and m["demos_all_sha256"] == _sha(out / "demos_all.jsonl")
    assert m["code"]["worktree"].get("vpd_tree_sha256") or m["code"]["worktree"].get("error")
    # a full (non-partial) merge of a 3-task shard is refused: it does not cover s2train
    r = _py("-m", "vpd.walkerbench.demos", "--merge", "--out", str(out), "--force")
    assert r.returncode != 0 and "partial" in r.stderr
    # re-rendering into the same out dir needs --force
    r = _py("-m", "vpd.walkerbench.demos", "--limit", "1", "--out", str(out), "--work", str(out.parent / "work"))
    assert r.returncode != 0 and "exists" in r.stderr


@need_split
@pytest.mark.skipif(not (PROTO_DEMOS / "demos_proto.jsonl").exists(), reason="survey prototype demos missing")
def test_demos_match_prototype(workdir):
    out = workdir / "demos_proto"
    r = _py("-m", "vpd.walkerbench.demos", "--stems", ",".join(PROTO_STEMS), "--out", str(out),
            "--work", str(workdir / "work"), "--strict")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    mine = {json.loads(x)["id"]: json.loads(x) for x in (out / "shards" / "shard0of1.jsonl").read_text().splitlines()}
    proto = {json.loads(x)["id"]: json.loads(x) for x in (PROTO_DEMOS / "demos_proto.jsonl").read_text().splitlines()}
    assert set(mine) == set(PROTO_STEMS)
    rotations = 0
    for s in PROTO_STEMS:
        assert mine[s]["messages"] == proto[s]["messages"]  # same expert (B6), same text, same replies
        assert len(mine[s]["images"]) == len(proto[s]["images"])
        for a, b in zip(mine[s]["images"], proto[s]["images"]):
            assert _sha(a) == _sha(PROTO_DEMOS / "images" / s / Path(b).name)  # same engine JPEG bytes
        rotations += mine[s]["expert"]["rotations"]
    assert rotations >= 1  # the nav prototype task exercises the rotate-first rule


@pytest.fixture(scope="module")
def warm(workdir):
    out = workdir / "warm"
    r = _py("-m", "vpd.walkerbench.warmup", "--out", str(out), "--work", str(workdir / "work"),
            "--proto-check", "strict" if W.PROTO.exists() else "off")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out


@need_split
def test_warmup_build(warm):
    man = json.loads((warm / "manifest.json").read_text())
    recs = [json.loads(x) for x in (warm / "warmup20.jsonl").read_text().splitlines()]
    assert man["n"] == 20 and man["kinds"] == {"move": 10, "rotation": 5, "stop": 5}
    assert man["sha256"] == _sha(warm / "warmup20.jsonl")
    assert len(set(man["fences"])) == 20 and man["selection"]["disjoint_from_test_val"]
    assert man["selection"]["candidate_panos"] == 1088 and man["selection"]["train_rest_fences"] == 47
    sys_prompt = (paths.REPO_WB / "config" / "system_prompt_nav.txt").read_text(encoding="utf-8").strip()
    for i, r in enumerate(recs):
        assert r["id"] == "nav_warm_%02d" % i and r["task_type"] == "warmup_" + W.PLAN[i]
        assert [m["role"] for m in r["messages"]] == ["system", "user", "assistant"]
        assert r["messages"][0]["content"] == sys_prompt
        u, a = r["messages"][1]["content"], r["messages"][2]["content"]
        assert u.startswith("**Step 1**") and u.endswith("<image>") and u.count("<image>") == 1
        instr = re.search(r"\*\*Task:\*\* (.*)\n", u).group(1)
        act = json.loads(a)["ACTION"]
        assert a == AR.reply(act) and D.regex_ok(a, man["checks"][i]["n_moves"])
        if W.PLAN[i] == "move":
            assert re.fullmatch(re.escape(W.T_MOVE).replace("%d", r"\d+").replace("%s", ".+"), instr)
            assert act == "MOVE(%s)" % re.search(r"move (\d+)", instr).group(1)
        elif W.PLAN[i] == "rotation":
            h = re.search(r"heading=([0-9.]+)", instr).group(1)
            assert act == "ROTATION(%s, 0)" % h
        else:
            assert instr == W.T_STOP and act == AR.STOP_ACTION
        assert len(r["images"]) == 1 and D.image_size(r["images"][0]) == (640, 400)
        assert r["source"]["pano"] == man["panos"][i] and r["source"]["geofence"] == man["fences"][i]
    if W.PROTO.exists():
        p = man["prototype"]
        assert p["present"] and p["sha256"] == W.PROTO_SHA256
        assert p["messages_differ"] == [] and p["images_byte_identical"] == 20
    # the frozen output is not replaced without --force
    r = _py("-m", "vpd.walkerbench.warmup", "--out", str(warm), "--work", str(warm.parent / "work"))
    assert r.returncode != 0 and "exists" in r.stderr


# ---------------------------------------------------------------- job payload --------------------------------------

def test_wb_demos_sh(workdir):
    sh = paths.VPD_PKG / "jobs" / "wb_demos.sh"
    assert subprocess.run(["bash", "-n", str(sh)], capture_output=True).returncode == 0
    run = workdir / "run_dry"
    r = subprocess.run(["bash", str(sh), str(run)], env=dict(_env(), VPD_DRYRUN="1"), capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = r.stdout
    assert "-m vpd.walkerbench.split --check" in out
    assert sum(f"--shard {i}/8 " in out for i in range(8)) == 8
    assert "-m vpd.walkerbench.demos --merge" in out and "-m vpd.walkerbench.warmup" in out
    for s in (1001, 1, 2, 3):
        assert f"tiers materialize --seed {s} --tiers 25,50,100,200,all --demos " \
               f"{paths.WB_DEMOS}/s2train_640x400/demos_all.jsonl" in out
    assert str(paths.PY_EVAL) in out
    steps = [json.loads(x) for x in (run / "steps.jsonl").read_text().splitlines()]
    assert [s["name"] for s in steps] == ["split_check", "demos_shards", "demos_merge", "warmup",
                                         "tiers_1001", "tiers_1", "tiers_2", "tiers_3"]
    assert all(s["rc"] == 0 for s in steps)
    assert (run / "node.txt").exists() and not (run / "DONE").exists()
    r = subprocess.run(["bash", str(sh)], env={k: v for k, v in _env().items() if k != "RUN"}, capture_output=True,
                       text=True)
    assert r.returncode == 2 and "usage" in r.stderr

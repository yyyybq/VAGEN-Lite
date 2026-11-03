"""The matrix v3 starts evaluated N1's way (downstream/n/nm_starts.py, nm_report.py, run_nm.sh; 2026-11-03)."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
sys.path.insert(0, str(ROOT))
import nm_report as NR  # noqa: E402
import nm_starts as S  # noqa: E402

M = "/m/exps/matrix_v3"
RL = M + "/m3_{g}_s2026101{c}/rl_run_m3{g}s{k}{r}/checkpoints/global_step_{n}/actor/huggingface"
VAL = {"prompt_format": "no_think", "prompt_format_example": "action_names", "max_actions_per_step": 5,
       "action_space": "strafe", "max_episode_steps": 12}


def test_entries_are_named_from_where_the_weights_lie():
    e = S.entry(RL.format(g="d", c=9, k=1, r="d", n=300) + "/")
    assert e["name"] == "D-s1-V4" and e["group"] == "D" and e["seed"] == 1 and e["reward"] == "V4"
    assert e["step"] == 300 and e["sft"] == "D-SFT" and e["base"] == "Q3B"
    assert e["val_env"] == M + "/m3_d_s20261019/rl_m3ds1d/val_env.yaml" and not e["path"].endswith("/")
    e = S.entry(RL.format(g="e", c=9, k=2, r="s0", n=150))
    assert e["name"] == "E-s2-S0" and e["sft"] == "Q3S"  # E's RL grew from the main line's formal SFT (N1's Q3S)
    e = S.entry(M + "/sft/C/model/checkpoint-1204", matrix=M)
    assert e["name"] == "C-SFT" and e["reward"] == "SFT" and e["seed"] is None and e["sft"] is None
    assert e["val_env"].endswith("/rl_m3cs1s0/val_env.yaml")
    with pytest.raises(ValueError):
        S.entry("/elsewhere/checkpoint-5")
    with pytest.raises(ValueError, match="D-s1-V4"):
        S.make([RL.format(g="d", c=9, k=1, r="d", n=250), RL.format(g="d", c=9, k=1, r="d", n=300)])


def test_the_starts_become_q0_ask_models(tmp_path):
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"starts": [{"name": "D-s1-V4", "path": "/w/d"}, {"name": "C-SFT", "path": "/w/c"}]}))
    q0 = types.SimpleNamespace(QWEN3={"Q3B": "/b"}, MODELS=("B", "Q3B"))
    assert S.register(q0, f) == ["D-s1-V4", "C-SFT"]
    assert q0.QWEN3["D-s1-V4"] == "/w/d" and q0.MODELS == ("B", "Q3B", "D-s1-V4", "C-SFT")
    assert S.register(q0, f) == ["D-s1-V4", "C-SFT"] and len(q0.MODELS) == 4  # twice is the same
    assert S.register(q0, tmp_path / "none.json") == []
    f.write_text(json.dumps({"starts": [{"name": "Q3B", "path": "/other"}]}))
    with pytest.raises(ValueError):
        S.register(q0, f)


def weights(path: Path, preprocessor=True):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model-00001-of-00001.safetensors").write_bytes(b"")
    (path / "config.json").write_text("{}")
    if preprocessor:
        (path / "preprocessor_config.json").write_text("{}")


def val_env(path: Path, **kw):
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"envs": [{"config": dict(VAL, **kw)}]}))


def test_check_finds_missing_weights_prompts_and_sft_starts(tmp_path):
    m = str(tmp_path / "exps/matrix_v3")
    paths = [RL.format(g="d", c=9, k=1, r="s0", n=300).replace(M, m),
             RL.format(g="d", c=9, k=1, r="d", n=300).replace(M, m),
             m + "/sft/D/model/checkpoint-3987"]
    starts = S.make(paths, matrix=m)
    for e in starts:
        weights(Path(e["path"]))
    val_env(Path(starts[0]["val_env"]))
    val_env(Path(starts[1]["val_env"]))
    assert starts[2]["val_env"] == starts[0]["val_env"]
    assert S.check(starts) == []
    val_env(Path(starts[1]["val_env"]), prompt_format="free_think")
    (Path(starts[2]["path"]) / "preprocessor_config.json").unlink()
    problems = S.check(starts[:2] + [starts[2]])
    assert any("group D" in p for p in problems) and any("preprocessor_config" in p for p in problems)
    assert any("D-SFT is not evaluated" in p for p in S.check(starts[:2]))


def test_cli_make_check_val_env(tmp_path, capsys):
    m = str(tmp_path / "exps/matrix_v3")
    p = RL.format(g="c", c=9, k=1, r="s0", n=50).replace(M, m)
    weights(Path(p))
    f = tmp_path / "s.json"
    assert S.main(["make", "--paths", p, "--out", str(f), "--note", "主线：C 挑 50"]) == 0
    data = json.loads(f.read_text())
    assert data["note"] == "主线：C 挑 50" and data["starts"][0]["name"] == "C-s1-S0"
    assert S.main(["check", "--starts", str(f)]) == 1  # no val_env, and C-SFT is not there
    capsys.readouterr()
    assert S.main(["val-env", "C-s1-S0", "--starts", str(f)]) == 0
    assert capsys.readouterr().out.strip() == data["starts"][0]["val_env"]
    assert S.main(["val-env", "X", "--starts", str(f)]) == 2


def test_submit_prints_without_go_and_prepares_root_jobs_with_it(tmp_path, monkeypatch):
    import sco_job

    calls = []
    monkeypatch.setattr(sco_job, "prepare", lambda run, gpus, pool, name, payload, as_root=False:
                        calls.append((run, gpus, pool, name, payload, as_root)))
    monkeypatch.setattr(sco_job, "submit", lambda run: "pt-test")
    starts = [{"name": "D-s1-V4"}, {"name": "C-SFT"}]
    rows = S.submit(starts, tmp_path / "nm", tmp_path / "i.jsonl")
    assert [r["name"] for r in rows] == ["D-s1-V4", "C-SFT"] and not calls and "job_id" not in rows[0]
    assert rows[0]["payload"][1].endswith("downstream/n/run_nm.sh")
    assert rows[0]["payload"][2:] == [str(tmp_path / "nm/D-s1-V4"), str(tmp_path / "i.jsonl"), "D-s1-V4", "12", "4"]
    rows = S.submit(starts, tmp_path / "nm", tmp_path / "i.jsonl", names=["C-SFT"], go=True)
    assert rows[0]["job_id"] == "pt-test"
    assert calls == [(tmp_path / "nm/C-SFT_job", 1, "zoetrope", "nm-c-sft", rows[0]["payload"], True)]
    with pytest.raises(KeyError):
        S.submit(starts, tmp_path / "nm", tmp_path / "i.jsonl", names=["X"])


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def ep(i, success):
    return {"instance_id": i, "success": success, "turns": 3, "steps": 6, "invalid_turns": 0, "blocked": 1,
            "max_shaping_score": 0.2, "end": "success" if success else "turns"}


def report_fixture(tmp_path):
    ids = [f"S{k}_fov_inclusion" for k in range(10)]
    inst = tmp_path / "instances_600.jsonl"
    write(inst, [{"instance_id": i, "task_type": "fov_inclusion", "near_threshold": [], "scene_id": f"S{k // 2}"}
                 for k, i in enumerate(ids)])
    n1a, n1b, nm = tmp_path / "n1", tmp_path / "n1b", tmp_path / "nm"
    write(n1a / "Q3B/episodes.jsonl", [ep(i, False) for i in ids[:5]])
    write(n1b / "Q3B/episodes.jsonl", [ep(i, False) for i in ids[5:]])
    write(n1a / "Q3S/episodes.jsonl", [ep(i, k < 2) for k, i in enumerate(ids[:5])])
    write(n1b / "Q3S/episodes.jsonl", [ep(i, False) for i in ids[5:]])
    write(nm / "C-SFT/episodes.jsonl", [ep(i, k < 3) for k, i in enumerate(ids)])
    write(nm / "C-s1-S0/episodes.jsonl", [ep(i, k < 5) for k, i in enumerate(ids)])
    write(nm / "C-s1-V4/episodes.jsonl", [ep(i, k < 6) for k, i in enumerate(ids)])
    write(nm / "E-s1-S0/episodes.jsonl", [ep(i, k < 4) for k, i in enumerate(ids)])
    starts = [S.entry(RL.format(g="e", c=9, k=1, r="s0", n=300)), S.entry(RL.format(g="c", c=9, k=1, r="d", n=50)),
              S.entry(RL.format(g="c", c=9, k=1, r="s0", n=50)), S.entry(M + "/sft/C/model/checkpoint-1204", M)]
    return ids, inst, (n1a, n1b), nm, starts


def test_report_pairs_rl_with_its_sft_start_and_v4_with_s0(tmp_path):
    ids, inst, n1, nm, starts = report_fixture(tmp_path)
    rep = NR.build(inst, nm, starts, n1)
    # N1's Q3RS and Q3RD are not in these N1 folders: left out
    assert list(rep["models"]) == ["Q3B", "Q3S", "C-SFT", "C-s1-S0", "C-s1-V4", "E-s1-S0"]
    m = rep["models"]
    assert m["Q3S"]["rates"]["all"]["k"] == 2 and m["Q3B"]["rates"]["all"]["n"] == 10
    assert m["C-s1-S0"]["versus"]["base"] == "Q3B" and m["C-s1-S0"]["versus"]["all"]["only_a"] == 5
    assert m["C-s1-S0"]["versus_sft"]["base"] == "C-SFT" and m["C-s1-S0"]["versus_sft"]["all"]["diff"] == pytest.approx(0.2)
    assert m["C-s1-V4"]["versus_s0"]["base"] == "C-s1-S0" and m["C-s1-V4"]["versus_s0"]["all"]["only_a"] == 1
    assert m["E-s1-S0"]["versus_sft"]["base"] == "Q3S" and m["E-s1-S0"]["versus_sft"]["all"]["diff"] == pytest.approx(0.2)
    assert "versus_sft" not in m["C-SFT"] and "versus" in m["C-SFT"] and "versus_s0" not in m["C-s1-S0"]
    md = NR.markdown(rep)
    assert "## 总览" in md and "| C | SFT | 3/10" in md and "对 C-SFT +20.0" in md and "| E | S0 |" in md
    assert "同组同种子 V4 对 S0" in md and "| C-s1-V4 | C-s1-S0 |" in md and "E 对 Q3S" in md
    rep = NR.build(inst, nm, starts, n1, exclude={ids[0]})
    assert rep["instances"] == 9 and rep["models"]["C-SFT"]["rates"]["all"]["k"] == 2


def test_report_main_and_a_twice_played_instance(tmp_path):
    ids, inst, n1, nm, starts = report_fixture(tmp_path)
    f = tmp_path / "s.json"
    f.write_text(json.dumps({"starts": starts}))
    scan = tmp_path / "scan.json"
    scan.write_text(json.dumps({"evals": {"x": {"instances": {ids[9]: {}}}}}))
    out = tmp_path / "rep"
    assert NR.main(["--instances", str(inst), "--eval", str(nm), "--n1", *map(str, n1), "--starts", str(f),
                    "--exclude", str(scan), "--out", str(out)]) == 0
    assert json.loads((out / "report.json").read_text())["instances"] == 9 and (out / "report.md").is_file()
    write(nm / "C-SFT/episodes.jsonl", [ep(ids[0], True), ep(ids[0], False)])
    with pytest.raises(ValueError, match="C-SFT"):
        NR.build(inst, nm, starts, n1)


RUNNER_STUB = r"""#!/usr/bin/env bash
# a fake run_n_episodes.sh: OUT INST NAME WORKERS LIMIT
printf '%s|%s|%s|%s\n' "$3" "$4" "$5" "${N_VAL_ENV:-}" >> "$STUB_LOG"
mkdir -p "$1/frames"
n=$(grep -c . "$STUB_LOG")
for i in $(seq 1 "${STUB_EPISODES:-3}"); do
  grep -q "\"i$i\"" "$1/episodes.jsonl" 2>/dev/null && continue
  echo "{\"instance_id\": \"i$i\", \"success\": true}" >> "$1/episodes.jsonl"
  kind=clean; [[ $n -eq 1 && $i -eq 2 ]] && kind=noise
  "$STUB_PY" -c 'import sys, numpy as np; from PIL import Image
a = np.random.default_rng(0).integers(0, 256, (16, 16, 3)) if sys.argv[2] == "noise" else np.zeros((16, 16, 3))
Image.fromarray(a.astype(np.uint8)).save(sys.argv[1])' "$1/frames/i$i.png" "$kind"
done
"""


def run_nm(tmp_path, episodes: int):
    stub = tmp_path / "runner.sh"
    stub.write_text(RUNNER_STUB)
    starts = tmp_path / "s.json"
    starts.write_text(json.dumps({"starts": [{"name": "D-s1-V4", "path": "/w", "val_env": str(tmp_path / "v.yaml")}]}))
    (tmp_path / "v.yaml").write_text("x: 1\n")
    inst = tmp_path / "i.jsonl"
    inst.write_text("".join(json.dumps({"instance_id": f"i{k}"}) + "\n" for k in (1, 2, 3)))
    env = dict(os.environ, NM_RUNNER=str(stub), NM_STARTS=str(starts), N_VLM_PY=sys.executable,
               STUB_PY=sys.executable, STUB_LOG=str(tmp_path / "stub.log"), STUB_EPISODES=str(episodes))
    out = tmp_path / "out"
    rc = subprocess.run(["bash", str(ROOT / "n/run_nm.sh"), str(out), str(inst), "D-s1-V4", "12", "4"], env=env,
                        capture_output=True, text=True).returncode
    return rc, out, (tmp_path / "stub.log").read_text().splitlines()


def test_run_nm_replays_the_corrupted_episodes_with_fewer_at_once(tmp_path):
    rc, out, log = run_nm(tmp_path, 3)
    assert rc == 0
    assert log == [f"D-s1-V4|12|0|{tmp_path / 'v.yaml'}", f"D-s1-V4|4|0|{tmp_path / 'v.yaml'}"]
    assert [json.loads(x)["instance_id"] for x in (out / "episodes_corrupted.jsonl").read_text().splitlines()] == ["i2"]
    assert sorted(json.loads(x)["instance_id"] for x in (out / "episodes.jsonl").read_text().splitlines()) == \
        ["i1", "i2", "i3"]
    assert (out / "frames_scan.json").is_file() and (out / "frames_scan_repaired.json").is_file()
    scan = json.loads((out / "frames_scan_repaired.json").read_text())
    assert all(not r["instances"] for r in scan["evals"].values())
    assert "done: 3 of 3 episodes" in (out / "nm.log").read_text()


def test_run_nm_fails_when_episodes_are_missing_or_the_name_is_unknown(tmp_path):
    rc, out, log = run_nm(tmp_path, 1)
    assert rc == 1 and len(log) == 1  # nothing corrupted: no replay; 1 of 3 episodes
    env = dict(os.environ, NM_STARTS=str(tmp_path / "s.json"), N_VLM_PY=sys.executable)
    r = subprocess.run(["bash", str(ROOT / "n/run_nm.sh"), str(tmp_path / "o2"), str(tmp_path / "i.jsonl"), "X"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "X is not in the starts" in (tmp_path / "o2/nm.log").read_text()

"""WP-WBEVAL tests: vLLM server command, step regex, EvalAgent hooks, score()/summarize(), oracle, manifest, smoke.

Login node, 1 core, no GPU, no model (PY_Q3: vLLM 0.11 env with xgrammar 0.1.25, transformers 4.57.6):
    cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD \
      nice -n 19 $PY_Q3 -m pytest -q -p no:cacheprovider vpd/tests/test_wbeval.py
Scratch goes under eval_external_assets/vpd/scratch/wbeval/pytest_<rand>/ (env VPD_WBEVAL_TEST_DIR overrides),
never /tmp, and is removed at the end.  test_smoke_off_k2_and_rescore runs the fake-server smoke once (about a
minute: a sandbox copy and real 640x400 renders).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from eval_external.common.stats import wilson  # noqa: E402
from vpd import paths  # noqa: E402
from vpd.common import vllm_server_vpd as V  # noqa: E402
from vpd.walkerbench import history  # noqa: E402
from vpd.walkerbench import run_vpd as R  # noqa: E402
from vpd.walkerbench.action_regex import action_regex, reply  # noqa: E402


@pytest.fixture(scope="module")
def workdir():
    base = Path(os.environ.get("VPD_WBEVAL_TEST_DIR", paths.scratch("wbeval")))
    d = base / f"pytest_{uuid.uuid4().hex[:8]}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d)  # only what this test module created


@pytest.fixture(autouse=True)
def fresh_counts():
    R.COUNTS.clear()
    yield
    R.COUNTS.clear()


# ---------------------------------------------------------------- vLLM server --------------------------------------

def _fake_model_dir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in V.MODEL_DIR_REQUIRED:
        (d / f).write_text("{}")
    return d


def test_server_command_model_dir_and_defaults(workdir):
    md = _fake_model_dir(workdir / "merged")
    srv = V.VLLMServer("c0q3_sft_e8", log=workdir / "vllm.log", port=8123, model_dir=md, gpu_memory_utilization=0.8)
    cmd = srv.cmd
    assert cmd[cmd.index("--model") + 1] == str(md)
    assert cmd[cmd.index("--served-model-name") + 1] == "c0q3_sft_e8"
    assert json.loads(cmd[cmd.index("--limit-mm-per-prompt") + 1]) == {"image": 80}
    assert cmd[cmd.index("--max-model-len") + 1] == "128000"
    assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.8"
    assert json.loads(cmd[cmd.index("--mm-processor-kwargs") + 1]) == {
        "max_pixels": 1_024_000, "size": {"longest_edge": 1_024_000, "shortest_edge": 65536}}
    assert json.loads(cmd[cmd.index("--structured-outputs-config") + 1]) == {"backend": "xgrammar"}
    assert "--enforce-eager" in cmd and cmd[cmd.index("--generation-config") + 1] == "vllm"
    assert srv.base_url == "http://127.0.0.1:8123/v1"
    man = srv.manifest()
    assert man["model_dir"] == str(md) and man["extra_args"] == list(V.DEFAULT_EXTRA_ARGS)
    assert man["cfg"]["max_images"] == 80 and man["cfg"]["gpu_memory_utilization"] == 0.8
    env = srv.env()
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["OMP_NUM_THREADS"] == "4" == env["MKL_NUM_THREADS"]


def test_server_view_dir_free_port_and_refusals(workdir):
    srv = V.VLLMServer("qwen3vl-8B", log=workdir / "v.log")
    assert srv.model_path == paths.MODEL_VIEWS / "qwen3vl-8B"
    assert 0 < srv.port < 65536
    srv2 = V.VLLMServer("qwen3vl-8B", log=workdir / "v.log", extra_args=[])
    assert "--structured-outputs-config" not in srv2.cmd
    with pytest.raises(TypeError):
        V.VLLMServer("qwen3vl-8B", log=workdir / "v.log", max_imgs=3)
    empty = workdir / "empty_model"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        V.VLLMServer("qwen3vl-8B", log=workdir / "v.log", model_dir=empty)
    assert "--mm-processor-kwargs" not in V.VLLMServer("qwen3vl-8B", log=workdir / "v.log",
                                                       mm_processor_kwargs=None).cmd


def test_server_pixel_cap_takes_effect_in_the_processor(workdir):
    """vLLM 0.11 builds Qwen3VLProcessor.from_pretrained(model, **mm_processor_kwargs); the server's kwargs must
    cap size.longest_edge there (max_pixels alone does not in transformers 4.57.6) and keep 640x400 / 1280x800."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    tf = pytest.importorskip("transformers")
    from PIL import Image

    srv = V.VLLMServer("c0q3_sft_e8", log=workdir / "v.log")
    kw = srv.mm_processor_kwargs()
    assert kw["size"]["shortest_edge"] == 65536  # read from the view's preprocessor_config.json
    ip = tf.Qwen3VLProcessor.from_pretrained(str(srv.model_path), use_fast=True, **kw).image_processor
    assert ip.size["longest_edge"] == paths.MAX_PIXELS
    grid = {wh: ip(images=[Image.new("RGB", wh)], return_tensors="pt")["image_grid_thw"][0].tolist()
            for wh in ((640, 400), (1280, 800), (2000, 1200))}
    assert grid == {(640, 400): [1, 24, 40], (1280, 800): [1, 50, 80], (2000, 1200): [1, 48, 80]}


# ---------------------------------------------------------------- step regex ---------------------------------------

def test_step_regex_modes():
    assert R.step_regex(3, "off") is None
    assert R.step_regex(3, "empty") == action_regex(3, 0)
    short = R.step_regex(3, "short", 32)
    assert re.fullmatch(short, reply("MOVE(2)", "go left at the corner"))
    assert re.fullmatch(short, reply('STOP("ARRIVED")', "x" * 32))
    assert not re.fullmatch(short, reply("MOVE(2)", "x" * 33))
    assert not re.fullmatch(short, reply("MOVE(2)", "向左"))  # non-ASCII: could exceed 1 token per char
    assert not re.fullmatch(short, reply("MOVE(4)", "ok"))
    assert not re.fullmatch(R.step_regex(3, "empty"), reply("MOVE(2)", "ok"))
    with pytest.raises(ValueError):
        R.step_regex(3, "loose")
    with pytest.raises(ValueError):
        R.EvalOptions(constrained="short", thought_max_chars=0)


def test_short_regex_compiles_in_xgrammar():
    xgr = pytest.importorskip("xgrammar")
    for n in (0, 1, 5):
        xgr.Grammar.from_regex(R.step_regex(n, "short", 32))
        xgr.Grammar.from_regex(R.step_regex(n, "empty"))


# ---------------------------------------------------------------- EvalAgent hooks ----------------------------------

class _Completions:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def create(self, **k):
        self.calls.append(k)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),
                                                        finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100 * R.count_images(k["messages"]),
                                                     completion_tokens=3))


class _FakeBase:
    """The parts of WalkerBench's VLNAgent the hooks touch (decide_action -> _trim_history -> create)."""
    REPLIES: list = []

    def __init__(self, config=None):
        self.config = SimpleNamespace(max_tokens=16384, max_history_turns=1000, temperature=0.0)
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(self.REPLIES)))
        self.messages = [{"role": "system", "content": "sys"}]
        self.step_count = 0
        self.next_result = None
        self.stock_trims = 0

    def _trim_history(self):
        self.stock_trims += 1

    def decide_action(self, observation):
        self.messages.append({"role": "user", "content": [
            {"type": "text", "text": f"**Step {self.step_count + 1}**"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]})
        self._trim_history()
        r = self.client.chat.completions.create(model="m", messages=self.messages,
                                                max_tokens=self.config.max_tokens, temperature=0.0)
        self.messages.append({"role": "assistant", "content": r.choices[0].message.content})
        return {"type": "move", "move_id": 1, "reason": ""}

    def execute_action(self, action):
        return self.next_result

    def run(self, task_id, max_steps=25, agent_id="a"):
        return {"task": task_id}


def _agent(opts, replies, copy):
    _FakeBase.REPLIES = replies
    return R.make_eval_agent(_FakeBase, opts, copy)()


def test_agent_constrained_window_and_regex_check(workdir):
    calls_log = workdir / "agent_calls.jsonl"
    opts = R.EvalOptions(constrained="empty", history_turns=2, calls_log=calls_log,
                         prompt_dump=workdir / "agent_dump.jsonl", prompt_dump_n=2)
    good = reply("MOVE(2)")
    agent = _agent(opts, [good, good, reply("MOVE(2)", "oops"), good, good], workdir)
    assert agent.config.max_tokens == 512
    agent.run("nav_0001_x")
    obs = {"available_moves": [{"id": 1}, {"id": 2}]}
    for step in range(5):
        agent.step_count = step
        agent.decide_action(obs)
    sent = agent.client.chat.completions.calls
    assert len(sent) == 5
    for k in sent:
        assert k["max_tokens"] == 512
        assert k["extra_body"] == {"structured_outputs": {"regex": action_regex(2, 0)}}
    # the message list is shared, so check the logged pattern of each call: clean window, <= 2 previous turns
    recs = [json.loads(x) for x in open(calls_log)]
    assert [r["pattern"] for r in recs] == ["su", "suau", "suauau", "suauau", "suauau"]
    assert [r["n_images"] for r in recs] == [1, 2, 3, 3, 3]
    assert [r["regex_ok"] for r in recs] == [True, True, False, True, True]
    assert all(r["task"] == "nav_0001_x" for r in recs)
    assert R.COUNTS["constrained_calls"] == 5 and R.COUNTS["regex_mismatch"] == 1
    assert history.is_clean_window(agent.messages[:-1], 2) and agent.stock_trims == 0
    dump = [json.loads(x) for x in open(workdir / "agent_dump.jsonl")]
    assert len(dump) == 2 and dump[0]["messages"][1]["content"][1]["image_sha256"] == hashlib.sha256(
        b"\0\0\0").hexdigest()


def test_agent_off_mode_full_history(workdir):
    opts = R.EvalOptions(constrained="off", history_turns=0, max_tokens=64)
    agent = _agent(opts, ["anything", "x"], workdir)
    for step in range(2):
        agent.step_count = step
        agent.decide_action({"available_moves": []})
    for k in agent.client.chat.completions.calls:
        assert "extra_body" not in k and k["max_tokens"] == 64
    assert agent.stock_trims == 2 and R.COUNTS["constrained_calls"] == 0


def test_agent_invalid_move_crash_is_terminal(workdir):
    agent = _agent(R.EvalOptions(constrained="off"), [], workdir)
    agent.next_result = {"success": False, "observation": None, "done": False, "done_reason": None,
                         "error": "Invalid move_id: 9. Available: [1, 2]"}
    action = {"type": "move", "move_id": 9, "reason": "go"}
    out = agent.execute_action(action)
    assert out["done"] is True and out["done_reason"] == "invalid_move_crash"
    assert R.CRASH_TAG in action["reason"] and action["reason"].startswith("go ")
    assert R.step_kind(action["reason"]) == "invalid_move_crash"
    rot = {"type": "rotation", "heading": 0, "pitch": 0, "reason": ""}
    out = agent.execute_action(rot)
    assert out["done_reason"] == "action_failed" and R.step_kind(rot["reason"]) == "action_failed"
    assert R.COUNTS["invalid_move_crash"] == 1 and R.COUNTS["action_failed"] == 1
    agent.next_result = {"success": True, "observation": {"x": 1}, "done": False, "done_reason": None, "error": None}
    ok = {"type": "move", "move_id": 1, "reason": ""}
    assert agent.execute_action(ok) is agent.next_result and ok["reason"] == ""


def test_step_kind_classes():
    assert R.step_kind("All API retries failed, using fallback. [eval_external: context overflow]") == "api_failed"
    assert R.step_kind("Failed to parse response, using fallback.") == "unparseable"
    assert R.step_kind("Invalid action format: GO") == "unparseable"
    assert R.step_kind("[Repaired] x") == "repaired"
    assert R.step_kind("[Repaired] x [vpd: invalid_move_crash] Invalid move_id") == "invalid_move_crash"
    assert R.step_kind("") == "ok" and R.step_kind(None) == "ok"


# ---------------------------------------------------------------- oracle -------------------------------------------

def _hav(a, b):
    R_E = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_E * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def _fake_ev(coords):
    def reconstruct(events, start_pano=None):  # evaluate_all.reconstruct_path_from_events
        path = [start_pano] if start_pano else []
        for e in events:
            st = e.get("initial_state", {}) if e.get("event") == "session_start" else e.get("state", {})
            if st.get("pano_id") and (not path or path[-1] != st["pano_id"]):
                path.append(st["pano_id"])
        return path

    def dist(a, b, local=None):
        la = (local or {}).get(a) or coords.get(a)
        lb = (local or {}).get(b) or coords.get(b)
        return float("inf") if not la or not lb else _hav(la, lb)

    return SimpleNamespace(NAV_SUCCESS_THRESHOLD_M=15.0, reconstruct_path_from_events=reconstruct,
                           get_distance=dist)


def test_oracle_passes_by_target():
    coords = {"S": (0.0, 0.0), "P1": (0.0010, 0.0), "T": (0.00109, 0.0), "P2": (0.0030, 0.0)}
    ev = _fake_ev(coords)
    events = [{"event": "session_start", "initial_state": {"pano_id": "S", "lat": 0.0, "lng": 0.0}},
              {"event": "action", "state": {"pano_id": "S", "lat": 0.0, "lng": 0.0}},
              {"event": "action", "state": {"pano_id": "P1", "lat": 0.0010, "lng": 0.0}},
              {"event": "action", "state": {"pano_id": "P2", "lat": 0.0030, "lng": 0.0}}]
    cfg = {"spawn_point": "S", "target_pano_ids": ["T"]}
    o, d = R.oracle_for(events, cfg, "nav", ev)
    assert o == 1 and 9 < d < 11  # P1 is ~10 m from T; the final P2 is ~212 m away (success 0)
    o, d = R.oracle_for(events[:2], cfg, "vis", ev)
    assert o == 0 and d > 100
    assert R.oracle_for(events, cfg, "height", ev) == (None, None)
    assert R.oracle_for(events, {"spawn_point": "S", "target_pano_ids": ["NOWHERE"]}, "nav", ev) == (0, None)


# ---------------------------------------------------------------- summarize ----------------------------------------

def _stats(steps, crash=0, unparse=0):
    s = {k: 0 for k in ("steps_logged", "unparseable_steps", "api_failed_steps", "context_overflow_steps",
                        "invalid_move_crash_steps", "action_failed_steps", "repaired_steps", "parse_valid_steps")}
    s.update(steps_logged=steps, invalid_move_crash_steps=crash, unparseable_steps=unparse,
             parse_valid_steps=steps - crash - unparse)
    return s


def _rec(stem, success, spl, error, oracle, steps, **st):
    return {"task_id": stem, "task_type": stem.split("_")[0], "success": success, "spl": spl, "steps": steps,
            "length": 1.0, "error": error, "oracle": oracle, "oracle_error": None,
            "step_stats": _stats(steps, **st), "log": f"m_{stem}.jsonl"}


def test_summarize_synthetic():
    tasks = ["nav_a", "nav_b", "nav_c", "vis_a", "vis_b", "vis_c"]
    records = {"nav_a": _rec("nav_a", 1, 0.8, 5.0, 1, 10),
               "nav_b": _rec("nav_b", 0, 0.0, 40.0, 1, 4, crash=1),
               "nav_c": _rec("nav_c", 0, 0.0, 100.0, 0, 20, unparse=2),
               "vis_a": _rec("vis_a", 1, 0.5, 3.0, 1, 6),
               "vis_b": _rec("vis_b", 0, 0.0, -1, 0, 10)}  # vis_c: no log
    routes = {t: {"route_id": f"r_{t[-1]}"} for t in tasks}
    hard = {"nav_odo_fail": ["nav_b", "nav_c", "nav_zzz"], "vis_fwd_fail": ["vis_a", "vis_c"],
            "baselines_sha256": "abc"}
    calls = [{"task": "nav_a", "step": 1, "n_images": 1, "prompt_tokens": 100, "constrained": True, "regex_ok": True},
             {"task": "nav_a", "step": 2, "n_images": 2, "prompt_tokens": 342, "constrained": True, "regex_ok": True},
             {"task": "nav_a", "step": 3, "n_images": 3, "prompt_tokens": 584, "constrained": True,
              "regex_ok": False, "finish_reason": "length"},
             {"task": "nav_b", "step": 1, "error": "boom", "constrained": True, "n_images": 1}]
    res = R.summarize(tasks, records, routes=routes, hard=hard, calls=calls)
    p, lo, hi = wilson(2, 6)
    assert res["pooled"] == {"k": 2, "n": 6, "rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)]}
    assert res["per_kind"]["nav"]["k"] == 1 and res["per_kind"]["nav"]["n"] == 3
    assert res["per_kind"]["vis"]["k"] == 1 and res["per_kind"]["vis"]["n"] == 3
    assert set(res["per_kind"]) == {"nav", "vis"}
    assert res["type_mean"] == round(1 / 3, 4) and res["total_sr"] == 33.3
    assert res["n_scored"] == 5 and res["tasks_without_log"] == 1
    assert res["spl_mean"] == round(1.3 / 6, 4)
    assert res["final_error_mean"] == 37.0 and res["final_error_n"] == 4
    assert res["oracle"]["pooled"]["k"] == 3 and res["oracle"]["pooled"]["n"] == 6
    assert res["oracle"]["per_kind"]["nav"]["k"] == 2 and res["oracle"]["per_kind"]["vis"]["k"] == 1
    hs = res["hard_subsets"]
    assert hs["nav_odo_fail"]["k"] == 0 and hs["nav_odo_fail"]["n"] == 2 and hs["nav_odo_fail"]["n_listed"] == 3
    assert hs["nav_odo_fail"]["complete"] is False
    assert hs["vis_fwd_fail"]["k"] == 1 and hs["vis_fwd_fail"]["n"] == 2 and hs["vis_fwd_fail"]["complete"] is True
    assert res["steps_logged"] == 50 and res["invalid_move_crash"] == 1 and res["unparseable_steps"] == 2
    assert res["parse_valid_steps"] == 47 and res["parse_valid_pct"] == 94.0  # crash step in the denominator
    assert res["regex"] == {"calls": 4, "answered": 3, "errors": 1, "constrained_calls": 3, "regex_mismatch": 1,
                            "regex_valid_pct": 66.67, "finish_length": 1}
    assert res["regex_mismatch"] == 1
    d = res["prompt_tokens_per_image_delta"]
    assert d["n"] == 2 and d["min"] == d["max"] == d["median"] == 242
    rows = {r["stem"]: r for r in res["per_task_rows"]}
    assert list(rows) == tasks
    assert rows["nav_b"]["crash"] is True and rows["nav_b"]["oracle"] == 1 and rows["nav_b"]["route"] == "r_b"
    assert rows["vis_c"]["has_log"] is False and rows["vis_c"]["success"] == 0 and rows["vis_c"]["oracle"] == 0
    for r in rows.values():
        assert {"stem", "kind", "route", "success", "spl", "error", "steps", "oracle"} <= set(r)
    json.dumps(res)  # serialisable, no NaN
    assert "NaN" not in json.dumps(res)


def test_summarize_rejects_oracle_below_success():
    rec = _rec("nav_a", 1, 1.0, 2.0, 0, 3)
    with pytest.raises(AssertionError):
        R.summarize(["nav_a"], {"nav_a": rec})


def test_summarize_without_hard_and_calls():
    res = R.summarize(["vis_a"], {"vis_a": _rec("vis_a", 0, 0.0, 30.0, 0, 3)})
    assert res["hard_subsets"] is None and res["regex"]["calls"] == 0
    assert res["prompt_tokens_per_image_delta"]["n"] == 0 and res["per_kind"]["vis"]["rate"] == 0.0


def test_is_valid_unchanged():
    assert R.is_valid({"server_alive_at_end": True, "api_failed_steps": 2, "context_overflow_steps": 2})
    assert not R.is_valid({"server_alive_at_end": True, "api_failed_steps": 2, "context_overflow_steps": 1})
    assert not R.is_valid({"server_alive_at_end": False, "api_failed_steps": 0, "context_overflow_steps": 0})


# ---------------------------------------------------------------- task sets, model resolution ----------------------

def test_task_sets_and_selection():
    assert len(list(R.TASK_SETS["s2test"].glob("*.json"))) == 179
    assert len(list(R.TASK_SETS["s2val"].glob("*.json"))) == 57
    assert isinstance(R.tasks_source("s2test"), dict)  # selection.json next to tasks/
    assert R.max_step_cap(R.TASK_SETS["s2test"]) <= 80  # full history fits --max-images 80
    stems = ["vis_2", "nav_1", "nav_2", "vis_1", "nav_3"]
    assert R.select_tasks(stems, None) == sorted(stems)
    assert R.select_tasks(stems, 3) == ["nav_1", "vis_1", "nav_2"]


def test_resolve_model():
    assert R.resolve_model("Q3S") == {"arg": "Q3S", "start": "Q3S", "code": "c0q3_sft_e8", "served": "c0q3_sft_e8"}
    assert R.resolve_model("qwen3vl-8B")["start"] == "Q3B"
    assert R.resolve_model("some-code") == {"arg": "some-code", "start": None, "code": "some-code",
                                            "served": "some-code"}
    q = R.resolve_model("QRL-s1")
    assert q["code"] is None and q["served"] == "QRL-s1"


# ---------------------------------------------------------------- manifest -----------------------------------------

def test_adapter_info_and_manifest_fields(workdir):
    ad = workdir / "adapters" / "200"
    ad.mkdir(parents=True)
    (ad / "adapter.pt").write_bytes(b"abc")
    (ad / "meta.json").write_text(json.dumps({"start": "Q3S", "phase": "tier200", "seed": 1001,
                                              "code_sha256": {"x": "y"}}))
    info = R.adapter_info(ad / "meta.json", "Q3S")
    assert info["adapter_sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert (info["start"], info["tier"], info["seed"]) == ("Q3S", "tier200", 1001)
    with pytest.raises(ValueError):
        R.adapter_info(ad / "meta.json", "E2")
    md = _fake_model_dir(workdir / "merged2")
    srv = V.VLLMServer("c0q3_sft_e8", log=workdir / "v.log", port=8124, model_dir=md)
    run = workdir / "manifest_run"
    opts = R.EvalOptions(constrained="empty", history_turns=2, max_tokens=512)
    R.write_run_manifest(run, task_set="s2test", n_tasks=179, diff="--- a\n+++ b\n", render_size=(640, 400),
                         opts=opts, server=srv.manifest(), models=[{"code": "c0q3_sft_e8"}], workers=8,
                         temperature=0.0, model_dir=md, adapter=info, run_name="pilot__Q3S__200__640__empty",
                         lean=True, limit=None, lenient=False)
    man = json.loads((run / "manifest.json").read_text())
    assert man["decoding"]["max_tokens"] == 512 and man["decoding"]["structured_outputs"] is True
    assert man["backend"]["server"]["cfg"]["max_images"] == 80
    assert man["backend"]["server"]["extra_args"] == list(V.DEFAULT_EXTRA_ARGS)
    assert man["backend"]["server"]["model_dir"] == str(md)
    assert man["backend"]["render_size"] == [640, 400] and man["backend"]["history"]["turns"] == 2
    assert man["prompt"]["constrained"]["mode"] == "empty"
    assert man["prompt"]["constrained"]["example_regex_3_moves"] == action_regex(3, 0)
    vpd = man["extra"]["vpd"]
    assert vpd["model_dir"] == str(md) and vpd["adapter"]["adapter_sha256"] == info["adapter_sha256"]
    assert vpd["adapter"]["tier"] == "tier200" and vpd["adapter"]["seed"] == 1001
    assert vpd["git"]["commit"] and vpd["run_name"] == "pilot__Q3S__200__640__empty"
    assert set(vpd["code_sha256"]) == set(R.CODE_FILES) and all(vpd["code_sha256"].values())
    assert man["benchmark"]["items_expected"] == 179 and man["subset"]["kind"] == "full"


def test_main_refusals(workdir):
    with pytest.raises(SystemExit):
        R.main(["--model", "Q3S"])  # neither --run-dir nor --run-name
    with pytest.raises(SystemExit):
        R.main(["--run-dir", str(workdir / "x"), "--model", "QRL-s1"])  # no view and no --model-dir
    done = workdir / "done_run"
    done.mkdir()
    (done / "results.json").write_text("{}")
    with pytest.raises(SystemExit, match="exists"):
        R.main(["--run-dir", str(done), "--model", "Q3S"])


# ---------------------------------------------------------------- smoke (subprocess) ------------------------------

def test_smoke_off_k2_and_rescore(workdir):
    run = workdir / "smoke_off_k2"
    env = {**os.environ, "PYTHONPATH": str(paths.WORKTREE), "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    cmd = [sys.executable, "-m", "vpd.walkerbench.smoke_vpd", "--task-set", "s2test", "--limit", "4",
           "--render-size", "640x400", "--constrained", "off", "--history-turns", "2", "--run-dir", str(run),
           "--audit-processor", "", "--keep"]
    p = subprocess.run(cmd, cwd=paths.WORKTREE, env=env, capture_output=True, text=True, timeout=900)
    report = json.loads((run / "smoke_report.json").read_text())
    assert p.returncode == 0, (report["checks"], report["problems"], p.stderr[-2000:])
    before = json.loads((run / "results.json").read_text())
    p2 = subprocess.run([sys.executable, "-m", "vpd.walkerbench.run_vpd", "--rescore", "--run-dir", str(run)],
                        cwd=paths.WORKTREE, env=env, capture_output=True, text=True, timeout=600)
    assert p2.returncode == 0, p2.stderr[-2000:]
    after = json.loads((run / "results.json").read_text())
    for k in ("pooled", "per_kind", "oracle", "parse_valid_pct", "invalid_move_crash", "unparseable_steps",
              "tasks_without_log", "regex", "counts"):
        assert after[k] == before[k], k
    assert after["rescored"]["from"] == "results.json"


def test_server_env_disables_deep_gemm(monkeypatch):
    """wb-smoke pt-v9mmly5e: DeepGEMM's start-up warmup asserts in sco jobs; the server env must turn it off."""
    from vpd.common import vllm_server_vpd as V
    monkeypatch.setenv("VLLM_USE_DEEP_GEMM", "1")
    env = V.VLLMServer.env()
    assert env["VLLM_USE_DEEP_GEMM"] == "0" and env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"

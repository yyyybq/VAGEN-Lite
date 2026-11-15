"""Tests of the EXPLORATORY round-5 experiment B (forced opening actions): vpd/tvr/run_tvr_forced.py (run_tvr eval
+ --task-subset / --forced / --forced-k; run_tvr.py itself unchanged), vpd/tvr/intervene.py and the payload
vpd/jobs/tvr_intervene.sh.

Hermetic, as test_tvr_eval.py (whose fake world and helpers are reused): TVRBench's real thor_env / thor_view_env on
the fake ai2thor of tvr_fakes.py, the model = fake_server.FakeServer; the payload only with VPD_DRYRUN=1.  With
VPD_TEST_HEAD_WT = an untouched tree of the registered harness, test_outputs_equal_head runs the same fake eval with
run_tvr there and with run_tvr_forced here and compares every output file.  Login node, 1 core:
    cd TREE && env OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_tvr_intervene.py
"""

from __future__ import annotations

import base64
import collections
import difflib
import dis
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vpd import paths
from vpd.tests import test_tvr_eval as E
from vpd.tests import tvr_fakes as F
from vpd.tvr import fake_server, intervene, pilot_cfg, run_tvr
from vpd.tvr import run_tvr_forced as RF

JOBS_DIR = paths.VPD_PKG / "jobs"
SCRIPT = JOBS_DIR / "tvr_intervene.sh"
TIDS = [t["task_id"] for t in E.EVAL_TASKS]
REGISTERED = "64edd575f36b"  # run_tvr.code_sha256() of the pilot evals (pilot_tvr.HARNESS_SHA256_PREFIX)
# the registered harness's output keys (run_tvr at ea5196e8), in file order
PROTO_KEYS = ["schema", "arm", "start", "code", "tier", "adapter_sha256", "model_dir_is_merged", "task_set",
              "task_ids_sha256", "protocol", "max_model_len_used", "render", "blind_concurrency",
              "reach_index_sha256", "code_sha256", "mechanism_only"]
VPD_KEYS = ["schema", "arm", "category", "cap", "states", "replies", "action_success", "valid", "user_texts", "images",
            "target", "n_blocked", "n_unparseable", "regex_mismatch", "reach_digest", "reach_used", "reach_live_equal",
            "kin_equal", "kin_first_mismatch", "kin_mismatch_counts", "worker", "attempt", "wall_s", "protocol_sha256"]
TRAJ_KEYS = ["task_id", "dataset", "scene", "difficulty", "result", "actions", "start", "target", "raw_first_response",
             "raw_last_response", "vpd"]
CALL_KEYS = ["task_id", "t", "arm", "n_images", "prompt_tokens", "completion_tokens", "finish_reason", "reply",
             "action", "regex_ok", "parse_ok", "valid", "action_in_valid", "action_success", "pose_before",
             "pose_after", "cur_sha256", "latency_vlm_s", "latency_env_s", "retries", "utc", "error"]
DUMP_KEYS = ["task", "task_id", "step", "t", "arm", "n_images", "prompt_tokens", "model", "max_tokens",
             "structured_outputs", "messages"]
MANIFEST_KEYS = ["schema", "utc", "run_name", "arm", "start", "served_model_name", "tier", "task_set", "n_tasks",
                 "server", "pool", "settings", "code", "git", "adapter", "protocol_sha256", "argv", "host", "uid"]
META_KEYS = sorted(["run_name", "start", "code", "served_model_name", "tier", "task_set", "mechanism_only", "model_dir",
                    "adapter", "audit", "kin", "reach", "renderer", "settings", "server", "code_fingerprint",
                    "code_sha256", "protocol_sha256", "wall_seconds", "calls", "api_errors", "task_failures",
                    "failed_tasks", "requeued_tasks", "server_alive_at_end", "perf", "bfs_path", "deadline_s",
                    "deadline", "server_dead", "utc_end"]) + ["meta_keys"]
FORCED_KEYS = ["k", "file_sha256", "donor", "module_sha256", "harness_r5_sha256"]
# donor actions: TIDS[1] / TIDS[2] have fewer than 2 forceable actions (InvalidAction / Stop are skipped)
DONOR = {TIDS[0]: ["MoveAhead", "RotateLeft", "LookDown", "Stop"],
         TIDS[1]: ["InvalidAction", "MoveBack", "Stop"],
         TIDS[2]: ["Stop"],
         TIDS[3]: ["RotateRight", "InvalidAction", "MoveLeft", "MoveAhead"],
         TIDS[4]: ["LookUp", "LookDown"]}
SUBSET = [TIDS[3], TIDS[0], TIDS[1], TIDS[2]]  # not in set order on purpose; TIDS[4] left out
STOP_AT = 3  # FakeServer stop_at: the reply at turn t is Stop iff t >= 3 (forced turns are assistant turns too)


@pytest.fixture
def world(tmp_path):
    if not E.HAVE_REPO:
        pytest.skip("pinned TVRBench checkout not present")
    w = E.make_world(tmp_path)
    saved = list(sys.path)
    E._purge()
    _te, w.tv = F.import_real_tvrbench(w.fake)
    try:
        yield w
    finally:
        E._purge()
        sys.path[:] = saved


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def eval_argv(w, out, *extra, arms="c", tier="raw", task_set="eval500"):
    """test_tvr_eval.eval_args' argv (run_tvr_forced's parser takes run_tvr's eval options)."""
    return ["eval", "--start", "Q3B", "--tier", tier, "--arms", arms, "--task-set", task_set, "--out-root", str(out),
            "--envs", "2", "--lp", "2", "--profile", "envlib", "--tvr-out", str(w.tvr_out), "--tasks-dir",
            str(w.tasks_dir), "--reach", str(w.reach), "--bfs", str(w.bfs), "--houses", str(w.houses), *extra]


POOLS = []  # the pool_factory calls of the last run_eval (one session = one pool)


def run_eval(w, out, *extra, policy="stop_at", stop_after=STOP_AT, srv=None, arms="c", **kw):
    """run_tvr_forced.cmd_eval on the fake world (test_tvr_eval.run_eval's way)."""
    srv = srv or fake_server.FakeServer(policy=policy, stop_after=stop_after, model=E.SERVED)
    fac = E.local_pool(w)
    POOLS.clear()

    def counting(*args):
        POOLS.append(args)
        return fac(*args)

    a = RF.build_parser().parse_args(eval_argv(w, out, *extra, arms=arms, **kw))
    rc = RF.cmd_eval(a, ctl=run_tvr.Ctl(None), pool_factory=counting, server_factory=lambda: srv)
    return rc, srv


def write_inputs(tmp: Path, subset=SUBSET, prefix=DONOR, donor="Q3S", **over):
    tmp.mkdir(parents=True, exist_ok=True)
    sp, fp = tmp / "subset.json", tmp / "forced.json"
    sp.write_text(json.dumps(subset))
    doc = {"schema": RF.FORCED_SCHEMA, "donor": donor, "source": "/fake/donor", "prefix": prefix}
    doc.update(over)
    fp.write_text(json.dumps(doc))
    return sp, fp


def run_forced(w, out, sp, fp, ks="0,2", *extra, **kw):
    return run_eval(w, out, "--task-subset", str(sp), "--forced", str(fp), "--forced-k", ks, *extra, **kw)


def rdk(out, k, prefix="pilot", tier="raw"):
    return out / f"{prefix}__Q3B__{tier}__c__f{k}"


def first_cur_sha(req) -> str:
    users = [m for m in req["messages"] if m["role"] == "user"]
    url = [p for p in users[0]["content"] if p["type"] == "image_url"][0]["image_url"]["url"]
    return hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()


# ================================================================================================================
# run_tvr is the registered harness; run_tvr_forced's copies of its functions differ only in the marked lines
# ================================================================================================================
def test_run_tvr_is_the_registered_harness():
    assert run_tvr.code_sha256()["code_sha256"].startswith(REGISTERED)
    assert len(run_tvr.CODE_FILES) == 12 and RF.MODULE_REL not in run_tvr.CODE_FILES
    assert "vpd/tvr/intervene.py" not in run_tvr.CODE_FILES
    src = (paths.VPD_PKG / "tvr" / "run_tvr.py").read_text()
    assert "--forced" not in src and "task_subset" not in src and not hasattr(run_tvr, "forced_prefix")
    head = os.environ.get("VPD_TEST_HEAD_WT")
    if head:
        assert sha(paths.VPD_PKG / "tvr" / "run_tvr.py") == sha(Path(head) / "vpd" / "tvr" / "run_tvr.py")


COPIES = [("fake_prompt_parity", run_tvr.fake_prompt_parity, RF.fake_prompt_parity),
          ("hf_prompt_parity", run_tvr.hf_prompt_parity, RF.hf_prompt_parity),
          ("Harness.episode_image", run_tvr.Harness.episode_image, RF.Harness.episode_image),
          ("Harness._audit", run_tvr.Harness._audit, RF.Harness._audit),
          ("Harness.finalize", run_tvr.Harness.finalize, RF.Harness.finalize),
          ("cmd_eval", run_tvr.cmd_eval, RF.cmd_eval)]
OVERRIDES = {"ArmRun", "Harness", "fake_prompt_parity", "hf_prompt_parity"}  # the module's own objects by design


def _names(code) -> set:
    """The global names a code object (and its nested ones) loads (LOAD_GLOBAL / LOAD_NAME; not attributes)."""
    out = {i.argval for i in dis.get_instructions(code) if i.opname in ("LOAD_GLOBAL", "LOAD_NAME")}
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            out |= _names(c)
    return out


def test_copied_functions_are_run_tvr_verbatim_except_marked_lines():
    total = 0
    for name, orig, copy in COPIES:
        a, b = inspect.getsource(orig).splitlines(), inspect.getsource(copy).splitlines()
        marked = [ln for ln in b if ln.endswith(RF.MARK)]
        seen = 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag == "equal":
                continue
            assert tag in ("insert", "replace"), (name, tag, a[i1:i2])  # nothing of run_tvr's is dropped silently
            assert all(ln.endswith(RF.MARK) for ln in b[j1:j2]), (name, b[j1:j2])
            seen += j2 - j1
        assert marked and seen == len(marked), name  # every marked line is a difference, every difference marked
        total += len(marked)
        # every global the copy uses is run_tvr's own object (or one of the module's documented overrides)
        g = vars(run_tvr)
        for n in _names(copy.__code__):
            if n in g and n not in OVERRIDES:
                assert vars(RF).get(n) is g[n], (name, n)
    assert total < 70  # the copies stay copies
    for cls in ("ArmRun", "Harness"):
        assert issubclass(getattr(RF, cls), getattr(run_tvr, cls))
    assert RF.Refused is run_tvr.Refused and RF.Ctl is run_tvr.Ctl


def test_forced_prefix_rule_and_shared_constants():
    cases = [(["MoveAhead", "Stop"], 3, ["MoveAhead"]), (["Stop"], 2, []), ([], 1, []),
             (["InvalidAction", "MoveBack", "InvalidAction", "LookUp", "LookDown"], 2, ["MoveBack", "LookUp"]),
             (["RotateLeft", "RotateLeft"], 0, []), (["RotateLeft", "MoveAhead", "Stop"], 1, ["RotateLeft"])]
    for acts, k, want in cases:
        assert RF.forced_prefix(acts, k) == want == intervene.forced_prefix(acts, k), (acts, k)
    assert intervene.FORCED_SCHEMA == RF.FORCED_SCHEMA == "vpd.tvr.forced/1"
    assert intervene.FORCED_SKIP == RF.FORCED_SKIP == ("Stop", "InvalidAction")
    assert intervene.ACTIONS == pilot_cfg.ACTIONS == run_tvr.ACTIONS


def test_provenance_and_cli(tmp_path):
    code = run_tvr.code_sha256()["code_sha256"]
    msha = sha(paths.WORKTREE / RF.MODULE_REL)
    prov = RF.provenance()
    assert prov == {"code_sha256": code, "module_sha256": msha,
                    "harness_r5_sha256": hashlib.sha256(f"{code} {msha}".encode()).hexdigest()}
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE))
    r = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.run_tvr_forced", "provenance", "--out",
                        str(tmp_path / "p.json")], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == prov["harness_r5_sha256"], r.stderr
    got = json.loads((tmp_path / "p.json").read_text())
    assert {k: got[k] for k in prov} == prov and got["module"] == RF.MODULE_REL
    for args in (["--help"], ["eval", "--help"]):
        r = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.run_tvr_forced", *args], env=env,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0 and "usage" in r.stdout
    assert "--forced-k" in r.stdout and "--task-subset" in r.stdout


def test_intervene_is_stdlib_only():
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE))
    code = ("import sys; import vpd.tvr.intervene as m; ok = ('vpd.tvr.intervene', 'vpd.tvr'); "
            "bad = sorted(k for k in sys.modules if k.startswith('vpd.') and k not in ok); "
            "print(bad); assert not bad, bad; assert 'numpy' not in sys.modules")
    r = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr


# ================================================================================================================
# (a) without the new options every output is run_tvr's
# ================================================================================================================
@E.needs_repo
def test_default_outputs_keys_unchanged(world, tmp_path):
    out = tmp_path / "evals"
    srv = fake_server.FakeServer(policy=E.mixed_policy, model=E.SERVED)
    rc, srv = run_eval(world, out, srv=srv, arms="c,u,blind")
    assert rc == 0 and len(POOLS) == 1
    for arm in ("c", "u", "blind"):
        run = E.rd(out, arm)
        proto = json.loads((run / "protocol.json").read_text())
        assert list(proto) == PROTO_KEYS and proto["code_sha256"].startswith(REGISTERED)
        assert list(json.loads((run / "manifest.json").read_text())) == MANIFEST_KEYS
        res = json.loads((run / "results.json").read_text())
        assert res["meta_keys"] == META_KEYS and "forced" not in res and "task_subset" not in res
        assert "dump" not in res["audit"]["prompt_parity"] and res["code_sha256"].startswith(REGISTERED)
        assert not (run / "prompt_dump_model.jsonl").exists()
        for tid in TIDS:
            tr = E.traj(run, tid)
            assert list(tr) == TRAJ_KEYS and list(tr["vpd"]) == VPD_KEYS
            assert all(list(c) == CALL_KEYS for c in jl(run / "tasks" / tid / "calls.jsonl"))
        assert all(list(c) == CALL_KEYS for c in jl(run / "calls.jsonl"))
        assert all(list(d) == DUMP_KEYS for d in jl(run / "prompt_dump.jsonl"))
    assert not any("__f" in p.name for p in out.iterdir())
    assert srv.n_chat == 45  # as test_eval_three_arms_one_session


DRIVER = r"""
import json, sys
from pathlib import Path
from vpd.tests import test_tvr_eval as E, tvr_fakes as F
from vpd.tvr import fake_server, run_tvr
tmp, mode = Path(sys.argv[1]), sys.argv[2]
w = E.make_world(tmp)
E._purge()
_te, w.tv = F.import_real_tvrbench(w.fake)
srv = fake_server.FakeServer(policy=E.mixed_policy, model=E.SERVED)
if mode == "head":
    rc, srv = E.run_eval(w, tmp / "evals", srv=srv, arms="c,u,blind")
else:
    from vpd.tvr import run_tvr_forced as RF
    extra = [] if mode == "new" else ["--forced", sys.argv[3], "--forced-k", "0"]
    argv = ["eval", "--start", "Q3B", "--tier", "raw", "--arms", "c,u,blind" if mode == "new" else "c", "--task-set",
            "eval500", "--out-root", str(tmp / "evals"), "--envs", "2", "--lp", "2", "--profile", "envlib",
            "--tvr-out", str(w.tvr_out), "--tasks-dir", str(w.tasks_dir), "--reach", str(w.reach), "--bfs",
            str(w.bfs), "--houses", str(w.houses)] + extra
    rc = RF.cmd_eval(RF.build_parser().parse_args(argv), ctl=run_tvr.Ctl(None), pool_factory=E.local_pool(w),
                     server_factory=lambda: srv)
print(json.dumps({"rc": rc, "n_chat": srv.n_chat}))
"""
VOLATILE = {"utc", "utc_end", "utc_scored", "wall_s", "wall_seconds", "latency_vlm_s", "latency_env_s", "perf",
            "renderer", "pid", "startup_s", "host", "argv", "git", "pool", "server", "ready", "history",
            "cores_mean", "cores_max", "port", "worker", "attempt"}


def _strip(obj, volatile):
    if isinstance(obj, dict):
        return {k: _strip(v, volatile) for k, v in obj.items() if k not in volatile}
    if isinstance(obj, list):
        return [_strip(x, volatile) for x in obj]
    return obj


def _unforce(obj):
    """An output of a forced run without the keys forcing adds (forced, the audit's dump note)."""
    if isinstance(obj, dict):
        return {k: (sorted(set(v) - {"forced", "meta_keys"}) + ["meta_keys"] if k == "meta_keys" else _unforce(v))
                for k, v in obj.items() if k not in ("forced", "dump")}
    if isinstance(obj, list):
        return [_unforce(x) for x in obj]
    return obj


def _drive(tree: Path, tmp: Path, mode: str, *args) -> dict:
    env = dict(os.environ, PYTHONPATH=str(tree), VPD_WORKTREE=str(tree), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-c", DRIVER, str(tmp), mode, *args], env=env, capture_output=True,
                       text=True, timeout=900, cwd=str(tree))
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def _load(p: Path, rel: str, rename=None):
    text = p.read_text()
    if rename:
        text = text.replace(*rename)
    if rel.endswith(".jsonl"):
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        if rel != "perf_parts.jsonl":
            rows.sort(key=lambda r: (str(r.get("task_id") or r.get("task")), r.get("t") or 0))
        return rows
    return json.loads(text)


def _compare(da: Path, db: Path, volatile, norm=lambda x: x, rename=None, extra_b=()):
    fa = sorted(str(p.relative_to(da)) for p in da.rglob("*") if p.is_file())
    fb = sorted(str(p.relative_to(db)) for p in db.rglob("*") if p.is_file() and p.name not in extra_b)
    assert fa == fb, (da, sorted(set(fa) ^ set(fb)))
    for rel in fa:
        if rel.endswith(".png"):
            assert (da / rel).read_bytes() == (db / rel).read_bytes(), rel
            continue
        ja, jb = _load(da / rel, rel), norm(_load(db / rel, rel, rename))
        keys = (lambda j: [list(x) for x in j] if isinstance(j, list) and j and isinstance(j[0], dict)
                else list(j) if isinstance(j, dict) else None)
        assert keys(ja) == keys(jb), rel  # the same keys in the same order
        if rel != "perf_parts.jsonl":
            assert _strip(ja, volatile) == _strip(jb, volatile), rel


@pytest.mark.skipif(not os.environ.get("VPD_TEST_HEAD_WT"), reason="VPD_TEST_HEAD_WT (the registered tree) not set")
@E.needs_repo
def test_outputs_equal_head(tmp_path):
    """The same fake eval with run_tvr of the registered tree and with run_tvr_forced of this tree: (1) without the
    new options (3 arms) every output file is equal (protocol.json byte for byte) outside the volatile fields (time,
    pids, ports, worker slots, argv, the git root); (2) --forced with K = 0 (arm c, every task, no subset) is the
    registered c run plus only the keys forcing adds."""
    head = Path(os.environ["VPD_TEST_HEAD_WT"])
    assert run_tvr.code_sha256(head)["code_sha256"] == run_tvr.code_sha256()["code_sha256"]
    _sp, fp = write_inputs(tmp_path / "in", prefix=DONOR)
    w = tmp_path / "w"  # every run in tmp/w (the same absolute paths), each moved aside after
    assert _drive(head, w, "head") == {"rc": 0, "n_chat": 45}
    w.rename(tmp_path / "head")
    assert _drive(paths.WORKTREE, w, "new") == {"rc": 0, "n_chat": 45}
    w.rename(tmp_path / "new")
    assert _drive(paths.WORKTREE, w, "k0", str(fp)) == {"rc": 0, "n_chat": 15}
    w.rename(tmp_path / "k0")
    H, N, K = (tmp_path / x / "evals" for x in ("head", "new", "k0"))
    for arm in ("c", "u", "blind"):
        da, db = E.rd(H, arm), E.rd(N, arm)
        for rel in ("protocol.json", "task_ids.json"):
            assert (da / rel).read_bytes() == (db / rel).read_bytes(), (arm, rel)
        _compare(da, db, VOLATILE)
    dk = K / "pilot__Q3B__raw__c__f0"
    pk = json.loads((dk / "protocol.json").read_text())
    assert list(pk) == PROTO_KEYS + ["forced"] and list(pk["forced"]) == FORCED_KEYS and pk["forced"]["k"] == 0
    assert {k: v for k, v in pk.items() if k != "forced"} == json.loads((E.rd(H, "c") / "protocol.json").read_text())
    _compare(E.rd(H, "c"), dk, VOLATILE | {"protocol_sha256"}, norm=_unforce,
             rename=("pilot__Q3B__raw__c__f0", "pilot__Q3B__raw__c"), extra_b=("prompt_dump_model.jsonl",))


# ================================================================================================================
# (b) (c) (e) forcing
# ================================================================================================================
@E.needs_repo
def test_forced_k0_k2_one_session(world, tmp_path):
    sp, fp = write_inputs(tmp_path / "in")
    out = tmp_path / "evals"
    rc, srv = run_forced(world, out, sp, fp, ks="0,2")
    assert rc == 0 and len(POOLS) == 1 and POOLS[0][3] == out / "pilot__Q3B__raw__pool"  # one pool, one server
    pre = {k: {t: RF.forced_prefix(DONOR[t], k) for t in TIDS[:4]} for k in (0, 2)}
    assert pre[2] == {TIDS[0]: ["MoveAhead", "RotateLeft"], TIDS[1]: ["MoveBack"], TIDS[2]: [],
                      TIDS[3]: ["RotateRight", "MoveLeft"]}
    prov = RF.provenance()
    # (e) run dirs __c__f<k> (no plain __c dir), protocols differ only in forced.k
    assert sorted(p.name for p in out.iterdir() if p.is_dir()) == ["pilot__Q3B__raw__c__f0", "pilot__Q3B__raw__c__f2"]
    protos = {k: json.loads((rdk(out, k) / "protocol.json").read_text()) for k in (0, 2)}
    for k, p in protos.items():
        assert list(p) == PROTO_KEYS + ["task_subset", "forced"]
        assert p["task_subset"] == {"sha256": sha(sp), "n": 4}
        assert p["forced"] == {"k": k, "file_sha256": sha(fp), "donor": "Q3S", "module_sha256": prov["module_sha256"],
                               "harness_r5_sha256": prov["harness_r5_sha256"]}
        assert p["code_sha256"].startswith(REGISTERED)
        assert json.loads((rdk(out, k) / "task_ids.json").read_text()) == TIDS[:4]  # set order
    assert {k: v for k, v in protos[0].items() if k != "forced"} == {k: v for k, v in protos[2].items()
                                                                       if k != "forced"}
    psha = {k: hashlib.sha256(run_tvr.canon(protos[k]).encode()).hexdigest() for k in (0, 2)}
    assert psha[0] != psha[2]
    # (b) the forced turns: no request, the donor's actions as "Action: X" assistant turns, then the model
    trajs = {k: {t: E.traj(rdk(out, k), t) for t in TIDS[:4]} for k in (0, 2)}
    for k in (0, 2):
        res = json.loads((rdk(out, k) / "results.json").read_text())
        assert res["valid"] and res["n_tasks"] == 4 and res["code_sha256"] == run_tvr.code_sha256()["code_sha256"]
        assert res["meta_keys"] == sorted(set(META_KEYS[:-1]) | {"forced"}) + ["meta_keys"]
        n_forced = sum(len(v) for v in pre[k].values())
        n_model = sum(STOP_AT + 1 - len(v) for v in pre[k].values())
        assert res["forced"] == {"k": k, "donor": "Q3S", "source": "/fake/donor", "file_sha256": sha(fp),
                                 "module_sha256": prov["module_sha256"], "harness_r5_sha256": prov["harness_r5_sha256"],
                                 "skipped_actions": ["Stop", "InvalidAction"],
                                 "short_prefix_tasks": ([TIDS[1], TIDS[2]] if k == 2 else []),
                                 "forced_turns": n_forced,
                                 "this_process": {"forced_turns": n_forced, "model_calls": n_model}}
        assert res["perf"]["steps"] == n_model and res["calls"] == n_model + n_forced
        assert res["audit"]["prompt_parity"]["equal"] is True
        assert res["audit"]["prompt_parity"]["dump"].startswith("prompt_dump_model.jsonl")
        for tid in TIDS[:4]:
            tr, p = trajs[k][tid], pre[k][tid]
            assert tr["actions"][:len(p)] == p and tr["vpd"]["replies"][:len(p)] == [f"Action: {x}" for x in p]
            assert tr["actions"][len(p):] == ["RotateRight"] * (STOP_AT - len(p)) + ["Stop"]
            assert tr["vpd"]["forced"] == {"k": k, "prefix": p, "n_forced": len(p), "donor": "Q3S"}
            assert list(tr) == TRAJ_KEYS and list(tr["vpd"]) == VPD_KEYS + ["forced"]
            calls = jl(rdk(out, k) / "tasks" / tid / "calls.jsonl")
            assert [c["forced"] for c in calls] == [True] * len(p) + [False] * (STOP_AT + 1 - len(p))
            assert all(list(c) == CALL_KEYS + ["forced"] for c in calls)
            assert all(c["prompt_tokens"] is None and c["latency_vlm_s"] is None and c["regex_ok"] is True
                       for c in calls if c["forced"])
            assert all(isinstance(c["prompt_tokens"], int) for c in calls if not c["forced"])
        dump = jl(rdk(out, k) / "prompt_dump.jsonl")
        model_dump = jl(rdk(out, k) / "prompt_dump_model.jsonl")
        assert all(list(d) == DUMP_KEYS + ["forced"] for d in dump)
        assert model_dump == [d for d in dump if not d["forced"]]
        assert sum(d["forced"] for d in dump) == n_forced  # every task is kept here and every forced t is <= 2
    # the requests: f0 first (16 = 4 tasks x 4 model turns), then f2 (11); every f2 request is a model turn of a task
    # after its prefix, its history the trajectory's replies (the forced ones included)
    n0 = sum(STOP_AT + 1 - len(v) for v in pre[0].values())
    n2 = sum(STOP_AT + 1 - len(v) for v in pre[2].values())
    assert (n0, n2) == (16, 11) and srv.n_chat == n0 + n2
    by_sha = {trajs[2][t]["vpd"]["images"][0]["cur_sha256"]: t for t in TIDS[:4]}
    seen = collections.Counter()
    for req in srv.requests[n0:]:
        tid = by_sha[first_cur_sha(req)]
        asst = [m["content"] for m in req["messages"] if m["role"] == "assistant"]
        assert len(pre[2][tid]) <= len(asst) <= STOP_AT, (tid, asst)
        assert asst == trajs[2][tid]["vpd"]["replies"][:len(asst)]
        assert req["structured_outputs"] == {"regex": pilot_cfg.REGEX} and req["max_tokens"] == 16
        seen[(tid, len(asst))] += 1
    assert sorted(seen) == sorted((t, n) for t in TIDS[:4] for n in range(len(pre[2][t]), STOP_AT + 1))
    assert set(seen.values()) == {1}
    # intervene runstate / expect on these run dirs (official_check.json stands in for the TVR env step)
    h5 = prov["harness_r5_sha256"]
    rs = ["runstate", "--tasks", str(sp), "--forced", str(fp)]
    assert intervene.main(rs + ["--dir", str(rdk(out, 2)), "--k", "2", "--harness-r5", h5]) == 0
    assert intervene.main(rs + ["--dir", str(rdk(out, 2)), "--k", "0", "--harness-r5", h5]) == 3  # other K
    assert intervene.main(rs + ["--dir", str(rdk(out, 2)), "--k", "2", "--harness-r5", "0" * 64]) == 3  # other code
    assert intervene.main(rs + ["--dir", str(out / "none"), "--k", "2"]) == 0  # no run yet
    sp2, _ = write_inputs(tmp_path / "in2", subset=SUBSET[:3])
    assert intervene.main(["runstate", "--tasks", str(sp2), "--forced", str(fp), "--dir", str(rdk(out, 2)), "--k",
                           "2"]) == 3  # other task file
    for k in (0, 2):
        (rdk(out, k) / "official_check.json").write_text(json.dumps({"equal": True}))
    rep = tmp_path / "expect.json"
    argv = ["expect", "--out-root", str(out), "--prefix", "pilot", "--start", "Q3B", "--tier", "raw", "--ks", "0,2",
            "--tasks", str(sp), "--forced", str(fp), "--harness-r5", h5, "--out", str(rep)]
    assert intervene.main(argv) == 0 and json.loads(rep.read_text())["ok"] is True
    assert intervene.main(argv[:-4] + ["--harness-r5", "1" * 64, "--out", str(rep)]) == 1
    tr = E.traj(rdk(out, 2), TIDS[0])
    tr["vpd"]["forced"]["prefix"] = ["MoveAhead"]
    (rdk(out, 2) / "tasks" / TIDS[0] / "trajectory.json").write_text(json.dumps(tr))
    assert intervene.main(argv) == 1
    assert json.loads(rep.read_text())["failed"] == ["f2:trajectories_forced"]


@E.needs_repo
def test_forced_resume_and_k3_short_donor(world, tmp_path):
    """--resume: a complete K is skipped (no request); K = 3 forces at most what the donor has; another forced file
    for the same K is refused on resume (its protocol differs)."""
    sp, fp = write_inputs(tmp_path / "in", subset=[TIDS[0], TIDS[4]])
    out = tmp_path / "evals"
    rc, srv = run_forced(world, out, sp, fp, ks="3")
    assert rc == 0
    t0, t4 = E.traj(rdk(out, 3), TIDS[0]), E.traj(rdk(out, 3), TIDS[4])
    assert t0["actions"] == ["MoveAhead", "RotateLeft", "LookDown", "Stop"]  # 3 forced, then the model's Stop
    assert t4["vpd"]["forced"]["prefix"] == ["LookUp", "LookDown"] and t4["vpd"]["forced"]["n_forced"] == 2
    assert t4["actions"] == ["LookUp", "LookDown", "RotateRight", "Stop"]
    assert srv.n_chat == 1 + 2
    before = (rdk(out, 3) / "results.json").read_bytes()
    rc, srv2 = run_forced(world, out, sp, fp, "3", "--resume")
    assert rc == 0 and srv2.n_chat == 0 and (rdk(out, 3) / "results.json").read_bytes() == before
    with pytest.raises(run_tvr.Refused, match="never overwritten"):  # no --resume
        run_forced(world, out, sp, fp, ks="3")
    sp2, fp2 = write_inputs(tmp_path / "in2", subset=[TIDS[0], TIDS[4]], donor="Q3B")
    with pytest.raises(run_tvr.Refused, match="protocol"):
        run_forced(world, out, sp2, fp2, "3", "--resume")


@E.needs_repo
def test_task_subset_alone(world, tmp_path):
    sp, _fp = write_inputs(tmp_path / "in", subset=[TIDS[4], TIDS[1]])
    out = tmp_path / "evals"
    rc, srv = run_eval(world, out, "--task-subset", str(sp), stop_after=2)
    assert rc == 0
    run = E.rd(out, "c")
    proto = json.loads((run / "protocol.json").read_text())
    assert list(proto) == PROTO_KEYS + ["task_subset"] and proto["task_subset"] == {"sha256": sha(sp), "n": 2}
    assert json.loads((run / "task_ids.json").read_text()) == [TIDS[1], TIDS[4]]
    res = json.loads((run / "results.json").read_text())
    assert res["valid"] and res["n_tasks"] == 2 and "forced" not in res and res["meta_keys"] == META_KEYS
    for tid in (TIDS[1], TIDS[4]):
        assert list(E.traj(run, tid)["vpd"]) == VPD_KEYS
    assert all(list(c) == CALL_KEYS for c in jl(run / "calls.jsonl"))


@E.needs_repo
def test_forced_reply_that_does_not_parse_is_a_protocol_error(world, tmp_path, monkeypatch):
    sp, fp = write_inputs(tmp_path / "in", subset=[TIDS[0]])
    monkeypatch.setattr(RF.Harness, "_forced_reply", staticmethod(
        lambda action: {"reply": "Action: Stop", "prompt_tokens": None, "completion_tokens": None,
                        "finish_reason": None, "latency_vlm_s": None, "retries": 0}))
    out = tmp_path / "evals"
    rc, srv = run_forced(world, out, sp, fp, ks="1")
    assert rc == 1 and srv.n_chat == 0
    inv = json.loads((rdk(out, 1) / "results_invalid.json").read_text())
    assert inv["task_failures"] == 1 and "protocol: forced turn 0" in inv["failed_tasks"][0]["error"]
    assert inv["tasks_missing"] == [TIDS[0]]


# ================================================================================================================
# (d) refusals
# ================================================================================================================
@E.needs_repo
def test_refusals(world, tmp_path):
    w = world
    inp = tmp_path / "in"
    sp, fp = write_inputs(inp)

    def refused(match, *extra, **kw):
        with pytest.raises(run_tvr.Refused, match=match):
            run_eval(w, tmp_path / "e", *extra, **kw)

    bad = inp / "bad.json"
    for ids, why in (([TIDS[0], "ithor_FloorPlan99_easy_t00_s00"], "not in the task set"),
                     ([TIDS[0], TIDS[1], TIDS[0]], "duplicated"), ([], "empty"), ({"a": 1}, "not a JSON list"),
                     ([TIDS[0], 3], "not a JSON list")):
        bad.write_text(json.dumps(ids))
        refused(why, "--task-subset", str(bad))
    refused("--task-subset", "--task-subset", str(inp / "missing.json"))
    _sp, fp_miss = write_inputs(inp / "m", prefix={t: DONOR[t] for t in TIDS[:3]})
    refused("no donor entry", "--task-subset", str(sp), "--forced", str(fp_miss), "--forced-k", "1")
    _sp, fp_bad = write_inputs(inp / "b", prefix=dict(DONOR, **{TIDS[0]: ["Fly"]}))
    refused("not lists of actions", "--task-subset", str(sp), "--forced", str(fp_bad), "--forced-k", "1")
    _sp, fp_schema = write_inputs(inp / "s", schema="vpd.tvr.forced/0")
    refused("not a vpd.tvr.forced/1 file", "--task-subset", str(sp), "--forced", str(fp_schema), "--forced-k", "1")
    refused("no donor entry", "--forced", str(fp_miss), "--forced-k", "1")  # no subset: every task needs one
    for arms in ("c,u", "u", "c,blind"):
        refused("--arms c exactly", "--task-subset", str(sp), "--forced", str(fp), "--forced-k", "1", arms=arms)
    refused("needs --forced", "--forced-k", "1")
    refused("needs --forced-k", "--forced", str(fp))
    for ks in ("-1", "1,1", "a", "", "1,", "0,2,02", "1.5"):
        refused("--forced-k must be", "--task-subset", str(sp), "--forced", str(fp), "--forced-k", ks)
    assert not (tmp_path / "e").exists()  # nothing written by a refused run
    rc, _ = run_eval(w, tmp_path / "ok", "--forced", str(fp), "--forced-k", "0", stop_after=1)
    assert rc == 0 and json.loads((rdk(tmp_path / "ok", 0) / "results.json").read_text())["n_tasks"] == 5


def test_cli_refusal_exit_2(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE))
    base = [sys.executable, "-B", "-m", "vpd.tvr.run_tvr_forced", "eval", "--start", "Q3B", "--tier", "raw",
            "--task-set", "eval500", "--out-root", str(tmp_path / "o")]
    for extra in (["--arms", "c", "--forced-k", "1"], ["--arms", "u", "--forced", "x.json", "--forced-k", "1"],
                  ["--arms", "c", "--forced", "x.json", "--forced-k", "-1"]):
        r = subprocess.run(base + extra, env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 2 and "refused:" in r.stderr, (extra, r.stderr[-500:])
    assert not (tmp_path / "o").exists()
    r = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.run_tvr", "eval", "--start", "Q3B", "--tier", "raw",
                        "--arms", "c", "--task-set", "eval500", "--out-root", str(tmp_path / "o"), "--forced-k", "1"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "unrecognized arguments" in r.stderr  # run_tvr itself has no such option


# ================================================================================================================
# (f) intervene.py on small fake dirs
# ================================================================================================================
def fake_eval_dir(d: Path, start: str, succ: dict, actions: dict = None, valid=True, adapter_sha=None) -> Path:
    d.mkdir(parents=True)
    proto = {"schema": "vpd.tvr.protocol/1", "arm": "c", "start": start, "tier": "ALL", "adapter_sha256": adapter_sha}
    (d / "protocol.json").write_text(json.dumps(proto, indent=1))
    (d / "results.json").write_text(json.dumps({"valid": valid, "n_tasks": len(succ),
                                                "adapter": {"adapter_sha256": adapter_sha}}))
    (d / "per_task.jsonl").write_text("".join(json.dumps({"task_id": t, "success": s}) + "\n" for t, s in succ.items()))
    for t in succ:
        acts = (actions or {}).get(t, ["MoveAhead", "Stop"])
        tr = {"task_id": t, "actions": acts, "vpd": {"protocol_sha256": intervene.canon_sha256(proto)}}
        (d / "tasks" / t).mkdir(parents=True)
        (d / "tasks" / t / "trajectory.json").write_text(json.dumps(tr))
    return d


def fake_pilot(root: Path) -> dict:
    """runs/pilot stand-in: Q3S (A) / Q3B (B) ALL c dirs of 200 tasks: 101 only B solved, 55 only A, 20 both."""
    ids = [f"t{i:03d}" for i in range(200)]
    a = {t: int(101 <= i < 176) for i, t in enumerate(ids)}
    b = {t: int(i < 101 or 156 <= i < 176) for i, t in enumerate(ids)}
    acts = {t: ["InvalidAction", "RotateLeft", "MoveAhead", "Stop"] for t in ids[:5]}
    return {"A": fake_eval_dir(intervene.pilot_c_dir(root, "Q3S"), "Q3S", a),
            "B": fake_eval_dir(intervene.pilot_c_dir(root, "Q3B"), "Q3B", b, actions=acts), "ids": ids}


def test_intervene_sets_pick_donors(tmp_path, capsys):
    P = tmp_path / "pilot"
    fp_ = fake_pilot(P)
    A, B = fp_["A"], fp_["B"]
    I = tmp_path / "inputs"
    m = intervene.main
    sets = ["sets", "--a-dir", str(A), "--b-dir", str(B), "--out-dir", str(I), "--a-name", "T55", "--b-name", "T101"]
    assert m(sets) == 0
    t101, t55 = json.loads((I / "T101.json").read_text()), json.loads((I / "T55.json").read_text())
    assert t101 == fp_["ids"][:101] and t55 == fp_["ids"][101:156]
    man = json.loads((I / "sets_manifest.json").read_text())
    assert man["schema"] == intervene.SETS_SCHEMA and man["n_tasks"] == 200 and man["both"] == 20
    assert man["neither"] == 24 and "utc" not in man
    assert man["sets"]["T101"] == {"file": "T101.json", "rule": "B success 1 and A success 0", "n": 101,
                                   "sha256": sha(I / "T101.json")}
    assert man["a"]["dir"] == str(A) and man["a"]["per_task_sha256"] == sha(A / "per_task.jsonl")
    # deterministic outputs: refused when present, --overwrite rewrites the same content, --replace changes it
    before = {p.name: p.read_bytes() for p in I.iterdir()}
    assert m(sets) == 2
    assert m(sets + ["--overwrite"]) == 0 and {p.name: p.read_bytes() for p in I.iterdir()} == before
    assert m(["sets", "--a-dir", str(A), "--b-dir", str(B), "--out-dir", str(I)]) == 2  # the manifest would change
    assert not (I / "only_a.json").exists()  # all outputs or none
    pick = ["pick", "--sets", f"{I}/T101.json,{I}/T55.json", "--out", str(I / "smoke.json")]
    assert m(pick + ["--n-each", "4"]) == 0 and len(json.loads((I / "smoke.json").read_text())) == 8
    assert m(pick + ["--n-each", "4", "--overwrite"]) == 0
    assert m(pick + ["--n-each", "3", "--overwrite"]) == 2  # other content: --replace needed
    assert m(pick + ["--n-each", "3", "--replace"]) == 0 and len(json.loads((I / "smoke.json").read_text())) == 6
    assert m(pick + ["--n-each", "4", "--replace"]) == 0
    # refusals: different task sets, an invalid result
    C = fake_eval_dir(tmp_path / "C", "Q3B", {t: 0 for t in fp_["ids"][:150]})
    D = fake_eval_dir(tmp_path / "D", "Q3B", {t: 0 for t in fp_["ids"]}, valid=False)
    for other in (C, D):
        assert m(["sets", "--a-dir", str(A), "--b-dir", str(other), "--out-dir", str(tmp_path / "x")]) == 2
    assert not (tmp_path / "x").exists()
    # donors: deterministic, the donor trajectories' actions
    out = I / "forced_Q3S_T101.json"
    don = ["donors", "--donor-dir", str(B), "--donor", "Q3B", "--tasks", str(I / "T101.json"), "--out", str(out)]
    assert m(don) == 0
    doc = json.loads(out.read_text())
    assert set(doc) == {"schema", "donor", "source", "prefix", "source_sha256", "tasks_file", "tasks_sha256",
                        "source_protocol_sha256"}
    assert doc["schema"] == RF.FORCED_SCHEMA and doc["donor"] == "Q3B" and doc["source"] == str(B)
    assert doc["prefix"]["t000"] == ["InvalidAction", "RotateLeft", "MoveAhead", "Stop"] and len(doc["prefix"]) == 101
    assert doc["source_sha256"]["t000"] == sha(B / "tasks" / "t000" / "trajectory.json")
    assert doc["tasks_sha256"] == sha(I / "T101.json") and doc["source_protocol_sha256"] == sha(B / "protocol.json")
    first = out.read_bytes()
    assert m(don) == 2 and m(don + ["--overwrite"]) == 0 and out.read_bytes() == first
    assert RF.load_forced(out, ["t000"])["actions"] == {"t000": ["InvalidAction", "RotateLeft", "MoveAhead", "Stop"]}
    assert m(["donors", "--donor-dir", str(B), "--donor", "Q3S", "--tasks", str(I / "T101.json"), "--out",
              str(tmp_path / "f.json")]) == 2  # the dir is Q3B's
    tr = json.loads((B / "tasks" / "t001" / "trajectory.json").read_text())
    tr["vpd"]["protocol_sha256"] = "0" * 64  # a trajectory of another run
    (B / "tasks" / "t001" / "trajectory.json").write_text(json.dumps(tr))
    assert m(don[:-1] + [str(tmp_path / "g.json")]) == 2 and not (tmp_path / "g.json").exists()
    capsys.readouterr()


def _inputs_for_check(tmp_path):
    P = tmp_path / "pilot"
    fp_ = fake_pilot(P)
    I = tmp_path / "inputs"
    m = intervene.main
    assert m(["sets", "--a-dir", str(fp_["A"]), "--b-dir", str(fp_["B"]), "--out-dir", str(I), "--a-name", "T55",
              "--b-name", "T101"]) == 0
    assert m(["pick", "--sets", f"{I}/T101.json,{I}/T55.json", "--n-each", "4", "--out", str(I / "smoke.json")]) == 0
    for s in ("T101", "T55", "smoke"):
        assert m(["donors", "--donor-dir", str(fp_["B"]), "--donor", "Q3B", "--tasks", str(I / f"{s}.json"), "--out",
                  str(I / f"forced_Q3S_{s}.json")]) == 0
    return P, I, fp_


def check_argv(P, I, s="T101", start="Q3S", donor="Q3B", out=None, **over):
    args = {"--set": s, "--tasks": str(I / f"{s}.json"), "--sets-manifest": str(I / "sets_manifest.json"),
            "--pilot-root": str(P), "--start": start, "--forced": str(I / f"forced_{start}_{s}.json"),
            "--donor": donor, "--source": str(intervene.pilot_c_dir(P, donor)), "--out": str(out)}
    args.update(over)
    return ["check"] + [x for kv in args.items() for x in kv]


def test_intervene_check(tmp_path, capsys):
    P, I, fp_ = _inputs_for_check(tmp_path)
    rep = tmp_path / "guard.json"
    m = intervene.main
    for s in ("T101", "T55", "smoke"):
        assert m(check_argv(P, I, s, out=rep)) == 0, json.loads(rep.read_text())["problems"]
    g = json.loads(rep.read_text())
    assert g["ok"] and g["files"] == {str(I / "smoke.json"): sha(I / "smoke.json"),
                                      str(I / "forced_Q3S_smoke.json"): sha(I / "forced_Q3S_smoke.json"),
                                      str(I / "sets_manifest.json"): sha(I / "sets_manifest.json")}

    def problems(*args, **over):
        assert m(check_argv(P, I, *args, out=rep, **over)) == 2
        return " ".join(json.loads(rep.read_text())["problems"])

    assert "unknown set" in problems("T101", **{"--set": "T7"})
    assert "not the manifest's T101" in problems("T101", **{"--tasks": str(I / "T55.json"), "--forced":
                                                            str(I / "forced_Q3S_T55.json")})
    assert "another task file" in problems("T101", **{"--forced": str(I / "forced_Q3S_T55.json")})
    assert "the other pilot start" in problems("T101", start="Q3B", **{"--forced": str(I / "forced_Q3S_T101.json")})
    assert "donor's pilot dir" in problems("T101", **{"--source": str(fp_["A"])})
    # the smoke set: 8 tasks of T101 + T55
    (I / "smoke7.json").write_text(json.dumps(json.loads((I / "smoke.json").read_text())[:7]))
    assert "smoke set must be 8" in problems("smoke", **{"--tasks": str(I / "smoke7.json")})
    # a manifest of other A / B dirs
    O = tmp_path / "other"
    shutil.copytree(fp_["A"], O / "A")
    shutil.copytree(fp_["B"], O / "B")
    assert m(["sets", "--a-dir", str(O / "A"), "--b-dir", str(O / "B"), "--out-dir", str(O / "in"), "--a-name",
              "T55", "--b-name", "T101"]) == 0
    assert "a dir" in problems("T101", **{"--sets-manifest": str(O / "in" / "sets_manifest.json")})
    # a prefix that is not the donor trajectory's actions (the forced file rewritten by hand)
    f = I / "forced_Q3S_T101.json"
    doc = json.loads(f.read_text())
    doc["prefix"]["t002"] = ["MoveAhead", "MoveAhead", "Stop"]
    f.write_text(json.dumps(doc))
    assert "prefixes are not the donor trajectory's actions" in problems("T101")
    # a donor trajectory changed after the forced file was built
    assert m(["donors", "--donor-dir", str(fp_["B"]), "--donor", "Q3B", "--tasks", str(I / "T101.json"), "--out",
              str(f), "--replace"]) == 0
    tr = fp_["B"] / "tasks" / "t003" / "trajectory.json"
    tr.write_text(tr.read_text() + " ")
    assert "changed since the forced file" in problems("T101")
    capsys.readouterr()


def test_intervene_adapter_match(tmp_path):
    ad = tmp_path / "adapters" / "ALL"
    ad.mkdir(parents=True)
    (ad / "adapter.pt").write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    ok = fake_eval_dir(tmp_path / "ok", "Q3S", {"t1": 1}, adapter_sha=good)
    other = fake_eval_dir(tmp_path / "other", "Q3S", {"t1": 1}, adapter_sha="0" * 64)
    assert intervene.main(["adapter-match", "--adapter-dir", str(ad), "--eval-dir", str(ok)]) == 0
    assert intervene.main(["adapter-match", "--adapter-dir", str(ad), "--eval-dir", str(other)]) == 2
    assert intervene.main(["adapter-match", "--adapter-dir", str(tmp_path), "--eval-dir", str(ok)]) == 2


def test_intervene_runstate_without_protocol(tmp_path):
    sp, fp = write_inputs(tmp_path / "in")
    d = tmp_path / "run"
    (d / "tasks" / "x").mkdir(parents=True)
    argv = ["runstate", "--dir", str(d), "--k", "1", "--tasks", str(sp), "--forced", str(fp)]
    assert intervene.main(argv) == 3  # a run without protocol.json
    shutil.rmtree(d / "tasks")
    assert intervene.main(argv) == 0  # an empty dir: no run yet


PILOT = paths.TVR_OUT / "runs" / "pilot"


@pytest.mark.skipif(not all((intervene.pilot_c_dir(PILOT, s) / "per_task.jsonl").is_file() for s in ("Q3S", "Q3B")),
                    reason="the pilot eval dirs are not mounted here")
def test_real_pilot_inputs(tmp_path):
    """Read only: the pilot's Q3S / Q3B ALL c dirs give T101 = 101 tasks (Q3B solved, Q3S failed) and T55 = 55; the
    donor files and the in-job check pass on them; each pilot ALL adapter is the one its ALL c eval served."""
    I = tmp_path / "inputs"
    m = intervene.main
    A, B = intervene.pilot_c_dir(PILOT, "Q3S"), intervene.pilot_c_dir(PILOT, "Q3B")
    assert m(["sets", "--a-dir", str(A), "--b-dir", str(B), "--out-dir", str(I), "--a-name", "T55", "--b-name",
              "T101"]) == 0
    t101, t55 = json.loads((I / "T101.json").read_text()), json.loads((I / "T55.json").read_text())
    assert (len(t101), len(t55)) == (101, 55) and not set(t101) & set(t55)
    assert m(["pick", "--sets", f"{I}/T101.json,{I}/T55.json", "--n-each", "4", "--out", str(I / "smoke.json")]) == 0
    for start, donor in (("Q3S", "Q3B"), ("Q3B", "Q3S")):
        for s in ("T101", "T55", "smoke"):
            assert m(["donors", "--donor-dir", str(intervene.pilot_c_dir(PILOT, donor)), "--donor", donor, "--tasks",
                      str(I / f"{s}.json"), "--out", str(I / f"forced_{start}_{s}.json")]) == 0
            assert m(check_argv(PILOT, I, s, start=start, donor=donor, out=I / "guard.json")) == 0, \
                json.loads((I / "guard.json").read_text())["problems"]
        assert m(["adapter-match", "--adapter-dir", str(PILOT / start / "adapters" / "ALL"), "--eval-dir",
                  str(intervene.pilot_c_dir(PILOT, start))]) == 0
    doc = json.loads((I / "forced_Q3S_T101.json").read_text())
    assert set(doc["prefix"]) == set(t101) and all(a != "InvalidAction" for v in doc["prefix"].values() for a in v)


# ================================================================================================================
# (g) the payload
# ================================================================================================================
_KNOBS = ("VPD_THREADS", "VPD_JOB_DEADLINE_S", "VPD_MIN_STEP_S", "VPD_MERGE_DIR", "VPD_RERUN_STALE", "VPD_DRYRUN",
          "RUN", "VPD_TVR_OUT", "VPD_DRY_NTOP", "VPD_WORKTREE", "VPD_TVR_PY", "VPD_TVR_ROOT")


def _bash(args, extra, cwd):
    env = {k: v for k, v in os.environ.items() if k not in _KNOBS}
    env.update(extra)
    return subprocess.run(["bash", *args], capture_output=True, text=True, env=env, cwd=str(cwd), timeout=300)


def _dry(out: str):
    return [ln[len("[dry] "):] for ln in out.splitlines() if ln.startswith("[dry] ")]


def dry(tmp_path, start, sset, merge_dir="/dev/shm/vpd_merged", rc=0):
    run = tmp_path / "jobs" / f"tvr-xint-{start.lower()}-{sset.lower()}-202611140000"
    res = _bash([str(SCRIPT), start, sset, str(run)], {"VPD_DRYRUN": "1", "VPD_TVR_OUT": str(tmp_path / "tvr"),
                                                       "VPD_MERGE_DIR": merge_dir}, tmp_path)
    assert res.returncode == rc, res.stderr + res.stdout
    return res, _dry(res.stdout), run


def test_payload_bash_syntax_mode_and_usage(tmp_path):
    r = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert os.access(SCRIPT, os.X_OK) and SCRIPT.stat().st_mode & 0o777 == 0o755
    r_ = str(tmp_path / "r")
    for args in (["Q3X", "T101", r_], ["Q3S", "T7", r_], ["E2", "T55", r_], ["Q3S", "T101"], []):
        rr = _bash([str(SCRIPT), *args], {"VPD_DRYRUN": "1"}, tmp_path)
        assert rr.returncode == 2 and "usage" in rr.stderr, (args, rr.stderr)
    text = SCRIPT.read_text()
    code = [ln.split(" #")[0].rstrip() for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    assert not [ln for ln in code if "explore/rl" in ln]  # never the RL root
    pilot = [ln for ln in code if "runs/pilot" in ln]
    assert pilot == ["PILOT=$TVR/runs/pilot"], pilot  # read only: AD, OWN_DIR, DONOR_DIR below it
    hook = [ln for ln in code if "VPD_TVR_OUT" in ln]
    assert 'if [[ $DRY == 1 && -n ${VPD_TVR_OUT:-} ]]; then' in text and len(hook) == 2
    assert "vpd.tvr.run_tvr eval" not in text and "vpd.tvr.run_tvr_forced eval" in text
    assert not [ln for ln in text.splitlines() if len(ln) > 120]


def _kinds(lines):
    out = []
    for ln in lines:
        if "intervene check" in ln:
            out.append("guard")
        elif "intervene expect" in ln:
            out.append("expect")
        elif "intervene runstate" in ln:
            out.append("runstate:" + ln.split("--k ")[1].split()[0])
        elif "intervene adapter-match" in ln:
            out.append("adapter-match")
        elif "run_tvr_forced provenance" in ln:
            out.append("provenance")
        elif ln.startswith("vpd_snapshot_file"):
            out.append("prereg_committed")
        elif "x_adapter_check" in ln:
            out.append("adapter")
        elif "run_tvr replay" in ln:
            out.append("replay")
        elif "render_parity" in ln:
            out.append("render_parity")
        elif "parity-ok" in ln:
            out.append("parity-ok")
        elif "merge.py" in ln:
            out.append("merge")
        elif "vpd.tvr.merge_parity" in ln:
            out.append("merge_parity")
        elif "run_tvr_forced eval" in ln:
            out.append("eval")
        elif ln.startswith("reap_unity"):
            out.append("reap:" + ln.split()[1])
        elif ln.startswith("rm -rf"):
            out.append("rm")
        elif "score_tvr official" in ln:
            out.append("official:" + ln.split("--run-dir ")[1].split()[0].rsplit("__", 1)[1])
        elif ln.startswith("date -u >"):
            out.append("DONE")
        else:
            out.append("?" + ln[:60])
    return out


def test_payload_dry_q3s_t101(tmp_path):
    res, lines, run = dry(tmp_path, "Q3S", "T101")
    T = tmp_path / "tvr"
    P = T / "runs" / "explore" / "intervene" / "Q3S" / "T101"
    I = T / "runs" / "explore" / "intervene" / "inputs"
    ev = [ln for ln in lines if "run_tvr_forced eval" in ln]
    assert len(ev) == 1
    assert re.fullmatch(
        r"timeout -k 30 (\d+) " + re.escape(
            f"{paths.PY_Q3} -B -m vpd.tvr.run_tvr_forced eval --start Q3S --tier ALL --arms c --task-set eval500 "
            f"--name-prefix xint --out-root {P}/evals --model-dir /dev/shm/vpd_merged/xint__Q3S__T101__ALL "
            f"--adapter-meta {T}/runs/pilot/Q3S/adapters/ALL/meta.json --task-subset {I}/T101.json --forced "
            f"{I}/forced_Q3S_T101.json --forced-k 0,1,2,3 --resume --deadline-s ") + r"(\d+)", ev[0]), ev[0]
    t, dl = (int(x) for x in re.fullmatch(r"timeout -k 30 (\d+) .* --deadline-s (\d+)", ev[0]).groups())
    assert t <= 8160 and t - dl == 240
    assert _kinds(lines) == ["prereg_committed", "provenance", "guard", "replay", "reap:canary_replay",
                             "render_parity", "adapter", "adapter-match", "runstate:0", "runstate:1", "runstate:2",
                             "runstate:3", "parity-ok", "rm", "merge", "merge_parity", "eval", "reap:eval_ALL", "rm",
                             "official:f0", "official:f1", "official:f2", "official:f3", "expect", "DONE"]
    g = next(ln for ln in lines if "intervene check" in ln)
    assert g.endswith(f"check --set T101 --tasks {I}/T101.json --sets-manifest {I}/sets_manifest.json --pilot-root "
                      f"{T}/runs/pilot --start Q3S --forced {I}/forced_Q3S_T101.json --donor Q3B --source "
                      f"{T}/runs/pilot/Q3B/evals/pilot__Q3B__ALL__c --out {run}/guard_inputs.json")
    ad = next(ln for ln in lines if "x_adapter_check" in ln)
    assert ad.endswith(" Q3S ALL 0W") and "adapter_root=L.pilot_root" in ad and str(paths.PY_EVAL) in ad
    am = next(ln for ln in lines if "adapter-match" in ln)
    assert am.endswith(f"--adapter-dir {T}/runs/pilot/Q3S/adapters/ALL --eval-dir "
                       f"{T}/runs/pilot/Q3S/evals/pilot__Q3S__ALL__c")
    rs = [ln for ln in lines if "intervene runstate" in ln]
    assert all(ln.endswith(f"--dir {P}/evals/xint__Q3S__ALL__c__f{k} --k {k} --tasks {I}/T101.json --forced "
                           f"{I}/forced_Q3S_T101.json --harness-r5 <harness_r5>") for k, ln in enumerate(rs))
    mp = next(ln for ln in lines if "vpd.tvr.merge_parity" in ln)
    assert (f"--start Q3S --adapter {T}/runs/pilot/Q3S/adapters/ALL --merged /dev/shm/vpd_merged/xint__Q3S__T101__ALL "
            f"--demos {T}/tiers/v1/warmup0w20.jsonl --n 20 --out {P}/parity/merge_ALL.json") in mp
    assert lines.index(mp) < lines.index(ev[0])  # merge parity before vLLM
    assert [ln for ln in lines if ln.startswith("rm -rf")] == ["rm -rf -- /dev/shm/vpd_merged/xint__Q3S__T101__ALL"] * 2
    assert f"--run-dir {P}/canary/sft2 --images-dir {P}/canary/sft2/images" in res.stdout
    offs = [ln for ln in lines if "score_tvr official" in ln]
    assert all(f"--run-dir {P}/evals/xint__Q3S__ALL__c__f{k} --served <code:Q3S> --bfs {T}/bfs/eval500_opt.json"
               in ln for k, ln in enumerate(offs))
    ex = next(ln for ln in lines if "intervene expect" in ln)
    assert ex.endswith(f"expect --out-root {P}/evals --prefix xint --start Q3S --tier ALL --ks 0,1,2,3 --tasks "
                       f"{I}/T101.json --forced {I}/forced_Q3S_T101.json --harness-r5 <harness_r5> --out "
                       f"{run}/expect.json")
    steps = [json.loads(s) for s in (run / "steps.jsonl").read_text().splitlines()]
    assert all(s["rc"] == 0 for s in steps) and not (run / "DONE").exists()
    assert not T.exists()  # a dry run writes nothing under OUT/tvr
    assert "runs/explore/rl" not in res.stdout and "pilot__Q3S__ALL__c__" not in res.stdout


def test_payload_dry_q3b_smoke(tmp_path):
    res, lines, run = dry(tmp_path, "Q3B", "smoke")
    T = tmp_path / "tvr"
    I = T / "runs" / "explore" / "intervene" / "inputs"
    ev = next(ln for ln in lines if "run_tvr_forced eval" in ln)
    assert (f"--start Q3B --tier ALL --arms c --task-set eval500 --name-prefix xint --out-root "
            f"{T}/runs/explore/intervene/Q3B/smoke/evals --model-dir /dev/shm/vpd_merged/xint__Q3B__smoke__ALL") in ev
    assert f"--task-subset {I}/smoke.json --forced {I}/forced_Q3B_smoke.json --forced-k 0,2 --resume" in ev
    offs = [ln.split("--run-dir ")[1].split()[0].rsplit("__", 1)[1] for ln in lines if "score_tvr official" in ln]
    assert offs == ["f0", "f2"]
    assert f"--set smoke --tasks {I}/smoke.json" in res.stdout
    assert f"--donor Q3S --source {T}/runs/pilot/Q3S/evals/pilot__Q3S__ALL__c" in res.stdout


@pytest.mark.parametrize("bad", ["/var/tmp/vpd_merged", "/tmp/vpd_merged/../etc", "/tmp/vpd_mergedX", "relative",
                                 "/scratch/vpd_merged"])
def test_payload_merge_dir_must_be_job_local(tmp_path, bad):
    res, lines, run = dry(tmp_path, "Q3S", "T101", merge_dir=bad, rc=1)
    steps = {json.loads(s)["name"]: json.loads(s)["rc"] for s in (run / "steps.jsonl").read_text().splitlines()}
    assert steps["merge_disk"] != 0 and steps["merge_ALL:no_disk"] != 0
    assert not [ln for ln in lines if "merge.py" in ln or "run_tvr_forced eval" in ln]
    assert "is not /tmp/vpd_merged or /dev/shm/vpd_merged" in res.stderr


def test_payload_merge_dir_below_the_job_local_roots(tmp_path):
    res, lines, run = dry(tmp_path, "Q3S", "T55", merge_dir="/tmp/vpd_merged/r5")
    assert "--model-dir /tmp/vpd_merged/r5/xint__Q3S__T55__ALL" in next(ln for ln in lines if "run_tvr_forced" in ln
                                                                          and " eval " in ln)

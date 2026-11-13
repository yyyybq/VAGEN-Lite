"""Integration checks of the wave-2 TVR pilot code (CONTRACT_wave2 section 9 "Integration", section 11 item 3).

Runs only in a tree that holds every wave-2 module (the combined tree); in a package's own staging tree, where
another package's module does not exist yet, the whole file is skipped (module-level skip, no stubs).  Hermetic:
fakes and temp dirs; real files are only read (the pinned prereg, the README, the pinned TVRBench checkout and, once
the data job has run, OUT/tvr/{reach,floors}); no GPU, network beyond 127.0.0.1, Unity or model.
    cd TREE && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 ... nice -n 19 env VPD_WORKTREE=TREE PYTHONPATH=TREE \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider --basetemp=$SCRATCH/pt vpd/tests/test_tvr_integration.py
Covered: every wave-2 CLI answers --help; the prereg's TVR_FROZEN block == pilot_cfg.FROZEN (the single parser);
the trainer fingerprint prefix; README coverage of the wave-2 files; each payload's dry-run step names ==
pilot_tvr.required_steps (the expect list the job checks); recheck_tvr floors == the data job's floors_eval500.json
(real data, once present); and THE PIPELINE on a fake TVR tree (module fixture `pipe`, see the block below): every
data step, run_tvr evals of three starts with FakeServer, TVRBench's own aggregate, a second-process render canary,
then gate_tvr and power_sim on the files those steps wrote (run_tvr's real results.json layout).
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from vpd import paths

WAVE2_MODULES = ("vpd.tvr.pilot_cfg", "vpd.tvr.tvr_text", "vpd.tvr.tasks", "vpd.tvr.kin_env", "vpd.tvr.bfs",
                 "vpd.tvr.reach", "vpd.tvr.floors", "vpd.tvr.demos", "vpd.tvr.thor_worker", "vpd.tvr.run_tvr",
                 "vpd.tvr.score_tvr", "vpd.tvr.parity_tvr", "vpd.tvr.fake_server", "vpd.tvr.render_parity",
                 "vpd.tvr.pilot_tvr", "vpd.tvr.gate_tvr", "vpd.tvr.recheck_tvr", "vpd.tvr.merge_parity",
                 "vpd.common.power_sim", "vpd.tests.tvr_fakes")
_MISSING = [m for m in WAVE2_MODULES if importlib.util.find_spec(m) is None]
if _MISSING:
    pytest.skip(f"not the combined tree: {', '.join(_MISSING)} missing", allow_module_level=True)

from vpd.tests import tvr_fakes as F  # noqa: E402
from vpd.tests import test_tvr_jobs as TJ  # noqa: E402
from vpd.tvr import bfs, demos, fake_server, floors, kin_env, lvp, reach, render_parity, run_tvr  # noqa: E402
from vpd.tvr import gate_tvr as G  # noqa: E402
from vpd.tvr import pilot_cfg  # noqa: E402
from vpd.tvr import pilot_tvr as PT  # noqa: E402
from vpd.tvr import recheck_tvr as RC  # noqa: E402
from vpd.tvr import score_tvr, thor_worker  # noqa: E402
from vpd.tvr import tasks as TK  # noqa: E402

# section 2: every file of the five packages (README coverage)
WAVE2_FILES = (
    "vpd/tvr/tvr_text.py", "vpd/tvr/tasks.py", "vpd/tvr/kin_env.py", "vpd/tvr/bfs.py", "vpd/tvr/reach.py",
    "vpd/tvr/floors.py", "vpd/tvr/demos.py", "vpd/tests/tvr_fakes.py", "vpd/tests/test_tvr_data.py",
    "vpd/tvr/thor_worker.py", "vpd/tvr/run_tvr.py", "vpd/tvr/score_tvr.py", "vpd/tvr/parity_tvr.py",
    "vpd/tvr/fake_server.py", "vpd/tvr/render_parity.py", "vpd/tests/test_tvr_eval.py",
    "vpd/tvr/pilot_tvr.py", "vpd/tvr/gate_tvr.py", "vpd/tvr/recheck_tvr.py", "vpd/tvr/merge_parity.py",
    "vpd/jobs/tvr_data.sh", "vpd/jobs/tvr_cal.sh", "vpd/jobs/tvr_cal400.sh", "vpd/jobs/tvr_mech_smoke.sh",
    "vpd/jobs/tvr_train.sh", "vpd/jobs/tvr_eval.sh", "vpd/tests/test_tvr_jobs.py", "vpd/tests/test_tvr_integration.py",
    "vpd/PREREG_TVR_PILOT.md", "vpd/tvr/pilot_cfg.py", "vpd/common/power_sim.py", "vpd/tests/test_tvr_prereg.py",
    "vpd/tests/test_power_sim.py", "vpd/tests/test_wave1c.py")
CLI_MODULES = ("vpd.tvr.tasks", "vpd.tvr.reach", "vpd.tvr.bfs", "vpd.tvr.floors", "vpd.tvr.demos", "vpd.tvr.run_tvr",
               "vpd.tvr.score_tvr", "vpd.tvr.parity_tvr", "vpd.tvr.render_parity", "vpd.tvr.pilot_tvr",
               "vpd.tvr.gate_tvr", "vpd.tvr.recheck_tvr", "vpd.tvr.merge_parity", "vpd.common.power_sim")


@pytest.mark.parametrize("mod", CLI_MODULES)
def test_every_wave2_cli_answers_help(mod):
    res = subprocess.run([sys.executable, "-B", "-m", mod, "--help"], capture_output=True, text=True, timeout=180,
                         env=dict(os.environ, PYTHONPATH=str(paths.WORKTREE), OMP_NUM_THREADS="1"),
                         cwd=str(paths.WORKTREE))
    assert res.returncode == 0 and "usage" in (res.stdout + res.stderr).lower(), (mod, res.stderr[-800:])


def test_files_of_section_2_exist_and_payload_modes():
    for rel in WAVE2_FILES:
        assert (paths.WORKTREE / rel).is_file(), rel
    for rel in WAVE2_FILES:
        if rel.startswith("vpd/jobs/"):
            assert (paths.WORKTREE / rel).stat().st_mode & 0o777 == 0o755, rel


def test_prereg_block_equals_pilot_cfg():
    prereg = paths.WORKTREE / "vpd" / "PREREG_TVR_PILOT.md"
    assert pilot_cfg.check_prereg(prereg) == []
    assert PT.check_prereg() == []  # pilot_tvr's prepare guard uses the same parser on the same file


def test_trainer_fingerprint_prefix(tmp_path):
    """The frozen recipe prefix is still the pilot's 0865dd75f1ee, which this tree's trainer files give with the
    pilot's starts.json (this tree's without the Q3SG entry); this tree's own fingerprint is the Q3SG one under the
    invariant (5 .py shas unchanged, starts.json only gains Q3SG; PREREG section 17, row 2026-11-13), so the pilot
    jobs' trainer check refuses here (intended: the pilot is closed)."""
    from vpd.tests import test_tvr_jobs as TJ
    from vpd.train import n2_train_vpd as T

    want = pilot_cfg.FROZEN["recipe"]["trainer_code_sha256_prefix"]
    assert want == "0865dd75f1ee" and PT.PILOT_TRAINER_SHA256.startswith(want)
    assert T.code_fingerprint(TJ.pilot_trainer_root(tmp_path / "pilot"))["code_sha256"] == PT.PILOT_TRAINER_SHA256
    fp = T.code_fingerprint()
    assert fp["code_sha256"] == PT.Q3SG_TRAINER_SHA256 and PT.q3sg_trainer_problems() == []
    assert {k: v for k, v in fp["code_files"].items() if k != PT.STARTS_JSON} == PT.TRAINER_PY_SHA256
    assert PT.trainer_problems() == [f"trainer code_sha256 {PT.Q3SG_TRAINER_PREFIX} != frozen prefix {want}"]


def test_readme_mentions_every_wave2_file():
    text = (paths.VPD_PKG / "README.md").read_text(encoding="utf-8")
    missing = [rel for rel in WAVE2_FILES if Path(rel).name not in text and rel not in text]
    assert not missing, missing


@pytest.mark.parametrize("job", sorted(PT.JOBS))
def test_payload_dry_run_records_every_required_step(tmp_path, job):
    """The step names a payload records (dry run) cover pilot_tvr.required_steps(job): the expect list the job's own
    expect step checks can be met by that payload."""
    spec = PT.JOBS[job]
    args = [a for a in (spec["start"], spec["part"]) if a]
    run = tmp_path / "jobs" / f"{job}-202611120000"
    env = {k: v for k, v in os.environ.items() if k not in TJ._KNOBS}
    env.update(VPD_DRYRUN="1", VPD_TVR_OUT=str(tmp_path / "tvr"), VPD_MERGE_DIR="/dev/shm/vpd_merged",
               VPD_DRY_NTOP="ALL")
    res = subprocess.run(["bash", str(TJ.JOBS_DIR / spec["script"]), *args, str(run)], capture_output=True, text=True,
                         env=env, cwd=str(tmp_path), timeout=300)
    assert res.returncode == 0, res.stderr[-800:]
    names = [json.loads(s)["name"] for s in (run / "steps.jsonl").read_text().splitlines()]
    cfg = TJ.filled_cfg("ALL")
    absent = [r for r in PT.required_steps(job, cfg) if not any(n == r or n.startswith(r + ":") for n in names)]
    assert not absent, (job, absent, names)


def test_recheck_equals_the_data_jobs_floors():
    """recheck_tvr's independent floors == floors.py's floors_eval500.json on the real frozen inputs (G0(17)); skipped
    until the data job has written them."""
    L = PT.Layout()
    fp = L.floors / "floors_eval500.json"
    if not fp.is_file() or not (L.reach / "index.json").is_file():
        pytest.skip("the data job has not written OUT/tvr/floors and OUT/tvr/reach/v1 yet")
    pub = Path(str(pilot_cfg.FROZEN["public"]["sft_train_json"]).replace("OUT/", str(paths.OUT) + "/", 1))
    mine = RC.floors(L.reach, paths.TVR_REPO / "data" / "tasks", pub)
    assert RC.compare_floors(mine, json.loads(fp.read_text())) == []


# ================================================================================================================
# CONTRACT 9 "Integration": the pipeline on a fake TVR tree, every step through the real module (CLI where one exists)
# fake TVR tree -> dev-select -> houses -> reach dump (ThorPool of in-process workers on the fake ai2thor) -> replay
# sft1600 -> demos build -> tiers -> bfs -> floors -> recheck_tvr floors (== floors) -> run_tvr eval (FakeServer) for
# three starts (0W c, 100 c, ALL c,u,blind, Q3B 100s1002 c) -> score_tvr official -> an sft2 canary replay in a second
# process + render_parity -> pilot root -> gate_tvr -> power_sim on the produced per_task.jsonl files.
# Faked (they need a GPU, the trainer or vLLM): the adapters' meta.json / adapter.pt, the merge-parity files, the smoke
# verdict, the calibration's Q3B ALL dev200 result (the real cal-decide writes cal_decision.json from it); FakeServer's
# manifest is the one run_tvr.start_server's (unstarted) VLLMServer records, and the version block is pinned to the frozen versions (the test interpreter's packages are not the served ones; the fake ai2thor has
# no version).  Scale stand-ins (monkeypatched, visible): EVAL500_N 8, PUBLIC_N 9, gates.min_kept 8, the tier files
# of 100 -> demos_4.jsonl (tiers --sizes 4, --zero-w-per-category 1), power.n_sims 40.
# ================================================================================================================
REPO = lvp.TVR_REPO
HAVE_REPO = (REPO / "tvrbench" / "online_rl" / "thor_view_env.py").is_file()
HAVE_FONT = lvp.font_status()["mode"] == "pinned"
RENDER_FILLED = all(pilot_cfg.FROZEN["render"][k] is not None
                    for k in ("lvp_profile", "lp_threads", "replay_envs", "eval_envs"))
GRID_XS = [round(-1.0 + 0.25 * i, 2) for i in range(9)]
WORKER_ENV = {"LP_NUM_THREADS": "2", "VPD_LVP_PROFILE": "envlib", "CUDA_VISIBLE_DEVICES": "0"}
# sft: 2 kept per category + one public sequence whose first move THOR refuses (a blocked cell: move_failed@0)
SFT = [("ithor_FloorPlan11_easy_t00_s00", "FloorPlan11", "ithor", "easy", (0.0, 0.0, 0, 0), ["MoveAhead"]),
       ("ithor_FloorPlan11_easy_t01_s00", "FloorPlan11", "ithor", "easy", (0.5, 0.0, 0, 0), ["MoveAhead", "RotateRight"]),
       ("ithor_FloorPlan12_easy_t02_s00", "FloorPlan12", "ithor", "easy", (0.0, 0.0, 0, 0), ["MoveAhead"]),
       ("ithor_FloorPlan12_hard_t00_s00", "FloorPlan12", "ithor", "hard", (-0.5, -0.5, 0, 0), ["MoveAhead"]),
       ("ithor_FloorPlan12_hard_t01_s00", "FloorPlan12", "ithor", "hard", (0.5, -0.5, 0, 0), ["MoveLeft", "LookDown"]),
       ("procthor_procthor_09990_easy_t00_s00", "procthor_09990", "procthor", "easy", (0.0, 0.0, 0, 0), ["MoveAhead"]),
       ("procthor_procthor_09990_easy_t01_s00", "procthor_09990", "procthor", "easy", (0.5, 0.5, 90, 0),
        ["RotateLeft", "MoveAhead"]),
       ("procthor_procthor_09990_hard_t19_s01", "procthor_09990", "procthor", "hard", (-0.5, 0.0, 0, 0),
        ["MoveAhead", "MoveRight"]),
       ("procthor_procthor_09990_hard_t02_s00", "procthor_09990", "procthor", "hard", (-0.75, -0.75, 0, 0), ["MoveAhead"])]
FAILING = "ithor_FloorPlan12_easy_t02_s00"
# eval: 4 targets two RotateRight away (FakeServer stop_after 2 = the top tier solves them), 2 one away (stop_after 1 =
# the 100 tier), 2 position offsets nothing here reaches; 0W = stop_after 0 (Stop at once) solves none
EVAL = [("ithor_FloorPlan13_easy_t00_s00", "FloorPlan13", "ithor", "easy", (0.0, 0.0, 0, 0), (0.0, 0.0, 90, 0)),
        ("ithor_FloorPlan13_easy_t01_s00", "FloorPlan13", "ithor", "easy", (0.25, 0.0, 90, 0), (0.25, 0.0, 135, 0)),
        ("ithor_FloorPlan13_hard_t00_s00", "FloorPlan13", "ithor", "hard", (0.0, 0.25, 0, 0), (0.0, 0.25, 90, 0)),
        ("ithor_FloorPlan13_hard_t01_s00", "FloorPlan13", "ithor", "hard", (0.0, 0.0, 0, 0), (0.5, 0.25, 0, 0)),
        ("procthor_procthor_00003_easy_t00_s00", "procthor_00003", "procthor", "easy", (0.0, 0.0, 270, 0),
         (0.0, 0.0, 0, 0)),
        ("procthor_procthor_00003_easy_t01_s00", "procthor_00003", "procthor", "easy", (0.25, 0.25, 0, 0),
         (0.25, 0.25, 45, 0)),
        ("procthor_procthor_00003_hard_t00_s00", "procthor_00003", "procthor", "hard", (0.0, 0.0, 180, 0),
         (0.0, 0.0, 270, 0)),
        ("procthor_procthor_00003_hard_t01_s00", "procthor_00003", "procthor", "hard", (-0.25, 0.0, 0, 30),
         (0.25, 0.5, 0, 30))]
RL = [("ithor_FloorPlan14_easy_t00_s00", "FloorPlan14", "ithor", "easy", (0.0, 0.0, 0, 0), (0.0, 0.0, 90, 0)),
      ("ithor_FloorPlan14_hard_t00_s00", "FloorPlan14", "ithor", "hard", (0.25, 0.0, 0, 0), (0.25, 0.0, 45, 0)),
      ("procthor_procthor_00005_easy_t00_s00", "procthor_00005", "procthor", "easy", (0.0, 0.0, 0, 0), (0.0, 0.0, 45, 0)),
      ("procthor_procthor_00005_hard_t00_s00", "procthor_00005", "procthor", "hard", (0.0, 0.0, 0, 0), (0.0, 0.0, 90, 0))]
STOP_AFTER = {"0W": 0, "100": 1, "ALL": 2, "100s1002": 1}
TOP_K = 4  # successes of the ALL tier (c, u, blind) on the 8 eval tasks; 0W 0, 100 2
TIER_FILES_FAKE = {"100": "demos_4.jsonl", "100_s1002": "demos_4.jsonl"}


def _pose_t(p) -> tuple:
    return (p.x, p.z, p.rot, p.hor)


def _sft_tasks():
    """sft task dicts (targets = the free-space pose the public actions imply) and the blocked cell of FAILING."""
    out, blocked = [], None
    for tid, scene, ds, diff, start, acts in SFT:
        t0 = F.fake_task(tid, scene, ds, diff, start, start)
        p = kin_env.pose_of(t0["start"])
        for a in acts:
            p = kin_env.step(p, a, None)[0]
        out.append(F.fake_task(tid, scene, ds, diff, start, _pose_t(p)))
        if tid == FAILING:
            blocked = kin_env.move_target(kin_env.pose_of(t0["start"]), acts[0])
    return out, blocked


def _positions(scene, blocked):
    return F.grid_positions(GRID_XS, GRID_XS, blocked=[blocked] if scene == "FloorPlan12" else ())


def _pool_factory(tv, houses, procthor_dir):
    font = lvp.install_font_rule(tv, allow_missing=True)

    def factory(n, lp, prof, out_dir):
        def mk(wid):
            return thor_worker.Worker(tv, wid=wid, houses_dir=houses, procthor_dir=procthor_dir, font=font,
                                      environ=dict(WORKER_ENV, LP_NUM_THREADS=str(lp), VPD_LVP_PROFILE=prof))
        return thor_worker.ThorPool(n, lp, prof, out_dir=out_dir, houses_dir=houses, local_factory=mk)
    return factory


class ManifestFakeServer(fake_server.FakeServer):
    """FakeServer whose manifest() is the real VLLMServer manifest run_tvr.start_server would record (built, never
    started), plus the fake's own: the gate reads results.server.manifest exactly as on a real run."""

    def __init__(self, vllm_manifest, **kw):
        super().__init__(**kw)
        self._vllm_manifest = vllm_manifest

    def manifest(self):
        return dict(self._vllm_manifest, fake_server=super().manifest())


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _wj(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))
    return p


def _adapters(L, s, cfg, tman, code_sha):
    for tier, init in PT.train_tiers(s, cfg):
        d = L.adapter(s, tier)
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter.pt").write_bytes(f"fake adapter {s} {tier}".encode())
        meta = {"start": s, "seed": 1002 if tier.endswith("_s1002") else 1001, "phase": PT.PHASE[tier],
                "optimizer_steps": 30, "optimizer_steps_planned": 30, "code_sha256": code_sha, "attn": "sdpa",
                "attn_sites": {"config": "sdpa", "model.language_model.config": "sdpa"},
                "max_pixels": int(pilot_cfg.FROZEN["protocol"]["max_pixels"]),
                "data": PT.TIER_FILE[tier], "data_sha256": tman[PT.TIER_FILE[tier]]}
        if init:
            meta["init_adapter"] = {"sha256": _sha(L.adapter(s, init) / "adapter.pt")}
        _wj(d / "meta.json", meta)


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    if not HAVE_REPO:
        pytest.skip("pinned TVRBench checkout not present")
    if not HAVE_FONT:
        pytest.skip("the pinned badge font of the TVR env is not readable (renderer / canary checks need it)")
    if not RENDER_FILLED:
        pytest.skip("pilot_cfg render.* not filled yet")
    from vpd.train import n2_train_vpd as TR

    tmp = tmp_path_factory.mktemp("pipe")
    mp = pytest.MonkeyPatch()
    saved_path = list(sys.path)
    w = types.SimpleNamespace(tmp=tmp)
    try:
        # the fake ai2thor has no version; the test interpreter's packages are not the served ones
        mp.setattr(thor_worker, "ai2thor_version", lambda: pilot_cfg.FROZEN["protocol"]["versions"]["ai2thor"])
        vers = pilot_cfg.FROZEN["protocol"]["versions"]
        mp.setattr(run_tvr, "serve_versions", lambda: {"vllm": vers["vllm"], "xgrammar": vers["xgrammar"],
                                                       "transformers": vers["transformers_serve"], "torch": "2.8.0"})
        mp.setattr(PT, "prereg_state", lambda: {"file": PT.PREREG, "commit": "c0ffee", "committed": True,
                                                "status": "", "clean": True, "ok": True})
        L = PT.Layout(tmp / "OUT")
        w.L = L
        T = L.tvr
        sft, blocked = _sft_tasks()
        ev = [F.fake_task(*e) for e in EVAL]
        rl = [F.fake_task(*e) for e in RL]
        scenes = {t["scene"] for t in sft + ev + rl}
        table = {s: {"positions": _positions(s, blocked),
                     "spawn": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}, "rotation_y": 0.0, "horizon": 0.0}}
                 for s in scenes}
        # a well-formed fake: the kept sequences never touch the blocked cell, the failing one does
        for t, (tid, scene, *_r, acts) in zip(sft, SFT):
            rs = kin_env.ReachSet.from_positions(table[scene]["positions"])
            p, oks = kin_env.pose_of(t["start"]), []
            for a in acts:
                p, ok = kin_env.step(p, a, rs)
                oks.append(ok)
            assert all(oks) is (tid != FAILING), (tid, oks)
        w.fake = F.write_fake_ai2thor(tmp / "fake", table)
        tree = F.write_fake_tvr_tree(T, {"sft": sft, "eval": ev, "rl": rl},
                                     [{"task_id": tid, "actions": acts + ["Stop"]} for tid, *_r, acts in SFT])
        w.tree, td, pub = tree, tree["tasks_dir"], tree["sft_train"]
        w.td, w.pub = td, pub
        assert pub == T / "hf_meta" / "tvr-sft-va" / "sft_train.json"  # where pilot_cfg public.sft_train_json points
        F.purge_tvrbench()
        _te, tv = F.import_real_tvrbench(w.fake)
        factory = _pool_factory(tv, T / "houses", tree["procthor_dir"])
        sets = "sft1600,eval500,dev200"
        # dev-select, houses
        assert TK.main(["dev-select", "--out", str(T / "split" / "dev200"), "--tasks-dir", str(td),
                        "--per-category", "1", "--max-per-scene", "2", "--smoke-per-category", "1"]) == 0
        assert reach.main(["houses", "--task-sets", sets, "--out", str(T / "houses"), "--src",
                           str(tree["procthor_gz"]), "--tasks-dir", str(td), "--tvr-out", str(T)]) == 0
        # reach dump: two passes on a 2-worker pool
        pool = factory(2, 2, "envlib", tmp / "reach_pool")
        pool.start()
        try:
            w.dump = reach.dump(TK.parse_sets(sets), L.reach, pool, None, root=T, tasks_dir=td,
                                timeouts=dict(reach.DEFAULT_TIMEOUTS))
        finally:
            pool.close()
        assert w.dump["complete"] and reach.check(L.reach)["complete"], w.dump
        # replay of the public sequences (render.replay_envs workers), demos, tiers, bfs, floors
        ra = run_tvr.build_parser().parse_args(
            ["replay", "--task-set", "sft1600", "--public", str(pub), "--run-dir", str(L.replay), "--images-dir",
             str(L.demos / "images"), "--tvr-out", str(T), "--tasks-dir", str(td), "--reach", str(L.reach),
             "--houses", str(T / "houses")])
        assert run_tvr.cmd_replay(ra, pool_factory=factory) == 0
        assert demos.main(["build", "--replay", str(L.replay), "--images", str(L.demos / "images"), "--out",
                           str(L.demos), "--reach", str(L.reach), "--public", str(pub), "--tasks-dir", str(td)]) == 0
        assert TK.main(["tiers", "--demos", str(L.demos / "demos_all.jsonl"), "--out", str(L.tiers), "--tasks-dir",
                        str(td), "--zero-w-per-category", "1", "--sizes", "4"]) == 0
        assert bfs.main(["table", "--task-sets", "eval500,dev200,sft1600", "--reach", str(L.reach), "--out",
                         str(L.bfs), "--tasks-dir", str(td), "--tvr-out", str(T)]) == 0
        assert floors.main(["--task-sets", "eval500,dev200", "--reach", str(L.reach), "--out", str(L.floors),
                            "--public", str(pub), "--tasks-dir", str(td), "--tvr-out", str(T)]) == 0
        # the frozen data values the orchestrator would fill (from these files)
        cfg = copy.deepcopy(pilot_cfg.FROZEN)
        for key in PT.DATA_FILES:
            cfg["data"][key] = _sha(L.data_file(key))
        fl = json.loads((L.floors / "floors_eval500.json").read_text())
        man = json.loads((L.demos / "manifest.json").read_text())
        cfg["data"].update(F_eval500=fl["F"]["rate"], F_source=fl["F"]["source"], kept=man["kept"],
                           dropped=man["dropped"])
        cfg["tiers"]["n_top"] = "ALL"
        cfg["gates"]["min_kept"] = len(SFT) - 1
        cfg["power"]["n_sims"] = 40
        w.cfg = cfg
        # the calibration decision n_top is filled from (cal-decide on a stand-in Q3B ALL dev200 result: the cal job
        # needs the trainer and vLLM) against this tree's floors_dev200.json
        _wj(L.cal_eval("ALL") / "results.json", {"valid": True, "n_tasks": 200, "sr": {"k": 100, "n": 200},
                                                 "fake": "the calibration eval needs the trainer and vLLM"})
        w.cal_decision = PT.cal_decide(L.cal, L, cfg)
        assert w.cal_decision["n_top"] == cfg["tiers"]["n_top"]
        tman = json.loads((L.tiers / "manifest.json").read_text())
        mp.setitem(PT.TIER_FILE, "100", TIER_FILES_FAKE["100"])
        mp.setitem(PT.TIER_FILE, "100_s1002", TIER_FILES_FAKE["100_s1002"])
        # pilot evals: three starts, every gated eval, one run_tvr session per merged tier
        mdir = tmp / "merged_fake"
        mdir.mkdir()
        for f in ("config.json", "tokenizer_config.json"):
            (mdir / f).write_text("{}")
        (mdir / "preprocessor_config.json").write_text(json.dumps({"size": {"shortest_edge": 65536}}))
        # the pilot adapters' trainer: this tree's trainer files with the pilot's starts.json (0865dd75f1ee; this
        # tree's own fingerprint is the Q3SG one, PREREG section 17 row 2026-11-13)
        code_sha = TR.code_fingerprint(TJ.pilot_trainer_root(tmp / "pilot_trainer"))["code_sha256"]
        w.served = {}
        for s in pilot_cfg.STARTS:
            _adapters(L, s, cfg, tman, code_sha)
            served = run_tvr.served_name(s)
            w.served[s] = served
            sa = types.SimpleNamespace(fake_server=False, base_url=None, model_dir=str(mdir),
                                       gpu_memory_utilization=float(pilot_cfg.FROZEN["protocol"]["gpu_memory_utilization"]))
            vman = run_tvr.start_server(sa, served, tmp / "vllm.log", int(pilot_cfg.FROZEN["protocol"]["max_model_len"])
                                        ).manifest()
            sessions = [("0W", "c"), ("100", "c"), ("ALL", "c,u,blind")]
            if s == PT.replicate(cfg)["start"]:
                sessions.append(("100s1002", "c"))
            for tier, arms in sessions:
                ad = L.adapter(s, PT.EVAL_ADAPTER[tier])
                ea = run_tvr.build_parser().parse_args(
                    ["eval", "--start", s, "--tier", tier, "--arms", arms, "--task-set", "eval500", "--out-root",
                     str(L.start_dir(s) / "evals"), "--name-prefix", "pilot", "--model-dir", str(mdir),
                     "--adapter-meta", str(ad / "meta.json"), "--tvr-out", str(T), "--tasks-dir", str(td), "--reach",
                     str(L.reach), "--houses", str(T / "houses"), "--bfs", str(L.bfs / "eval500_opt.json")])
                srv = ManifestFakeServer(vman, policy="stop_at", stop_after=STOP_AFTER[tier], model=served)
                assert run_tvr.cmd_eval(ea, pool_factory=factory, server_factory=lambda: srv) == 0, (s, tier)
                _wj(L.parity(s, tier), dict(TJ.fake_parity(ad), fake="merge parity needs the GPU / PY_TRAIN"))
        # G0(7) (PREREG section 17, 2026-11-11): the merge-correctness reports (mp_diag.json of the diagnostic,
        # which needs the GPU / PY_TRAIN: faked) and the registry P/merge_correctness.json pinning their sha256
        TJ.write_merge_correctness(L, cfg)
        # TVRBench's own aggregate on every eval dir (the TVR env python is the test interpreter + the fake ai2thor)
        mp.setenv("PYTHONPATH", os.pathsep.join([str(w.fake), str(paths.WORKTREE)]))
        for d in sorted(L.pilot_root.glob("*/evals/pilot__*")):
            s = d.name.split("__")[1]
            assert score_tvr.main(["official", "--run-dir", str(d), "--served", w.served[s], "--bfs",
                                   str(L.bfs / "eval500_opt.json"), "--py", sys.executable, "--repo", str(REPO)]) == 0, d
        # the render canary: an sft2 replay in a SECOND process (subprocess workers on the fake ai2thor), then the
        # tolerance check against the data replay; the smoke's replay2 and every eval job's canary get this result
        can = tmp / "canary"
        env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE), OMP_NUM_THREADS="1",
                   PYTHONDONTWRITEBYTECODE="1")
        res = subprocess.run([sys.executable, "-B", "-m", "vpd.tvr.run_tvr", "replay", "--task-set", "sft2", "--public",
                              str(pub), "--run-dir", str(can), "--images-dir", str(can / "images"), "--envs", "1",
                              "--lp", "2", "--profile", "envlib", "--tvr-out", str(T), "--tasks-dir", str(td),
                              "--reach", str(L.reach), "--houses", str(T / "houses"), "--repo", str(REPO),
                              "--fake-thor", str(w.fake), "--deadline-s", "600"],
                             capture_output=True, text=True, env=env, cwd=str(paths.WORKTREE), timeout=600)
        assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
        rp = can / "render_parity.json"
        assert render_parity.main(["--replay", str(can), "--ref", str(L.replay), "--ref-images", str(L.demos / "images"),
                                   "--images", str(can / "images"), "--task-set", "sft2", "--out", str(rp)]) == 0
        w.canary = json.loads(rp.read_text())
        sv = _wj(L.smoke_root / "tvr-msmoke-202611120000" / "smoke_verdict.json", {
            "verdict": "PASS", "fake": "the mechanism smoke needs the trainer and vLLM",
            "checks": [{"name": f"{n}_{s}", "ok": True, "detail": None} for s in pilot_cfg.STARTS
                       for n in ("dry", "hfparity", "vllm_parity")]})
        (sv.parent / "replay2").mkdir()
        shutil.copy(rp, sv.parent / "replay2" / "render_parity.json")
        for s in pilot_cfg.STARTS:
            for part in ("top", "low"):
                L.canary(s, part).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(rp, L.canary(s, part))
        w.sv = sv
        yield w
    finally:
        mp.undo()
        F.purge_tvrbench(getattr(w, "fake", None))
        sys.path[:] = saved_path


def _gate(pipe, monkeypatch, **kw):
    monkeypatch.setattr(PT, "EVAL500_N", len(EVAL))
    monkeypatch.setattr(PT, "PUBLIC_N", len(SFT))
    monkeypatch.setattr(PT, "prereg_state", lambda: {"file": PT.PREREG, "commit": "c0ffee", "committed": True,
                                                     "status": "", "clean": True, "ok": True})
    for k, v in TIER_FILES_FAKE.items():
        monkeypatch.setitem(PT.TIER_FILE, k, v)
    return G.evaluate(pipe.L.pilot_root, L=pipe.L, cfg=pipe.cfg, smoke_verdict_path=pipe.sv,
                      recheck_args={"tasks_dir": pipe.td}, **kw)


def test_pipeline_data_products(pipe):
    L = pipe.L
    man = json.loads((L.demos / "manifest.json").read_text())
    assert man["kept"] == len(SFT) - 1 and man["dropped"] == 1 and man["kin_equal_all_kept"] is True
    dropped = json.loads((L.demos / "dropped.json").read_text())
    assert [(d["task_id"], d["reason"].split("@")[0]) for d in dropped] == [(FAILING, "move_failed")]
    assert all(d["reason"].split("@")[0] in PT.DROP_REASONS for d in dropped)
    tman = json.loads((L.tiers / "manifest.json").read_text())
    for f in ("warmup0w20.jsonl", "demos_4.jsonl", "demos_ALL.jsonl", "parity3.jsonl"):
        assert tman[f] == _sha(L.tiers / f), f
    idx = json.loads((L.reach / "index.json").read_text())
    assert {e[0] for e in SFT + EVAL + RL} <= set(idx["tasks"])
    rep = json.loads((L.replay / "results.json").read_text())
    assert rep["valid"] and rep["n_tasks"] == len(SFT) and rep["renderer"]["envs"] == pilot_cfg.FROZEN["render"][
        "replay_envs"]
    # the workers' own ready replies, which G0(4) judges (pilot_tvr.worker_problems), are in run_tvr's replay result
    assert rep["renderer"]["workers_ok"] is True and PT.worker_problems(rep["renderer"]) == []
    assert {w["font_mode"] for w in rep["renderer"]["workers"]} == {"pinned"}
    # recheck_tvr's independent floors == floors.py's file (G0(17) on the integration tree)
    fl = json.loads((L.floors / "floors_eval500.json").read_text())
    assert RC.compare_floors(RC.floors(L.reach, pipe.td, pipe.pub), fl) == []
    assert fl["warnings"] == [] and fl["F"]["source"] in ("stop0", "fixed", "random")
    assert pipe.canary["pass"] is True and pipe.canary["n_frames"] > 0


def test_pipeline_eval_results_in_the_layout_the_gate_reads(pipe):
    """run_tvr's own results.json: code = the served code string, the harness sha at the top level, the server block
    nesting the VLLMServer manifest; the counts the gate recounts from trajectory.json."""
    L = pipe.L
    for s in pilot_cfg.STARTS:
        for tier, arm, k in (("0W", "c", 0), ("100", "c", 2), ("ALL", "c", TOP_K), ("ALL", "u", TOP_K),
                             ("ALL", "blind", TOP_K)):
            r = json.loads((L.eval_dir(s, tier, arm) / "results.json").read_text())
            assert r["valid"] and r["n_tasks"] == len(EVAL) and r["sr"]["k"] == k, (s, tier, arm, r["sr"])
            assert r["code"] == pipe.served[s] and r["code_sha256"] == r["code_fingerprint"]["code_sha256"]
            man = r["server"]["manifest"]
            assert man["cfg"]["enforce_eager"] is True and "--generation-config" in man["cmd"]
            assert man["mm_processor_kwargs"]["max_pixels"] == pilot_cfg.FROZEN["protocol"]["max_pixels"]
            assert r["server"]["max_model_len"] == pilot_cfg.FROZEN["protocol"]["max_model_len"]
            assert RC.counts(L.eval_dir(s, tier, arm))["k"] == k
            assert json.loads((L.eval_dir(s, tier, arm) / "official_check.json").read_text())["equal"] is True


def test_pipeline_gate_and_power_sim(pipe, monkeypatch):
    from vpd.common import power_sim

    L = pipe.L
    rep = _gate(pipe, monkeypatch)
    bad = [(i["name"], i["detail"]) for i in rep["G0"]["items"] if i["ok"] is not True]
    assert not bad, bad
    assert (rep["G1"]["status"], rep["G2"]["status"], rep["G3"]["status"]) == ("pass", "pass", "pass"), \
        (rep["G1"], rep["floors"])
    assert rep["A4"]["status"] == "pending" and rep["decision"] == "incomplete"  # no POWER.json yet
    assert rep["recheck"]["counts_equal"] is True and rep["recheck"]["floors_equal"] is True
    assert rep["A3"]["per_start"]["Q3B"]["b"] == 0 and rep["A3"]["per_start"]["Q3B"]["p"] == 1.0
    rep = _gate(pipe, monkeypatch, power=True)
    pf = L.pilot_root / "POWER.json"
    pw = json.loads(pf.read_text())
    assert isinstance(pw["pass"], bool) and pw["registered"]["n_sims_used"] == 40
    assert rep["A4"]["status"] == ("pass" if pw["pass"] else "fail")
    assert rep["decision"] == ("ruler" if pw["pass"] else "power_insufficient")
    # the POWER.json power_sim wrote is current and made at the registered settings (the fields gate_tvr reads)
    assert G.power_unregistered(pw, pipe.cfg) == [] and pw["registered"]["test"] == "seed_aware_endpoint_t"
    assert pw["sigma"]["replicate_missing"] is False and pw["persistence"]["replicate_missing"] is False
    again = _gate(pipe, monkeypatch)
    assert again["A4"]["status"] == rep["A4"]["status"] and again["A4"]["note"] is None
    assert again["A4"]["power"] == G.a4_summary(pw) and again["A4"]["power"]["lambda_used"] == \
        pw["persistence"]["lambda_used"] and again["A4"]["power"]["type1"] == pw["verdicts"]["main"]["type1"]
    direct = power_sim.run_tvr(L.pilot_root, cfg=pipe.cfg, out=pipe.tmp / "POWER2.json")
    assert direct["pass"] == pw["pass"] and direct["inputs"]["files_sha256"] == pw["inputs"]["files_sha256"]
    md = G.render_md(rep)
    assert "不算起点之间的差" in md and f"| Q3B | ALL | c | 是 | {TOP_K}/{len(EVAL)}" in md
    # not at the registered settings (power_sim's own writer, current inputs and cfg): A4 pending, never read
    try:
        for kw, needle in (({"n_sims": 20}, "registered.n_sims_used 20 != power.n_sims 40"),
                           ({"allow_no_replicate": True}, "registered.allow_no_replicate True")):
            power_sim.run_tvr(L.pilot_root, cfg=pipe.cfg, out=pf, **kw)
            bad = _gate(pipe, monkeypatch)
            assert bad["A4"]["status"] == "pending" and bad["decision"] == "incomplete", kw
            assert needle in bad["A4"]["note"] and bad["A4"]["power"] is None, bad["A4"]
    finally:
        _gate(pipe, monkeypatch, power=True)  # the registered POWER.json again
    assert _gate(pipe, monkeypatch)["A4"]["status"] == rep["A4"]["status"]


def test_pipeline_gate_catches_a_doctored_real_result(pipe, monkeypatch, tmp_path):
    """On run_tvr's real files: a results.json whose k disagrees with its trajectories fails G0(17); one whose
    served-weights sha is not the adapter on disk fails G0(8)."""
    p = pipe.L.eval_dir("E2", "100", "c") / "results.json"
    orig = p.read_text()
    try:
        r = json.loads(orig)
        r["sr"]["k"] += 1
        r["adapter"]["adapter_sha256"] = "0" * 64
        p.write_text(json.dumps(r))
        rep = _gate(pipe, monkeypatch)
        failed = sorted(i["name"] for i in rep["G0"]["items"] if i["ok"] is False)
        assert failed == ["independent_recompute", "served_adapter"] and rep["decision"] == "engineering"
    finally:
        p.write_text(orig)

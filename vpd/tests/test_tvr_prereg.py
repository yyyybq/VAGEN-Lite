"""Tests of package prereg: vpd/tvr/pilot_cfg.py and vpd/PREREG_TVR_PILOT.md (CONTRACT_wave2 sections 3, 5.4, 8.4, 9).

Hermetic: reads only the tree's own files, plus (skipped when absent) the pinned TVRBench checkout, the public
sft_train.json and the finished probe summary, read-only.  Login node, 1 core:
    cd TREE && OMP_NUM_THREADS=1 PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_tvr_prereg.py
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import pytest

from vpd import paths
from vpd.tvr import pilot_cfg as C

PREREG = paths.WORKTREE / "vpd" / "PREREG_TVR_PILOT.md"
CFG_SRC = paths.WORKTREE / "vpd" / "tvr" / "pilot_cfg.py"
F = C.FROZEN


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _text() -> str:
    return PREREG.read_text(encoding="utf-8")


def _prose() -> str:
    """The prereg without the frozen JSON block."""
    t = _text()
    a, rest = t.split(C.BLOCK_BEGIN, 1)
    return a + rest.split(C.BLOCK_END, 1)[1]


def _none_keys(cfg=None):
    return sorted(k for k, v in C._leaves(F if cfg is None else cfg) if v is None)


def _filled(n_top="ALL"):
    cfg = copy.deepcopy(F)
    for k in _none_keys(cfg):
        if k in C.OPTIONAL_NONE:
            continue
        cur = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = "filled"
    cfg["tiers"]["n_top"] = n_top
    return cfg


# --------------------------------------------------------------------------------------------- block == FROZEN
def test_block_equals_frozen():
    assert C.check_prereg(PREREG) == []
    block = C.prereg_block(PREREG)
    assert json.dumps(block, sort_keys=True) == json.dumps(F, sort_keys=True)
    assert block == json.loads(json.dumps(F))


def test_block_is_the_as_json_dump():
    t = _text()
    inner = t.split(C.BLOCK_BEGIN, 1)[1].split(C.BLOCK_END, 1)[0]
    assert "```json\n" + C.as_json() + "\n```" in inner


def test_check_prereg_reports_differences(tmp_path):
    t = _text()
    p = tmp_path / "p.md"
    p.write_text(t.replace('"lp_threads": 2', '"lp_threads": 3'), encoding="utf-8")
    diff = C.check_prereg(p)
    assert len(diff) == 1 and diff[0].startswith("render.lp_threads")
    p.write_text(t + "\n" + C.BLOCK_BEGIN + "\n", encoding="utf-8")
    assert "exactly one TVR_FROZEN block" in C.check_prereg(p)[0]
    p.write_text(t.replace("```json", "```", 1), encoding="utf-8")
    assert "no ```json fence" in C.check_prereg(p)[0]
    assert C.check_prereg(tmp_path / "absent.md")
    # type-only differences and one-sided keys count (a plain != would miss them)
    p.write_text(t.replace('"rank": 16', '"rank": 16.0'), encoding="utf-8")
    diff = C.check_prereg(p)
    assert len(diff) == 1 and diff[0].startswith("recipe.rank")
    p.write_text(t.replace('"holm": true', '"holm": 1'), encoding="utf-8")
    assert [d.split(":")[0] for d in C.check_prereg(p)] == ["power.holm"]
    p.write_text(t.replace(' "seed": 1001,\n', ' "seed": 1001,\n "extra_key": null,\n'), encoding="utf-8")
    assert [d.split(":")[0] for d in C.check_prereg(p)] == ["extra_key"]
    p.write_text(t.replace('  "hf_revision": null,\n', ''), encoding="utf-8")
    assert [d.split(":")[0] for d in C.check_prereg(p)] == ["public.hf_revision"]
    # against a given cfg (pilot_tvr's test path)
    cfg = copy.deepcopy(F)
    cfg["render"]["lp_threads"] = 3
    assert [d.split(":")[0] for d in C.check_prereg(PREREG, cfg)] == ["render.lp_threads"]


# --------------------------------------------------------------------------------------------- None keys documented
def test_every_none_is_documented_as_tbd():
    prose = _prose()
    lines = prose.splitlines()
    for k in _none_keys():
        hits = [ln for ln in lines if f"`{k}`" in ln]
        assert hits, f"{k} not mentioned in the prereg prose"
        if k in C.OPTIONAL_NONE:
            assert any(("未核实" in ln) or ("按设计" in ln) for ln in hits), k
        else:
            assert any("待填" in ln for ln in hits), k
    # after the probe fill only data.* and tiers.n_top (+ the two optional ones) may still be None; later fills
    # (data job, calibration) shrink this set
    allowed = {f"data.{x}" for x in F["data"]} | {"tiers.n_top", "public.hf_revision", "protocol.chat_template_kwargs"}
    assert set(_none_keys()) <= allowed
    # before the calibration, the provisional eval_envs is marked as such
    assert any("`render.eval_envs`" in ln for ln in lines)
    if F["tiers"]["n_top"] is None:
        assert any("`render.eval_envs`" in ln and "暂定" in ln for ln in lines)


def test_prereg_mentions_every_top_level_key():
    prose = _prose()
    for k in F:
        assert f"`{k}`" in prose, k


# --------------------------------------------------------------------------------------------- REQUIRED / missing()
def test_required_keys_exist():
    leaves = dict(C._leaves(F))
    for job, req in C.REQUIRED.items():
        if req == "ALL":
            continue
        for k in req:
            assert k in leaves, (job, k)
            C.get(k)
    for k in C.OPTIONAL_NONE:
        assert k in leaves


def _pre_data():
    """The state right after the probe fill: render.* filled, data.* and tiers.n_top None."""
    cfg = copy.deepcopy(F)
    for k in cfg["data"]:
        cfg["data"][k] = None
    cfg["tiers"]["n_top"] = None
    return cfg


def test_missing_after_the_probe_fill():
    # render.* is filled from the probe: the data job can be prepared (true for every later state too)
    assert C.missing("tvr-data") == []
    cfg = _pre_data()
    assert C.missing("tvr-data", cfg) == []
    assert C.missing("tvr-cal", cfg) == sorted(k for k in C.REQUIRED["tvr-cal"] if k.startswith("data."))
    assert C.missing("tvr-cal400", cfg) == sorted(k for k in C.REQUIRED["tvr-cal400"] if k.startswith("data."))
    # the calibration jobs need EVERY data.* key (one fill commit after the data job; K and the drop list are
    # registered before any training, G10(a)): a single missing data key blocks them
    every = sorted(f"data.{k}" for k in F["data"])
    for job in ("tvr-cal", "tvr-cal400"):
        assert C.missing(job, cfg) == every, job
        assert {k for k in C.REQUIRED[job] if k.startswith("data.")} == set(every), job
    filled = _filled()
    for key in ("kept", "dropped", "F_eval500", "F_source", "houses_sha256", "replay_manifest_sha256"):
        one = copy.deepcopy(filled)
        one["data"][key] = None
        assert C.missing("tvr-cal", one) == [f"data.{key}"] == C.missing("tvr-cal400", one), key
    zero = copy.deepcopy(filled)
    zero["data"].update(F_eval500=0.0, dropped=0)     # a filled 0 is not missing
    assert C.missing("tvr-cal", zero) == []
    for job in ("tvr-msmoke", "tvr-train", "tvr-eval", "tvr-train-q3s", "tvr-eval-e2-top", "tvr-eval-q3b-low"):
        m = C.missing(job, cfg)
        assert m == sorted({f"data.{k}" for k in F["data"]} | {"tiers.n_top"}), job
    # the current FROZEN: whatever is still missing is only data.* / tiers.n_top
    for job in ("tvr-cal", "tvr-msmoke", "tvr-eval-e2-top"):
        assert all(k.startswith("data.") or k == "tiers.n_top" for k in C.missing(job)), job


def test_missing_all_filled_is_empty_for_every_job():
    for n_top in ("ALL", "400"):
        cfg = _filled(n_top)
        jobs = list(C.REQUIRED) + ["tvr-train-q3b", "tvr-train-q3s", "tvr-train-e2"]
        jobs += [f"tvr-eval-{s}-{h}" for s in ("q3b", "q3s", "e2") for h in ("top", "low")]
        for job in jobs:
            assert C.missing(job, cfg) == [], (n_top, job)
        assert C.n_top(cfg) == n_top and C.pilot_tiers(cfg) == ("0W", "100", n_top)
    bad = _filled("300")
    assert C.missing("tvr-eval-e2-top", bad) == ["tiers.n_top"]
    with pytest.raises(ValueError):
        C.n_top(bad)
    unfrozen = copy.deepcopy(F)        # independent of the shipped fill state (n_top is filled at the freeze)
    unfrozen["tiers"]["n_top"] = None
    with pytest.raises(ValueError):
        C.n_top(unfrozen)
    if F["tiers"]["n_top"] is None:
        with pytest.raises(ValueError):
            C.n_top()                  # the shipped cfg, before the calibration
    else:
        assert C.n_top() in ("ALL", "400")
    with pytest.raises(KeyError):
        C.missing("tvr-bogus")
    # a filled key set back to None is reported again
    cfg = _filled()
    cfg["render"]["eval_envs"] = None
    assert C.missing("tvr-cal", cfg) == ["render.eval_envs"]
    assert C.missing("tvr-eval-q3s-low", cfg) == ["render.eval_envs"]


# --------------------------------------------------------------------------------------------- constants
def test_constants_consistent():
    assert C.REGEX == "Action: (" + "|".join(C.ACTIONS) + ")"
    assert C.ACTIONS[-1] == "Stop" and C.NON_STOP == C.ACTIONS[:-1] and len(C.NON_STOP) == 8
    assert F["protocol"]["actions"] == list(C.ACTIONS) and F["protocol"]["regex"] == C.REGEX
    assert set(C.CATEGORY_OF.values()) == set(C.CATEGORIES)
    assert F["categories"] == {"SR-easy": ["ithor", "easy"], "SR-hard": ["ithor", "hard"],
                               "LR-easy": ["procthor", "easy"], "LR-hard": ["procthor", "hard"]}
    assert C.category_of({"dataset": "procthor", "difficulty": "hard"}) == "LR-hard"
    assert C.cap_of({"dataset": "ithor"}) == 30 and C.cap_of({"dataset": "procthor"}) == 40
    assert F["starts"] == ["Q3B", "Q3S", "E2"] and F["seed"] == 1001
    # 0W: 20 conversations, 12 epochs, 30 optimizer steps (accum 8)
    z = F["zero_w"]
    assert z["n"] == 4 * z["per_category"] == 20
    assert C.epochs_for(20) == z["expected_epochs"] == 12
    assert math.ceil(20 * C.epochs_for(20) / F["recipe"]["accum"]) == z["expected_optimizer_steps"] == 30
    assert [C.epochs_for(n) for n in (100, 400, 1580)] == [3, 3, 3]
    assert F["dev"]["n"] == 4 * F["dev"]["per_category"] == 200
    assert F["gates"]["min_kept"] == math.ceil(0.95 * 1600)
    assert F["protocol"]["max_tokens"] == {"c": 16, "u": 64, "blind": 16}
    assert F["protocol"]["image_tokens"] == F["protocol"]["image_pad_tokens"] + 2
    g = F["protocol"]["image_grid"]
    assert g[0] * g[1] * g[2] // 4 == F["protocol"]["image_pad_tokens"]
    assert F["protocol"]["max_images"] >= 2 * max(F["protocol"]["max_steps"].values())
    assert F["protocol"]["max_model_len_fallback"] == 128000
    # confirmatory design (O5): 6 starts, 2 tiers, seeds 1-3, weights 0.5 / 0.5
    d = F["power"]["design"]
    assert d["starts"] == ["Q3B", "Q3S", "E2", "Q3RD", "QRL-s1", "QRL-s2"]
    assert d["tiers"] == ["100", "TOP"] and d["weights"] == [0.5, 0.5]
    assert d["seeds"] == {"100": [1, 2, 3], "TOP": [1, 2, 3]}
    assert F["power"]["seed_sd"]["a_priori"] == 0.03 and F["power"]["outcome"]["min_effect"] == 0.05
    assert F["budget"]["round_cap_gpuh"] == 200.0 and F["budget"]["round_baseline_gpuh"] == 8.8404
    json.loads(C.as_json())


def test_power_registration_g13():
    """DECISIONS G13 (wave 2b, before the freeze): the endpoint-level test, the tier_corr cells of A4, the balanced
    fallback and the task-persistence term; the unchanged A4 numbers."""
    pw = F["power"]
    assert pw["test"] == "seed_aware_endpoint_t"
    for s in ("sigma2_E_hat", "seeds paired across tiers", "df = n_starts * (n_seeds - 1) (12; fallback 6)",
              "2 for Q3S-E2", "1.5 for Q3RD-mean(QRL)"):
        assert s in pw["test_def"], s
    assert "sqrt(rho) z_Ss + sqrt(1 - rho) z_SsT" in pw["seed_model"] and "sqrt(lambda)" in pw["seed_model"]
    assert pw["tier_corr"] == {"type1": [0.0, 1.0], "power": 0.5, "reported": [0.0, 1.0]}
    assert pw["fallback"]["seeds"] == {"100": [1, 2], "TOP": [1, 2]}
    assert pw["fallback"]["seeds"]["100"] == pw["fallback"]["seeds"]["TOP"]          # paired: the test needs it
    assert set(pw["fallback"]["seeds"]["TOP"]) < set(pw["design"]["seeds"]["TOP"])
    pe = pw["persistence"]
    assert set(pe) == {"source", "estimator", "range", "rule", "sensitivity", "check_reps"}
    assert pe["range"] == [0.0, 1.0] and pe["sensitivity"] == [0.0, 0.8] and pe["check_reps"] == 400
    # wave-2b refix: the reference is invariant to the start mean SRs (one-factor covariances, jackknifed); the
    # wave-2 H = m / (m - 1) mean p (1 - p) (half the between-start discordance) is gone
    assert "only through covariances over tasks" in pe["source"] and "no between-start difference" in pe["source"]
    assert "never a difference or dispersion" not in pe["source"]
    for s in ("V = the delete-one-scene jackknife", "C(l, y_a) C(l, y_b) / C(y_a, y_b)", "B = L (1 - L) - V",
              "R = 2 B + m2 / 2", "lambda_hat = (R - D) / (R - kappa)"):
        assert s in pe["estimator"], s
    assert "m / (m - 1)" not in pe["estimator"] and "R <= kappa" in pe["rule"]
    # the simulation backend is registered (the python and numpy streams differ)
    assert pw["backend"] == "numpy"
    # unchanged by G13
    assert (pw["type1_max"], pw["target_power"], pw["primary_delta"], pw["n_sims"], pw["alpha"]) == \
        (0.06, 0.80, 0.08, 2000, 0.05)
    assert pw["power_counts"] == "Holm-rejected AND treated better, other contrast at effect 0"
    # no start name in the registered strings POWER.json copies outside the whitelisted test_def
    for key in ("seed_model", "fallback", "persistence", "tier_corr"):
        assert not any(s in json.dumps(pw[key]) for s in F["starts"]), key


def test_prereg_a4_text_matches_the_registration():
    t = _text()
    prose = _prose()
    assert "我的建议" not in t
    s0 = prose.split("## 0.", 1)[1].split("### 0.1", 1)[0]
    a4_row = next(ln for ln in s0.splitlines() if ln.startswith("| **A4**"))
    for w in ("ρ = 0.5", "ρ = 0 和 ρ = 1", "0.80", "0.06", "λ_used", "终点层面"):
        assert w in a4_row, w
    s12 = prose.split("## 12.", 1)[1].split("## 13.", 1)[0]
    for w in ("seed_aware_endpoint_t", "seed_var_endpoint", "df = 起点数 × (n_s − 1) = 12", "备用设计 6",
              "`power.tier_corr`", "`power.persistence`", "persistence_hat", "persistence_check", "λ_used",
              "fallback_seeds_1_2", "power_insufficient", "sensitivity_superseded_test", "不插补"):
        assert w in s12, w
    s13 = prose.split("## 13.", 1)[1].split("## 14.", 1)[0]
    a4 = s13.split("### A4", 1)[1].split("### 决策表", 1)[0]
    for w in ("ρ = 0.5", "ρ = 0 和 ρ = 1", "λ_used", "σ_used", "≤ 0.06", "≥ 0.80"):
        assert w in a4, w
    assert "G0、G2" in s13.split("### G0", 1)[0]          # the replicate is checked by G0 and G2, feeds only A4
    s15 = prose.split("## 15.", 1)[1].split("## 16.", 1)[0]
    for w in ("终点层面", "df 12", "df 6", "fallback_pass", "两档都只用种子 1–2"):
        assert w in s15, w
    assert "冻结前由编排定" not in s15


def test_prereg_registers_the_wave2b_gate_conditions():
    """The conditions gate_tvr / pilot_tvr judge since wave 2b are registered text (review prereg wave 2b): the
    section-0 guard (n_top == cal_decision.json, data, futility), the O9 re-decision, A4 never reading an
    allow_no_replicate POWER.json, the per-worker renderer checks (G0(4), smoke), the code rule (G0(5)), the *_used
    settings (G0(10)), n_top (G0(13)) and the A4 pending conditions."""
    prose = _prose()
    s0 = prose.split("## 0.", 1)[0]
    for w in ("CAL/cal_decision.json", "with_400", "n_top_problems", "data_problems", "futility_override.json",
              "pilot_tvr guard", "prereg_committed"):
        assert w in s0, w
    s7 = prose.split("## 7.", 1)[1].split("## 8.", 1)[0]
    for w in ("fallback_used.json", ".prev-<UTC>", "重新判"):
        assert w in s7, w
    s12 = prose.split("## 12.", 1)[1].split("## 13.", 1)[0]
    i5 = s12.split("\n5. ", 1)[1].split("\n6. ", 1)[0]
    assert "从不读这样的 POWER.json" in i5 and "pending" in i5 and "incomplete" in i5
    assert "第 17 节的偏差" not in i5                      # --allow-no-replicate is not a usable A4 path
    for w in ("`power.backend`", "registered.backend", "vpd.power_sim.tvr/2", "shift_m2"):
        assert w in s12, w
    s13 = prose.split("## 13.", 1)[1].split("## 14.", 1)[0]
    g0 = s13.split("### G0", 1)[1].split("### 冒烟", 1)[0]
    item = {int(ln.split(".", 1)[0]): ln for ln in g0.splitlines() if re.match(r"\d+\. ", ln)}
    for w in ("workers_ok", "system_prompt_ok", "tvrbench_files_ok", "pinned", "worker_problems"):
        assert w in item[4], w
    for w in ("0W_s1002", "code_fingerprint.code_sha256", "只管评测"):
        assert w in item[5], w
    for w in ("max_tokens_used", "regex_used", "blind_concurrency_used", "request_fields", "max_model_len_used",
              "fallback_used.json"):
        assert w in item[10], w
    assert "CAL/cal_decision.json" in item[13] and "with_400" in item[13]
    assert "CAL/cal_decision.json" in s13.split("### G0", 1)[0]          # the read list
    smoke = s13.split("### 冒烟", 1)[1].split("### 无望停止", 1)[0]
    assert "workers_ok" in smoke and "worker_problems" in smoke
    a4 = s13.split("### A4", 1)[1].split("### 决策表", 1)[0]
    for w in ("power_stale", "power_unregistered", "n_sims_used", "allow_no_replicate", "replicate_missing",
              "registered.backend", "pilot_cfg_json_sha256", "incomplete"):
        assert w in a4, w


def test_render_filled_from_the_probe():
    r = F["render"]
    assert r["lvp_profile"] == "envlib"
    assert (r["replay_envs"], r["lp_threads"]) == (5, 2)
    assert r["replay_envs"] * r["lp_threads"] <= 11
    assert isinstance(r["eval_envs"], int) and r["eval_envs"] * r["lp_threads"] <= 11
    if F["tiers"]["n_top"] is None:
        assert r["eval_envs"] == r["replay_envs"]      # provisional until the calibration
    assert r["recommended_measured"] == [6, 2]
    assert r["probe_run"] == "OUT/jobs/tvr-probe-202611101501"
    assert r["compare_verdict"] == "same_rendering" and r["deterministic"] is False
    assert re.fullmatch(r"[0-9a-f]{64}", r["probe_summary_sha256"])
    assert r["consistency_rule"]["psnr_min_db"] == 40.0 and r["consistency_rule"]["ssim_min"] == 0.99


def test_ssim_definition_sha_matches_probe_lvp():
    from vpd.tvr import probe_lvp

    assert hashlib.sha256(probe_lvp.SSIM_DEFINITION.encode()).hexdigest() == F["render"]["ssim_definition_sha256"]
    assert probe_lvp.PSNR_MIN_DB == F["render"]["consistency_rule"]["psnr_min_db"]
    assert probe_lvp.SSIM_MIN == F["render"]["consistency_rule"]["ssim_min"]


PROBE_SUMMARY = paths.OUT / "jobs" / "tvr-probe-202611101501" / "out" / "summary.json"


@pytest.mark.skipif(not PROBE_SUMMARY.is_file(), reason="probe summary not present")
def test_render_values_equal_the_probe_summary():
    r = F["render"]
    assert _sha(PROBE_SUMMARY) == r["probe_summary_sha256"]
    s = json.loads(PROBE_SUMMARY.read_text())
    assert s["lvp_profile"]["profile"] == r["lvp_profile"]
    assert (s["recommended"]["E"], s["recommended"]["L"]) == (r["replay_envs"], r["lp_threads"])
    rm = s["recommended_measured"]
    assert [rm["E"], rm["L"]] == r["recommended_measured"]
    assert s["compare"]["verdict"] == r["compare_verdict"]
    assert s["determinism"]["deterministic"] is r["deterministic"]
    assert s["font"]["parent"]["sha256"] == r["badge_font_sha256"]


def test_trainer_fingerprint_prefix():
    from vpd.train import n2_train_vpd as T

    assert T.code_fingerprint()["code_sha256"].startswith(F["recipe"]["trainer_code_sha256_prefix"])


REPO = paths.TVR_REPO


@pytest.mark.skipif(not (REPO / "scripts" / "eval_va.py").is_file(), reason="pinned TVRBench checkout not present")
def test_tvrbench_file_hashes():
    for rel, sha in F["tvrbench"]["files_sha256"].items():
        assert _sha(REPO / rel) == sha, rel


PUBLIC = paths.OUT / "tvr" / "hf_meta" / "tvr-sft-va" / "sft_train.json"


@pytest.mark.skipif(not PUBLIC.is_file(), reason="public sft_train.json not present")
def test_public_sha():
    assert F["public"]["sft_train_json"] == "OUT/" + str(PUBLIC.relative_to(paths.OUT)).replace("\\", "/")
    assert _sha(PUBLIC) == F["public"]["sft_train_json_sha256"]


# --------------------------------------------------------------------------------------------- py3.10
def test_pilot_cfg_is_py310_stdlib():
    src = CFG_SRC.read_text(encoding="utf-8")
    tree = ast.parse(src, feature_version=(3, 10))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "copy", "json", "math", "re"}, mods


TVR_PY = paths.ENVS / "tvr" / "bin" / "python"


@pytest.mark.skipif(not TVR_PY.is_file(), reason="TVR env python not present")
def test_pilot_cfg_imports_under_tvr_py():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from vpd.tvr import pilot_cfg as C; "
            "import json; print(json.dumps({'v': sys.version_info[:2], 'sha': __import__('hashlib')"
            ".sha256(C.as_json().encode()).hexdigest()}))")
    env = {"OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"}
    out = subprocess.run([str(TVR_PY), "-B", "-I", "-c", code, str(paths.WORKTREE)], capture_output=True,
                         text=True, timeout=60, env=env)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    assert got["v"] == [3, 10]
    assert got["sha"] == hashlib.sha256(C.as_json().encode()).hexdigest()


# --------------------------------------------------------------------------------------------- prereg structure
def test_prereg_header_and_sections():
    t = _text()
    assert ("状态：**草稿 — 数据作业之后填 data.*，校准之后填 n_top 和 eval_envs，填完的那次提交即冻结，"
            "必须在冒烟（msmoke）之前。**") in t
    heads = [ln for ln in t.splitlines() if ln.startswith("## ")]
    nums = [re.match(r"## (\d+)\. ", h).group(1) for h in heads if re.match(r"## (\d+)\. ", h)]
    assert nums == [str(i) for i in range(18)], nums
    assert any(h.startswith("## 附 A.") for h in heads)
    assert "### 0.1 冻结常量块" in t
    for word in ("推断", "未核实", "闸门不看起点之间的差", "git status --porcelain -- vpd/PREREG_TVR_PILOT.md",
                 "pilot_cfg.check_prereg"):
        assert word in t, word


def test_appendix_lists_every_adopted_o_item():
    t = _text()
    app = t.split("## 附 A.", 1)[1]
    for i in list(range(1, 19)) + list(range(20, 26)):
        assert re.search(rf"\*\*O{i}\*\*", app), f"O{i}"
    o25 = app.split("**O25**", 1)[1].split("\n- **", 1)[0]
    for w in ("seed_aware_endpoint_t", "ρ = 0", "ρ = 0.5", "种子 1–2", "λ", "G13"):
        assert w in o25, w
    assert "O19" in app and "不在本文件里" in app
    assert not re.search(r"\*\*O19\*\*", app)


def test_deviation_log_seed_entries():
    t = _text()
    log = t.split("## 17. 偏差日志", 1)[1].split("## 附 A.", 1)[0]
    rows = [ln for ln in log.splitlines() if ln.startswith("| ") and not ln.startswith("| 日期") and "---" not in ln]
    assert len(rows) >= 6      # the six seed entries; later deviations are appended
    seeds = [lambda r: "lavapipe" in r and "用户（2026-11-10" in r and "不用 NVIDIA" in r,
             lambda r: "逐位可复现" in r and "容差" in r,
             lambda r: "可达集合" in r and "O1" in r,
             lambda r: "合并 = LoRA" in r and "O2" in r,
             lambda r: "终点层面" in r and "O25" in r and "DECISIONS G13" in r and "编排" in r,
             lambda r: "128000" in r and "O9" in r]
    for i, ok in enumerate(seeds):
        assert any(ok(r) for r in rows), f"seed entry {i}"


def test_confirmatory_and_cost_sections():
    t = _text()
    s15 = t.split("## 15.", 1)[1].split("## 16.", 1)[0]
    for word in ("Q3RD", "QRL-s1", "QRL-s2", "种子 1、2、3", "Holm", "min_effect = 0.05", "支持", "不支持", "无定论",
                 "100 卡时", "种子 1–2", "est_high 约 85", "0W 种子 1"):
        assert word in s15, word
    s16 = t.split("## 16.", 1)[1].split("## 17.", 1)[0]
    for word in ("200", "8.8404", "5.5 h", "HEADROOM_S", "SOFT_RESERVE_S", "MIN_OUTER_S", "1.3 ×", "17,500",
                 "round_budget"):
        assert word in s16, word


def test_prereg_numbers_match_frozen():
    t = _text()
    for v in (F["render"]["probe_summary_sha256"], F["render"]["ssim_definition_sha256"],
              F["render"]["badge_font_sha256"], F["public"]["sft_train_json_sha256"], F["tvrbench"]["commit"],
              F["render"]["build"], C.REGEX):
        assert t.count(v) >= 2, v          # once in the block, at least once in the prose
    for rel, sha in F["tvrbench"]["files_sha256"].items():
        assert f"| `{rel}` | `{sha}` |" in t, rel

"""Fixed locations of the VPD line: worktree, output tree (plan section 0), interpreters, external inputs.

Pure constants (stdlib only), importable from every interpreter (starVLA py3.10, eval-external py3.12).
Nothing here creates directories; callers mkdir what they write.
"""

from __future__ import annotations

from pathlib import Path

USER_ROOT = Path("/mnt/umm/users/yinbaiqiao")

# --- code ---------------------------------------------------------------------------------------------------
WORKTREE = USER_ROOT / "VAGEN-Lite-vpd"  # branch viewpoint-downstream; new code only under vpd/
VPD_PKG = WORKTREE / "vpd"
DOWNSTREAM = WORKTREE / "downstream"  # read-only imports (sco_job.py, n/n2_train.py, train/lora.py, x/q0_ask.py)
EVAL_EXTERNAL = WORKTREE / "eval_external"  # read-only imports (adapters, closed_loop)
STARTS_JSON = VPD_PKG / "starts.json"

# --- interpreters and tools ---------------------------------------------------------------------------------
PY_EVAL = USER_ROOT / ".conda" / "envs" / "eval-external" / "bin" / "python"  # WalkerBench env, CPU tools
PY_Q3 = USER_ROOT / "eval_external_assets" / "envs" / "eval-external-q3" / "bin" / "python"  # vLLM 0.11, tf 4.57.6
PY_TRAIN = USER_ROOT / ".conda" / "envs" / "starVLA" / "bin" / "python"  # torch 2.6, tf 4.57.0, flash_attn
SCO = USER_ROOT / ".sco" / "bin" / "sco"

# --- external inputs (read-only) ----------------------------------------------------------------------------
ASSETS_ROOT = USER_ROOT / "eval_external_assets"
MODEL_VIEWS = ASSETS_ROOT / "model_views"  # eval_external.adapters.models.view_dir(code) == MODEL_VIEWS / code
REPO_WB = ASSETS_ROOT / "repos" / "WalkerBench"  # checkout b197f51, never written
WB_COMMIT = "b197f51"
PANOS = ASSETS_ROOT / "data" / "walkerbench_pano" / "panoramas"
DSV_ROOT = ASSETS_ROOT / "downstream_search_20261108"  # survey material (report v2, prototypes)
REPORT_V2 = DSV_ROOT / "REPORT_downstream_search_v2.md"
SPLIT_CITY_SRC = DSV_ROOT / "login_tmp" / "dsv_wb" / "split_city.json"

# --- outputs (plan section 0) -------------------------------------------------------------------------------
OUT = ASSETS_ROOT / "vpd"
PLAN_DIR = OUT / "plan"
SCRATCH = OUT / "scratch"  # per-WP scratch: SCRATCH / <label>; never /tmp on the login node
LEDGER = OUT / "ledger.jsonl"
JOBS = OUT / "jobs"  # RUN dirs of sco_job.prepare
ENVS = OUT / "envs"

WB_OUT = OUT / "walkerbench"
WB_SPLIT = WB_OUT / "split"  # split_s2.json, expert_splice.json, s2test/, s2val/, s2train/
WB_S2TEST_TASKS = WB_SPLIT / "s2test" / "tasks"
WB_S2VAL_TASKS = WB_SPLIT / "s2val" / "tasks"
WB_S2TRAIN_TASKS = WB_SPLIT / "s2train" / "tasks"
WB_BASELINES = WB_OUT / "baselines"  # baselines_s2test.json, hard_subsets.json
WB_DEMOS = WB_OUT / "demos"  # s2train_<W>x<H>/{demos_all.jsonl, images/, manifest.json}
WB_WARMUP = WB_OUT / "warmup"  # warmup20.jsonl, images/, manifest.json
WB_TIERS = WB_OUT / "tiers"  # order_seed<s>.json, seed<s>/demos_<N>.jsonl
WB_RUNS = WB_OUT / "runs"  # smoke/, pilot/<START>/{adapters/, evals/}

TVR_OUT = OUT / "tvr"
NVGL = TVR_OUT / "nvgl_595.71.05"  # lib/, glvnd/, vulkan/, egl/
AI2THOR_HOME = TVR_OUT / "ai2thor_home"
TVR_REPO = TVR_OUT / "repos" / "TVRBench"
TVR_ENV = ENVS / "tvr"

E3VS_OUT = OUT / "e3vs"
E3VS_REPO = E3VS_OUT / "repo" / "E3VS-BENCH"
E3VS_COMMIT = "bb7e603ccdcf2dd77fbeedd15660a09310a8a881"
E3VS_DATA = E3VS_OUT / "data" / "scannetpp_v2_mcmc_3dgs"
E3VS_OCC = E3VS_OUT / "occ"
E3VS_BFS = E3VS_OUT / "bfs"
E3VS_SFT = E3VS_OUT / "sft"
E3VS_PARITY = E3VS_OUT / "parity"
E3VS_RUNS = E3VS_OUT / "runs"

# --- shared settings (DECISIONS B9, B16) ---------------------------------------------------------------------
# Image pixel cap given to vLLM (--mm-processor-kwargs) and to every training/parity processor.  640x400 and
# 1280x800 (= 1,024,000 px) pass unchanged; Qwen3-VL's own longest_edge default is 16,777,216.
MAX_PIXELS = 1_024_000
JOB_THREADS = 2  # OMP & co. for every in-job process except vLLM
VLLM_THREADS = 4  # OMP for vLLM processes
LOGIN_THREADS = 1  # every login-node check
JOB_UID = 20325  # as-root jobs chown their outputs back to this uid


def scratch(label: str) -> Path:
    """The scratch dir of one WP label (not created)."""
    if not label or "/" in label or label.startswith("."):
        raise ValueError(f"bad scratch label {label!r}")
    return SCRATCH / label

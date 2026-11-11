"""TVR pilot frozen constants (DECISIONS G3, G10; vpd/PREREG_TVR_PILOT.md section 0.1, block TVR_FROZEN).

Single source of every TVR pilot module.  vpd/tests/test_tvr_prereg.py asserts FROZEN == the JSON block between
<!-- TVR_FROZEN_BEGIN --> and <!-- TVR_FROZEN_END --> of vpd/PREREG_TVR_PILOT.md (parsed JSON, tuples as lists).
None = a value only a measurement gives; the orchestrator fills it (probe -> render.*, data job -> data.*,
calibration -> tiers.n_top) in BOTH files in one commit.  missing(job) lists what a job still lacks; pilot_tvr
refuses to prepare a job whose REQUIRED keys are missing.  Stdlib only; py3.10 and py3.12.
"""
from __future__ import annotations

import copy
import json
import math

ACTIONS = ("MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "RotateRight", "RotateLeft", "LookUp", "LookDown",
           "Stop")
NON_STOP = ACTIONS[:-1]
REGEX = "Action: (MoveAhead|MoveBack|MoveLeft|MoveRight|RotateRight|RotateLeft|LookUp|LookDown|Stop)"
CATEGORIES = ("SR-easy", "SR-hard", "LR-easy", "LR-hard")
CATEGORY_OF = {("ithor", "easy"): "SR-easy", ("ithor", "hard"): "SR-hard",
               ("procthor", "easy"): "LR-easy", ("procthor", "hard"): "LR-hard"}
STARTS = ("Q3B", "Q3S", "E2")
SEED = 1001
ARMS = ("c", "u", "blind")          # c = primary (regex, THOR); u = secondary (unconstrained, THOR); blind = no image (regex, kin_env)

FROZEN = {
    "schema": "vpd.tvr.pilot_cfg/1",
    "tvrbench": {
        "commit": "ad296bb7ff880b8a0758187edc3ddc97eff92791",
        "files_sha256": {
            "tvrbench/env/thor_env.py": "d4ac9688aa7a1c3dca9b42140cc2fc804ca163d7950c7d8ae94ad5c5659349c4",
            "tvrbench/online_rl/thor_view_env.py": "7e3c46c285a0f6344b811a0a5603c206753b3e7ace26a3c66500e29ba6c0072c",
            "tvrbench/evaluation/metrics.py": "2d91bbc87634bf70d49cd8dba7a1e989a28c49c24bdde80b84f9f1c492171fab",
            "scripts/eval_va.py": "d27faf8ed8a4ff798eadf5dbe8a95ff17402c04fd26684c1d2e1e545f6b82fb9",
            "data/tasks/eval.json": "aff32430551739a3c6c1336e7e7f75a174b73f337e273f5a5a7dee736b7d12c8",
            "data/tasks/sft.json": "09f4a538974f335505b8721f57afb8d8d35d3cc7a184d8f81ca372e6b29640ac",
            "data/tasks/rl.json": "4406ec551d9b1d3a389abf2a239220f4a2a75016f82ee8f0ae22b38a1ed42d15",
            "data/scene_splits.json": "a36d33acd32a88a984faef7b8f3dd1979972e27822ff72083c593124ede09f44",
            "data/procthor-10k/train.jsonl.gz": "ee3c4aa14b4d8f0895fecfb5fdaca59395427ca1018b2f9aeeedbc61e5824587",
        },
    },
    "public": {"sft_train_json": "OUT/tvr/hf_meta/tvr-sft-va/sft_train.json",
               "sft_train_json_sha256": "cbf6f36d524955f3aaa4f9f6e91f876b50d174c09f8f4a6a69f5c3a85f00e13d",
               "hf_revision": None},
    "render": {"renderer": "lavapipe", "width": 640, "height": 360, "quality": "Ultra", "anti_aliasing": 4,
               "fov": 60, "grid_size": 0.25, "rotate_step": 45, "platform": "CloudRendering",
               "build": "thor-CloudRendering-f0825767cd50d69f666c7f282e54abfe58f1e917",
               "badge_font_sha256": "b184b89e3c1075f22f6b71575b6fc20d4972b3cfd3b23322ca6fd596dcaef167",
               # filled from the finished probe (pt-s9d5txr3, OUT/jobs/tvr-probe-202611101501/out/summary.json):
               "lvp_profile": "envlib",      # probe summary.lvp_profile.profile (first try, no retry)
               "lp_threads": 2,              # probe summary.recommended.L
               "replay_envs": 5,             # probe summary.recommended.E; replay and reach dumps
               "eval_envs": 5,               # E for image arms; = replay_envs PROVISIONALLY (before the cal job),
                                             # confirmed or changed after it (Q3B-only), before the freeze
               "recommendation_field": "summary.recommended",   # E*L <= 11 rule; recommended_measured recorded only
               "recommended_measured": [6, 2],  # [E, L] of summary.recommended_measured (record only)
               "probe_run": "OUT/jobs/tvr-probe-202611101501",
               "probe_summary_sha256": "430e34ff2ec6e695c622b5bae135707c3a27bbfb921ec74aab005fcb711a5e5f",
               "ssim_definition_sha256": "9f5517b54436155b3aaec2dfc66b41e3496a37f1ef2e59276739454a1f6fc5eb",
               # = sha256(probe_lvp.SSIM_DEFINITION.encode()), probe_lvp.py at 4487309d (sha256 615668d7b119...)
               "compare_verdict": "same_rendering",   # probe summary.compare.verdict (27/27 frames)
               "deterministic": False,       # probe summary.determinism.deterministic (measured)
               "consistency_rule": {"psnr_min_db": 40.0, "ssim_min": 0.99, "ssim": "probe_lvp.SSIM_DEFINITION",
                                    "mask": "none (same pinned badge font on both sides)",
                                    "exact": ["states", "valid", "user_texts", "action_success"],
                                    "frame_cache": "forbidden"}},
    "protocol": {"actions": list(ACTIONS), "regex": REGEX, "max_steps": {"ithor": 30, "procthor": 40},
                 "history": "full", "temperature": 0.0, "top_p": 1.0, "top_k": -1,
                 "max_tokens": {"c": 16, "u": 64, "blind": 16}, "max_images": 96, "max_model_len": 32768,
                 "max_model_len_fallback": 128000,
                 "gpu_memory_utilization": 0.85, "image_tokens": 222, "image_pad_tokens": 220,
                 "image_grid": [1, 22, 40], "max_pixels": 1024000, "secondary_parser": "eval_va.parse_action",
                 "chat_template_kwargs": None, "reach": "frozen_per_task_start_pose", "png": "PIL save PNG defaults",
                 "blind_concurrency": 16,
                 "server": {"enforce_eager": True, "seed": 0, "generation_config": "vllm",
                            "structured_outputs_backend": "xgrammar", "prefix_caching": "vllm default (on)"},
                 "versions": {"vllm": "0.11.0", "xgrammar": "0.1.25", "transformers_serve": "4.57.6",
                              "transformers_train": "4.57.0", "ai2thor": "5.0.0"}},
    "starts": list(STARTS), "seed": SEED,
    "salts": {"zero_w": "tvr-0w-v1|", "tier": "tvr-tier-v1|", "dev": "tvr-dev-v1|", "rand_floor": "tvr-floor-rand-v1|"},
    "categories": {c: list(k) for k, c in CATEGORY_OF.items()},
    "zero_w": {"per_category": 5, "n": 20, "expected_epochs": 12, "expected_optimizer_steps": 30},
    "dev": {"per_category": 50, "max_per_scene_category": 2, "n": 200, "source": "rl.json", "smoke_per_category": 2},
    "tiers": {"calibration": ["0W", "100", "ALL"], "calibration_extra": "400", "all_dev_sr_max": 0.70,
              "n_top": None,                 # "ALL" or "400" (calibration rule, G10(e))
              "replicate": {"start": "Q3B", "tier": "100", "seed": 1002}},
    "recipe": {"rank": 16, "alpha": 32.0, "dropout": 0.05, "lr": 1e-4, "accum": 8, "warmup": 0.03, "epochs": "auto",
               "epochs_rule": "max(3, ceil(240/N))", "history_k": 0, "max_seq_tokens": 40000, "attn": "sdpa",
               "num_workers": 3, "threads": 2, "trainer_code_sha256_prefix": "0865dd75f1ee"},
    "floors": {"fixed_k_max": 39, "random_seeds": [0, 1, 2],
               "median": "statistics.median_low of the non-Stop action count per category over the 1600 public sequences"},
    "gates": {"z": 1.96, "saturation": 0.80, "mcnemar_alpha": 0.05, "min_kept": 1520,
              "merge_parity": {"n_demos": 20, "max_mean_abs_dce": 0.005, "min_reply_agreement": 0.99,
                               "tiers": "every merged pilot eval (0W, 100, TOP, 100s1002)"},
              "kin_eval": {"valid_list_mismatch_max": 0, "task_mismatch_max_frac": 0.01},
              "futility": "Q3B ALL (or 400) dev200 c Wilson LB <= floors_dev200 F -> stop and ask the user"},
    "bootstrap": {"resamples": 10000, "seed": 0, "cluster": "scene", "use": "descriptive only"},
    "power": {"deltas": [0.05, 0.08, 0.10], "primary_delta": 0.08, "target_power": 0.80, "alpha": 0.05,
              "holm": True, "n_sims": 2000, "seed": 0, "test": "seed_aware_endpoint_t",   # DECISIONS G13
              # the python and numpy backends of power_sim draw different random streams: seed 0 and n_sims pin
              # the result only with the backend (POWER.json registered.backend records the one used)
              "backend": "numpy",
              "test_def": "D = mean_i d_i (per-task endpoint differences; e_Si = sum_T w_T mean_s y_SsTi, seeds "
                          "paired across tiers); V = CR1 scene-cluster var of mean(d) + k_contrast * sigma2_E_hat / "
                          "n_seeds; sigma2_E_hat = max(0, mean over starts of [var_s(E_s) - mean_i var_s(e_si) / n]) "
                          "(ddof 1; e_si = sum_T w_T y_sTi, E_s = mean_i e_si); t reference, df = n_starts * "
                          "(n_seeds - 1) (12; fallback 6); two-sided; k = sum of squared contrast coefficients "
                          "(2 for Q3S-E2, 1.5 for Q3RD-mean(QRL))",
              "effect_model": "realized SR shift delta per tier, mapped to a pre-clip shift by calibrate_shift",
              "seed_model": "realized-scale shift r_SsT = sigma * (sqrt(rho) z_Ss + sqrt(1 - rho) z_SsT), z iid "
                            "N(0, 1), per (start, seed, tier); rho = tier_corr (same seed id across tiers); mapped "
                            "by calibrate_shift; y = 1[W < clip(p_i(T) + c, 0, 1)], W = V_Si (one uniform per start "
                            "and task, shared by its seeds and both tiers) with probability sqrt(lambda), else a "
                            "fresh uniform; lambda = persistence lambda_used",
              "tier_corr": {"type1": [0.0, 1.0], "power": 0.5, "reported": [0.0, 1.0]},
              "fallback": {"seeds": {"100": [1, 2], "TOP": [1, 2]},
                           "rule": "if est_high > 100 GPU-h: fallback iff fallback_pass, else stop and ask"},
              "persistence": {"source": "the seed replicate (tiers.replicate: seed 1001 vs its replicate seed, same "
                                        "start, replicate tier, c arm, same tasks); the other pilot starts' c-arm "
                                        "outcomes at that tier (seed 1001) only through covariances over tasks, "
                                        "for the spread of the task pattern at the replicate start's scale "
                                        "(one-factor model: every start its own mean and scale; lambda_hat is "
                                        "invariant to start mean SRs; no between-start difference is computed)",
                              "estimator": "D = mean_i (y1_i - y2_i)^2; kappa = E[|r1 - r2| | dbar, S] under "
                                           "r1 - r2 ~ N(0, 2 sigma_used^2), dbar ~ N(r1 - r2, S) (dbar, S, "
                                           "sigma_used as seed_sd): b = 2 sigma_used^2 / (2 sigma_used^2 + S), "
                                           "a = b dbar, v = b S, kappa = sqrt(v) sqrt(2 / pi) exp(-a^2 / (2 v)) + "
                                           "a (1 - 2 Phi(-a / sqrt(v))); m2 = E[(r1 - r2)^2 | dbar, S] = a^2 + v; "
                                           "l_i = (y1_i + y2_i) / 2, L = mean_i l_i; y_a, y_b = the other two "
                                           "pilot starts' rows; C = covariance over tasks (ddof 1); V = the "
                                           "delete-one-scene jackknife (G V - (G - 1) mean_g V_-g) of "
                                           "C(l, y_a) C(l, y_b) / C(y_a, y_b); B = L (1 - L) - V; R = 2 B + m2 / 2; "
                                           "lambda_hat = (R - D) / (R - kappa)",
                              "range": [0.0, 1.0],
                              "rule": "lambda_used = clip(lambda_hat, range); lambda_hat undefined (C(y_a, y_b) <= 0 "
                                      "in the full or a delete-one-scene sample, or R <= kappa) or replicate missing "
                                      "-> range[1]",
                              "sensitivity": [0.0, 0.8], "check_reps": 400},
              "power_counts": "Holm-rejected AND treated better, other contrast at effect 0",
              "type1_max": 0.06,
              "seed_sd": {"mode": "replicate_max_a_priori", "a_priori": 0.03, "sensitivity": [0.05],
                          "estimator": "e_i = y1_i - y2_i; d = mean e; S = sum (e_i - d)^2 / (n (n - 1)); "
                                       "sigma_hat = sqrt(max(0, (d^2 - S) / 2)); sigma_used = max(0.03, sigma_hat)"},
              "design": {"tiers": ["100", "TOP"], "weights": [0.5, 0.5], "seeds": {"100": [1, 2, 3], "TOP": [1, 2, 3]},
                         "starts": ["Q3B", "Q3S", "E2", "Q3RD", "QRL-s1", "QRL-s2"],
                         "contrasts": [["Q3S", ["E2"]], ["Q3RD", ["QRL-s1", "QRL-s2"]]]},
              "outcome": {"min_effect": 0.05,
                          "support": "Holm-rejected with the treated start better",
                          "no_support": "Holm-rejected with the treated start worse, OR the 95% t-CI upper bound < min_effect",
                          "inconclusive": "otherwise",
                          "overall": "支持 iff >= 1 contrast support and none Holm-rejected the other way; "
                                     "不支持 iff both no_support; else 无定论"},
              "p_source": "per task and tier: mean over the three pilot starts' primary-arm outcomes"},
    "budget": {"round_baseline_gpuh": 8.8404, "round_cap_gpuh": 200.0, "job_est_high_max_h": 5.5},
    "data": {"dev_selection_sha256": None, "houses_sha256": None, "reach_index_sha256": None,
             "replay_manifest_sha256": None, "demos_all_sha256": None, "kept": None, "dropped": None,
             "tiers_manifest_sha256": None, "bfs_eval500_sha256": None, "bfs_dev200_sha256": None,
             "floors_eval500_sha256": None, "F_eval500": None, "F_source": None},
}

# keys (dotted) that must be non-None before pilot_tvr prepares a job.  The calibration jobs need EVERY data.* key:
# the orchestrator fills them all in one commit after the data job, and G10(a) registers K and the drop list before
# any training (the calibration trains Q3B).
_DATA_KEYS = [f"data.{k}" for k in FROZEN["data"]]
REQUIRED = {
    "tvr-data": ["render.lvp_profile", "render.lp_threads", "render.replay_envs", "render.probe_run"],
    "tvr-cal": ["render.lvp_profile", "render.lp_threads", "render.replay_envs", "render.eval_envs"] + _DATA_KEYS,
    "tvr-cal400": ["render.lvp_profile", "render.lp_threads", "render.replay_envs", "render.eval_envs"] + _DATA_KEYS,
    "tvr-msmoke": "ALL",            # every None above must be filled (prereg frozen)
    "tvr-train": "ALL",
    "tvr-eval": "ALL",              # tvr-eval-<s>-top and tvr-eval-<s>-low map here
}
# may stay None even for "ALL": hf_revision may be unreadable; chat_template_kwargs is None BY DESIGN (we send none)
OPTIONAL_NONE = ("public.hf_revision", "protocol.chat_template_kwargs")


def get(dotted: str, cfg=None):
    cur = FROZEN if cfg is None else cfg
    for part in dotted.split("."):
        cur = cur[part]
    return cur


def _leaves(d, prefix=""):
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            yield from _leaves(v, key + ".")
        else:
            yield key, v


def _base_job(job: str) -> str:
    if job in REQUIRED:
        return job
    for base in ("tvr-train", "tvr-eval"):
        if job.startswith(base + "-"):
            return base
    raise KeyError(f"unknown job {job!r}")


def missing(job: str, cfg=None) -> list:
    """Dotted keys still None that `job` (a key of REQUIRED; tvr-train-<s> / tvr-eval-<s>-top|low map to tvr-train /
    tvr-eval) needs."""
    base = _base_job(job)
    req = REQUIRED[base]
    leaves = dict(_leaves(FROZEN if cfg is None else cfg))
    keys = [k for k in leaves if k not in OPTIONAL_NONE] if req == "ALL" else req
    out = [k for k in keys if leaves.get(k) is None]
    if req == "ALL" and get("tiers.n_top", cfg) not in ("ALL", "400"):
        out.append("tiers.n_top")
    return sorted(set(out))


def n_top(cfg=None) -> str:
    v = get("tiers.n_top", cfg)
    if v not in ("ALL", "400"):
        raise ValueError(f"tiers.n_top is {v!r}; the calibration has not been frozen")
    return v


def pilot_tiers(cfg=None) -> tuple:
    return ("0W", "100", n_top(cfg))


def category_of(task: dict) -> str:
    return CATEGORY_OF[(task["dataset"], task["difficulty"])]


def cap_of(task: dict) -> int:
    return FROZEN["protocol"]["max_steps"][task["dataset"]]


def epochs_for(n: int) -> int:
    return max(3, int(math.ceil(240 / n)))


def as_json(cfg=None) -> str:
    return json.dumps(copy.deepcopy(FROZEN if cfg is None else cfg), indent=1, sort_keys=True)


BLOCK_BEGIN = "<!-- TVR_FROZEN_BEGIN -->"
BLOCK_END = "<!-- TVR_FROZEN_END -->"


def prereg_block(path) -> dict:
    """The parsed JSON of the ```json fence between BLOCK_BEGIN and BLOCK_END of the prereg (exactly one of each;
    ValueError otherwise).  Single parser used by test_tvr_prereg AND pilot_tvr's prepare guard."""
    import re
    text = open(path, encoding="utf-8").read()
    if text.count(BLOCK_BEGIN) != 1 or text.count(BLOCK_END) != 1:
        raise ValueError(f"{path}: expected exactly one TVR_FROZEN block")
    inner = text.split(BLOCK_BEGIN, 1)[1].split(BLOCK_END, 1)[0]
    m = re.search(r"```json\s*\n(.*?)\n```", inner, re.S)
    if m is None:
        raise ValueError(f"{path}: no ```json fence inside the TVR_FROZEN block")
    return json.loads(m.group(1))


def check_prereg(path, cfg=None) -> list:
    """[] iff the prereg block equals FROZEN (or the given cfg; tuples compared as lists); else human-readable
    differences.  A leaf on one side only (also a null against an absent key) and a type-only difference (32 vs
    32.0, false vs 0) count: leaves are compared by their JSON text, and the whole dumps must be equal."""
    want = json.loads(json.dumps(FROZEN if cfg is None else cfg))
    try:
        got = prereg_block(path)
    except (OSError, ValueError) as e:
        return [str(e)]
    a, b = dict(_leaves(want)), dict(_leaves(got))
    out = []
    for k in sorted(set(a) | set(b)):
        if k not in b:
            out.append(f"{k}: absent from the prereg block (pilot_cfg {a[k]!r})")
        elif k not in a:
            out.append(f"{k}: prereg {b[k]!r} is not a pilot_cfg key")
        elif json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True):
            out.append(f"{k}: prereg {b[k]!r} != pilot_cfg {a[k]!r}")
    if not out and json.dumps(want, sort_keys=True) != json.dumps(got, sort_keys=True):
        out.append("prereg block != pilot_cfg FROZEN (structure, e.g. an empty object)")
    return out

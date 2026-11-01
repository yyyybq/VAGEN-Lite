"""The train-chain plan of one matrix group (configs/matrix_v2/<group>.yaml) with its variables filled in.

    python -m data_gen.sft_mixture.matrix_plan --group D [--rl-package v10_p2] [--commit SHA] [--chain-patch] [--out plan.json]

``--chain-patch``: for a training chain with docs/transition_qa/patches 0001 and 0002 applied, the group's
``needs_chain_support`` goes into the plan (sft_lf_dirs, start_model, rl.output_format, rl.max_response_length,
rl.guards.invalid_until, eval.rl_prompt_format, diagnostics with the probe sets of vars.yaml).

The group files write the RL standard as a variable (user, 2026-10-19): ``{rl_package}`` (configs/matrix_v2/vars.yaml,
or --rl-package) and ``{pkg}``, its short name without "v10_" (the RL folder names).  The plan is refused unless
``<rl_package>_S0`` and ``<rl_package>_D`` both exist in data_gen/interiorgs_v10/configs/rl_standards.json and differ
only in the reward (round 12, decision 2).  Group C0 trains nothing: it prints the round-12 chain whose results it
uses (``reuse``) instead of a plan.  v10_p1 / v10_p2 name RL optimisation packages, not the data pools P1 / P2.

    python -m data_gen.sft_mixture.matrix_plan --version 3 --group D --commit SHA [--seed S] [--dense ...] \
        [--out '{tr}/plan.json']

``--version 3`` (Qwen3-VL-8B, configs/matrix_v3; the user's decision B of 2026-11-01): one plan per group and seed for
qwen3-main's chain with patches 0001 / 0002 (the integration branch matrix-v3).  The first seed's chain trains the
SFT, the second starts from its pick; each trains one S0 / dense pair.  The dense side stays a placeholder (the chain
stops at rl_config) until its baseline is given: --dense registered (the 2026-10-27 V4) or a JSON object, or
vars.yaml chain.dense_baseline.  Groups C and D first; B' and E later.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "matrix_v2"
STANDARDS = ROOT / "data_gen" / "interiorgs_v10" / "configs" / "rl_standards.json"
VAR = re.compile(r"\{(rl_package|pkg)\}")


def variables(rl_package: Optional[str] = None) -> Dict[str, str]:
    name = rl_package or yaml.safe_load((MATRIX / "vars.yaml").read_text(encoding="utf-8"))["rl_package"]
    m = re.fullmatch(r"v10_(p\d+)", str(name))
    if not m:
        raise SystemExit(f"rl_package must look like v10_p<N>, not {name!r}")
    standards = json.loads(STANDARDS.read_text(encoding="utf-8"))
    s0, d = standards.get(f"{name}_S0"), standards.get(f"{name}_D")
    if s0 is None or d is None:
        raise SystemExit(f"{name}_S0 / {name}_D missing from {STANDARDS}")
    differ = {k for k in set(s0) | set(d) if s0.get(k) != d.get(k)}
    if differ != {"reward_preset"}:
        raise SystemExit(f"{name}_S0 and {name}_D must differ only in reward_preset, not in {sorted(differ)}")
    return {"rl_package": name, "pkg": m.group(1)}


def fill(obj: Any, values: Dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {fill(k, values): fill(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fill(x, values) for x in obj]
    if isinstance(obj, str):
        return VAR.sub(lambda m: values[m.group(1)], obj)
    return obj


def load_group(group: str, rl_package: Optional[str] = None) -> Dict[str, Any]:
    cfg = yaml.safe_load((MATRIX / f"{group}.yaml").read_text(encoding="utf-8"))
    return fill(cfg, variables(rl_package))


def plan(group: str, rl_package: Optional[str] = None, commit: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_group(group, rl_package)
    if "train_chain_plan" not in cfg:
        raise SystemExit(f"group {group} trains nothing; its results come from {json.dumps(reuse(group, rl_package))}")
    p = cfg["train_chain_plan"]
    if commit:
        p["commit"] = commit
    left = VAR.findall(json.dumps(p, ensure_ascii=False))
    if left:
        raise SystemExit(f"unfilled variables {left}")
    return p


# needs_chain_support key -> where the patched chain reads it in the plan (docs/transition_qa/patches 0001 / 0002)
CHAIN_FIELDS: Dict[str, Sequence[str]] = {
    "sft_lf_dirs": ("sft_lf_dirs",),
    "start_model": ("start_model",),
    "rl_output_format": ("rl", "output_format"),
    "rl_max_response_length": ("rl", "max_response_length"),
    "guards_invalid_until": ("rl", "guards", "invalid_until"),
    "eval_rl_prompt_format": ("eval", "rl_prompt_format"),
    "diagnostics_after_eval": ("diagnostics",),
}


def chain_fields(group: str, rl_package: Optional[str] = None) -> Dict[str, Any]:
    """The group's needs_chain_support as plan fields of the patched chain: {plan path: value}."""
    cfg = load_group(group, rl_package)
    probes = yaml.safe_load((MATRIX / "vars.yaml").read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    for key, value in (cfg.get("needs_chain_support") or {}).items():
        if key not in CHAIN_FIELDS:
            raise SystemExit(f"{group}: needs_chain_support {key!r} has no plan field in the patches")
        if key == "start_model" and not (isinstance(value, dict) and value.get("from_chain") or str(value).startswith("/")):
            raise SystemExit(f"{group}: start_model must be a path or {{from_chain: TR}}, not {value!r}")
        if key == "diagnostics_after_eval":
            if not value:
                continue
            value = {"failure_breakdown": True, "probes": {"P1": probes.get("p1_probe"), "P0": probes.get("p0_probe")}}
        out[".".join(CHAIN_FIELDS[key])] = value
    return out


def plan_with_chain_patch(group: str, rl_package: Optional[str] = None, commit: Optional[str] = None) -> Dict[str, Any]:
    """The plan for a chain with patches 0001 and 0002 applied: the plan plus the group's needs_chain_support."""
    p = plan(group, rl_package, commit)
    for dotted, value in chain_fields(group, rl_package).items():
        node = p
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return p


def reuse(group: str, rl_package: Optional[str] = None) -> Dict[str, Any]:
    """The existing chain whose results a no-training group uses, for the chosen RL standard."""
    cfg = load_group(group, rl_package)
    name = variables(rl_package)["rl_package"]
    if name not in (cfg.get("reuse") or {}):
        raise SystemExit(f"group {group} has no reused chain for {name}")
    return cfg["reuse"][name]


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True)
    ap.add_argument("--rl-package", default=None, help="override configs/matrix_v2/vars.yaml (e.g. v10_p2)")
    ap.add_argument("--commit", default=None, help="the main commit the chain checks out (plan field commit)")
    ap.add_argument("--out", default=None, help="write the plan JSON here (default: print it)")
    ap.add_argument("--chain-patch", action="store_true",
                    help="for a chain with docs/transition_qa/patches 0001 and 0002 applied: add needs_chain_support")
    ap.add_argument("--version", type=int, choices=(2, 3), default=2,
                    help="3: the Qwen3 matrix (configs/matrix_v3), one plan per seed; --out may name {seed} / {tr}")
    ap.add_argument("--seed", type=int, default=None, help="v3: one seed (default: every matrix seed)")
    ap.add_argument("--dense", default=None,
                    help='v3: the dense side: "registered" (the 2026-10-27 V4) or a JSON object with a standard; '
                         'default vars.yaml chain.dense_baseline, else the placeholder')
    a = ap.parse_args(argv)
    if a.version == 3:
        v3_main(a)
        return
    cfg = yaml.safe_load((MATRIX / f"{a.group}.yaml").read_text(encoding="utf-8"))
    if "train_chain_plan" not in cfg:
        print(json.dumps({"group": a.group, "reuse": reuse(a.group, a.rl_package)}, indent=1, ensure_ascii=False))
        return
    build = plan_with_chain_patch if a.chain_patch else plan
    text = json.dumps(build(a.group, a.rl_package, a.commit), indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


# ---------------------------------------------------------------------------------------------------------------------
# Matrix v3 (Qwen3-VL-8B, configs/matrix_v3; the user's decision B of 2026-11-01, on the integration branch matrix-v3)
# ---------------------------------------------------------------------------------------------------------------------
MATRIX_V3 = ROOT / "configs" / "matrix_v3"
V3_GROUPS = ("C", "D")  # the first batch (the user, 2026-11-01); B' and E later
# the dense side's rl until the user relays the main line's answer on its baseline: not a standard, so rl-config
# refuses it and the chain stops at its rl_config step (SFT, its evaluation and the pick are done by then); the plan
# is then written again with the baseline (--dense) and the chain run again, which redoes rl_config and starts both
DENSE_PENDING = "PENDING_dense_baseline"


def v3_vars() -> Dict[str, Any]:
    return yaml.safe_load((MATRIX_V3 / "vars.yaml").read_text(encoding="utf-8"))


def v3_dense(dense: Any, v: Dict[str, Any]) -> Dict[str, Any]:
    """The dense run's rl options: ``dense`` None -> vars.yaml chain.dense_baseline, and while that is null the
    placeholder; "registered" -> the 2026-10-27 registration's V4 (vars.yaml rl.dense on the D_cond standard); a dict
    (or its JSON) -> as given, with a known "standard"."""
    if dense is None:
        dense = v["chain"].get("dense_baseline")
    if dense is None:
        return {"standard": DENSE_PENDING}
    if dense == "registered":
        spec = {k: x for k, x in v["rl"]["dense"].items() if k != "variant"}
        return {"standard": v["rl"]["pair"]["d"], **spec}
    if isinstance(dense, str):
        dense = json.loads(dense)
    standards = json.loads(STANDARDS.read_text(encoding="utf-8"))
    if not isinstance(dense, dict) or dense.get("standard") not in standards:
        raise SystemExit(f"the dense side needs a known standard of {STANDARDS.name}, not {dense!r}")
    return dict(dense)


def dense_pending(plan: Dict[str, Any]) -> List[str]:
    """The runs of a v3 plan whose dense side is still the placeholder."""
    return [n for n, spec in (plan.get("rl_runs") or {}).items()
            if ((spec or {}).get("rl") or {}).get("standard") == DENSE_PENDING]


def v3_names(group: str, seed: int, v: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Chain folder, SFT group, RL name and runs of a group's chain for one seed (all m3_* / m3*: new names)."""
    v = v or v3_vars()
    cfg = yaml.safe_load((MATRIX_V3 / f"{group}.yaml").read_text(encoding="utf-8"))
    seeds = [int(s) for s in v["rl"]["seeds"]]
    if int(seed) not in seeds:
        raise SystemExit(f"seed {seed} is not one of the matrix seeds {seeds}")
    k = seeds.index(int(seed)) + 1
    g = group.lower()
    ch = cfg["chain"]
    return {"k": k, "first": k == 1, "tr": f"{ch['tr']}_s{seed}", "tr_first": f"{ch['tr']}_s{seeds[0]}",
            "sft_group": next(iter(ch["sft_groups"])), "rl_name": f"{ch['rl_name']}_s{k}",
            "s0": f"m3{g}s{k}s0", "d": f"m3{g}s{k}d", "cfg": cfg}


def plan_v3(group: str, seed: int, commit: str, dense: Any = None) -> Dict[str, Any]:
    """The train-chain plan of one v3 group for one seed (patches 0001 / 0002 applied, qwen3-main's chain).  The first
    seed's chain trains the group's SFT (plan "sft_lf_dirs", the copy in the matrix worktree), evaluates every epoch
    and picks on val; the second seed's starts from that pick ("start_model": {"from_chain": first seed's folder}).
    Each chain trains one S0 / dense pair from the same start with the same tasks, updates and stop lines."""
    if group not in V3_GROUPS:
        raise SystemExit(f"v3 plans exist for {V3_GROUPS} (the first batch); {group} comes later")
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit or "")):
        raise SystemExit("--commit must be the full commit the chain checks out")
    v = v3_vars()
    n = v3_names(group, seed, v)
    cfg, ch, rl = n["cfg"], v["chain"], v["rl"]
    standards = json.loads(STANDARDS.read_text(encoding="utf-8"))
    if rl["pair"]["s0"] not in standards:
        raise SystemExit(f"{rl['pair']['s0']} missing from {STANDARDS}")
    p: Dict[str, Any] = {
        "tr": n["tr"], "commit": commit, "model": v["base_qwen3"], "model_type": v["model_type"],
        "job_prefix": ch["job_prefix"], "gs_root": ch["gs_root"], "scenes_root": ch["scenes_root"],
        "keep_text": True, "sft_records": False,
        "sft_dir": cfg["chain"]["sft_dir"], "sft_groups": dict(cfg["chain"]["sft_groups"]),
    }
    if n["first"]:
        p["sft_lf_dirs"] = {n["sft_group"]: cfg["chain"]["sft_lf_dir"]}
        p["sft_epochs"] = cfg["sft"]["epochs"]
    else:
        p["start_model"] = {"from_chain": n["tr_first"]}
    p.update({
        "eval_sets": dict(ch["eval_sets"]), "select_by": "val", "max_eval_jobs": ch["max_eval_jobs"],
        "pool": ch["pool"], "gpu_log": True, "retries": ch["retries"], "startup_retries_separate": True,
        "record_nodes": True, "suspended_grace_min": ch["suspended_grace_min"], "endpoint": rl["updates"],
        "trainer_attn": ch["trainer_attn"], "rl_name": n["rl_name"],
        "eval": {"every": v["eval"]["every"], "options": []},
        "rl": {"standard": rl["pair"]["s0"], "seed": int(seed), "task_pool": rl["task_pool"], "rows": rl["rows"],
               "exclude": rl["exclude"], "format_example": v["format_example"], "guards": dict(rl["guards"]),
               **ch["rl"]},
        "rl_runs": {n["s0"]: {}, n["d"]: {"rl": v3_dense(dense, v)}},
        "diagnostics": {"failure_breakdown": True, "probes": {"P1": v["p1_probe"], "P0": v["p0_probe"]}},
        "retain": ch["retain"],
        "note": (f"matrix v3 group {group}, seed {seed} (the user's decision B of 2026-11-01): "
                 + ("SFT on the group's mixture, then " if n["first"] else f"from {n['tr_first']}'s SFT pick, ")
                 + f"the S0 / dense pair {n['s0']} / {n['d']}; generated by data_gen.sft_mixture.matrix_plan --version 3"),
    })
    return p


def v3_main(a: argparse.Namespace) -> None:
    seeds = [a.seed] if a.seed else [int(s) for s in v3_vars()["rl"]["seeds"]]
    plans = {s: plan_v3(a.group, s, a.commit, a.dense) for s in seeds}
    for s, p in plans.items():
        text = json.dumps(p, indent=1, ensure_ascii=False)
        if a.out:
            out = Path(a.out.format(seed=s, tr=p["tr"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            print(out)
        else:
            print(text)
        if dense_pending(p):
            print(f"note: {', '.join(dense_pending(p))} is the dense placeholder {DENSE_PENDING}; the chain stops at "
                  "rl_config until the plan is written again with --dense", file=sys.stderr)


if __name__ == "__main__":
    main()

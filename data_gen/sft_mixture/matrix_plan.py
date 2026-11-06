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

    python -m data_gen.sft_mixture.matrix_plan --version 3 --group D_noP0 --commit SHA --sft-chain --out '{tr}/plan.json'
    python -m data_gen.sft_mixture.matrix_plan --version 3 --group D_noP0 --commit SHA --start CHECKPOINT \
        --dense registered [--seed S] --out '{tr}/plan.json'

The cells added once matrix v3 was done (V3_CELLS; the user, 2026-11-03; exploratory, PREREG_D_vs_C.md notes 6
and 7): ``--sft-chain`` writes the SFT-only chain (every epoch evaluated on val and test and kept); the epoch is picked
outside the chain, by the sampling check first and then val; ``--start`` writes one chain per seed from that epoch
(E2, E as redefined on 2026-11-03, has no RL: its SFT-only chain is all; Cr, C's SFT trained again so that every
epoch goes through the sampling check, part C of 2026-11-03, has C's two seeds from its pick, the user's 甲 of 11-04).
Each of these chains evaluates in its own copies of the two screened sets (TR/<set folder>, tasks.jsonl copied when
the plan is written): chains sharing a set folder read each other's results while they are being written (11-03).
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
    ap.add_argument("--sft-chain", action="store_true", help="v3 cells (V3_CELLS): the SFT-only chain")
    ap.add_argument("--start", default=None, help="v3 cells: the picked SFT checkpoint the RL chains start from")
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
# the cells added once matrix v3 was done (the user, 2026-11-03; exploratory): {group: (short name, seeds)}
V3_CELLS = {"D_noP0": ("dn", 2), "B": ("b", 1),   # B': RL only when an epoch passes, one seed
            "E2": ("e2", 0),                    # E as redefined on 2026-11-03: SFT only, no RL
            "Cr": ("cr", 2),                    # C's SFT again (part C, 2026-11-03); RL from its pick, two seeds (11-04)
            "G": ("g", 2)}                      # "point first, then act" (2026-11-06): prep only until the user approves
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
    seeds = [int(s) for s in v["rl"]["seeds"]][:V3_CELLS[group][1] if group in V3_CELLS else None]
    if seed is None and group not in V3_CELLS:
        raise SystemExit(f"{group}: a chain of the first batch needs a seed")
    if seed is not None and int(seed) not in seeds:
        raise SystemExit(f"seed {seed} is not one of the matrix seeds {seeds}")
    k = None if seed is None else seeds.index(int(seed)) + 1   # None: a cell's SFT-only chain
    g = V3_CELLS[group][0] if group in V3_CELLS else group.lower()
    ch = cfg["chain"]
    return {"k": k, "first": k == 1, "tr": f"{ch['tr']}_s{seed}" if k else None,
            "tr_first": f"{ch['tr']}_s{seeds[0]}" if seeds else None, "sft_tr": f"{ch['tr']}_sft",
            "sft_group": next(iter(ch["sft_groups"])), "rl_name": f"{ch['rl_name']}_s{k}" if k else None,
            "s0": f"m3{g}s{k}s0" if k else None, "d": f"m3{g}s{k}d" if k else None, "cfg": cfg}


def plan_v3(group: str, seed: int, commit: str, dense: Any = None) -> Dict[str, Any]:
    """The train-chain plan of one v3 group for one seed (patches 0001 / 0002 applied, qwen3-main's chain).  The first
    seed's chain trains the group's SFT (plan "sft_lf_dirs", the copy in the matrix worktree), evaluates every epoch
    and picks on val; the second seed's starts from that pick ("start_model": {"from_chain": first seed's folder}).
    Each chain trains one S0 / dense pair from the same start with the same tasks, updates and stop lines."""
    if group not in V3_GROUPS:
        raise SystemExit(f"v3 plans exist for {V3_GROUPS} (the first batch); {group} comes later")
    v = v3_vars()
    n = v3_names(group, seed, v)
    note = (f"matrix v3 group {group}, seed {seed} (the user's decision B of 2026-11-01): "
            + ("SFT on the group's mixture, then " if n["first"] else f"from {n['tr_first']}'s SFT pick, ")
            + f"the S0 / dense pair {n['s0']} / {n['d']}; generated by data_gen.sft_mixture.matrix_plan --version 3")
    return _plan_v3(v, n, commit, dense, n["tr"], n["rl_name"], None if n["first"] else {"from_chain": n["tr_first"]},
                    note=note)


def plan_v3_cell_sft(group: str, commit: str) -> Dict[str, Any]:
    """The SFT-only chain of a cell added after matrix v3: the cell's SFT on the copy of its mixture, every epoch
    evaluated on val and test (the chain's own set copies) and kept, no retention (the sampling check needs every
    epoch); the chain's own val pick is not the cell's pick (PREREG note 6)."""
    if group not in V3_CELLS:
        raise SystemExit(f"cells after matrix v3: {sorted(V3_CELLS)}, not {group}")
    v = v3_vars()
    n = v3_names(group, None, v)
    note = (f"matrix v3 cell {group} (the user, 2026-11-03; exploratory): SFT only, every epoch kept; the epoch is "
            "picked by the sampling check, then val (PREREG note 6), and the RL chains start from it; generated by "
            "data_gen.sft_mixture.matrix_plan --version 3 --sft-chain")
    return _plan_v3(v, n, commit, None, n["sft_tr"], f"{n['cfg']['chain']['rl_name']}_sft", None, sft_only=True,
                    eval_sets=v3_own_sets(v, n["sft_tr"]), note=note)


def plan_v3_cell_rl(group: str, seed: int, commit: str, start: str, dense: Any = None) -> Dict[str, Any]:
    """One seed's chain of a cell added after matrix v3: the S0 / dense pair from the picked epoch (a checkpoint
    path), the chain's own set copies, retention and diagnostics as the first batch."""
    if group not in V3_CELLS:
        raise SystemExit(f"cells after matrix v3: {sorted(V3_CELLS)}, not {group}")
    if not str(start or "").startswith("/"):
        raise SystemExit("--start must be the picked SFT checkpoint's absolute path")
    v = v3_vars()
    n = v3_names(group, seed, v)
    note = (f"matrix v3 cell {group}, seed {seed} (the user, 2026-11-03; exploratory): from the picked SFT epoch "
            f"{start}, the S0 / dense pair {n['s0']} / {n['d']}; generated by data_gen.sft_mixture.matrix_plan "
            "--version 3 --start")
    return _plan_v3(v, n, commit, dense, n["tr"], n["rl_name"], str(start), eval_sets=v3_own_sets(v, n["tr"]),
                    note=note)


def v3_own_sets(v: Dict[str, Any], tr: str) -> Dict[str, str]:
    """A chain's own copies of the two screened sets: TR/<the set folder's name>."""
    return {k: f"{tr}/{Path(p).name}" for k, p in v["chain"]["eval_sets"].items()}


def v3_prepare_sets(plan: Dict[str, Any]) -> None:
    """Copy the screened sets' tasks.jsonl into a plan's own set folders, byte for byte; a copy already there is
    checked, never replaced."""
    import hashlib
    import shutil

    shared = v3_vars()["chain"]["eval_sets"]
    for k, folder in plan["eval_sets"].items():
        src, dst = Path(shared[k]) / "tasks.jsonl", Path(folder) / "tasks.jsonl"
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        if hashlib.sha256(dst.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
            raise SystemExit(f"{dst} differs from {src}")


def _plan_v3(v: Dict[str, Any], n: Dict[str, Any], commit: str, dense: Any, tr: str, rl_name: str, start: Any,
             sft_only: bool = False, eval_sets: Optional[Dict[str, str]] = None, note: str = "") -> Dict[str, Any]:
    """A v3 chain's plan: SFT on the group's mixture copy (start None) or RL from ``start``; ``sft_only`` leaves out
    the RL runs, the diagnostics and the retention."""
    if not re.fullmatch(r"[0-9a-f]{40}", str(commit or "")):
        raise SystemExit("--commit must be the full commit the chain checks out")
    cfg, ch, rl = n["cfg"], v["chain"], v["rl"]
    standards = json.loads(STANDARDS.read_text(encoding="utf-8"))
    if rl["pair"]["s0"] not in standards:
        raise SystemExit(f"{rl['pair']['s0']} missing from {STANDARDS}")
    p: Dict[str, Any] = {
        "tr": tr, "commit": commit, "model": v["base_qwen3"], "model_type": v["model_type"],
        "job_prefix": ch["job_prefix"], "gs_root": ch["gs_root"], "scenes_root": ch["scenes_root"],
        "keep_text": True, "sft_records": False,
        "sft_dir": cfg["chain"]["sft_dir"], "sft_groups": dict(cfg["chain"]["sft_groups"]),
    }
    if start is None:
        p["sft_lf_dirs"] = {n["sft_group"]: cfg["chain"]["sft_lf_dir"]}
        p["sft_epochs"] = cfg["sft"]["epochs"]
    else:
        p["start_model"] = start
    p.update({
        "eval_sets": dict(eval_sets or ch["eval_sets"]), "select_by": "val", "max_eval_jobs": ch["max_eval_jobs"],
        "pool": ch["pool"], "gpu_log": True, "retries": ch["retries"], "startup_retries_separate": True,
        "record_nodes": True, "suspended_grace_min": ch["suspended_grace_min"], "endpoint": rl["updates"],
        "trainer_attn": ch["trainer_attn"], "rl_name": rl_name,
        "eval": {"every": v["eval"]["every"], "options": []},
        "rl": {"standard": rl["pair"]["s0"], "seed": int(v["rl"]["seeds"][(n["k"] or 1) - 1]), "task_pool": rl["task_pool"],
               "rows": rl["rows"],
               # group G (2026-11-06): its own format example (a targets format) and the pointing reward, both sides
               "exclude": rl["exclude"], "format_example": cfg.get("format_example") or v["format_example"],
               "guards": dict(rl["guards"]),
               **({"pointing_reward": float(cfg["rl"]["pointing_reward"])}
                  if (cfg.get("rl") or {}).get("pointing_reward") else {}),
               **ch["rl"]},
    })
    if sft_only:
        p["sft_only"] = True
    else:
        p.update({
            "rl_runs": {n["s0"]: {}, n["d"]: {"rl": v3_dense(dense, v)}},
            "diagnostics": {"failure_breakdown": True, "probes": {
                "P1": v["p1_probe"], "P0": v["p0_probe"],
                **({"Pg": cfg["eval"]["pg_probe"]} if (cfg.get("eval") or {}).get("pg_probe") else {})}},  # group G
            "retain": ch["retain"],
        })
    p["note"] = note
    return p


def v3_main(a: argparse.Namespace) -> None:
    seeds = [a.seed] if a.seed else [int(s) for s in v3_vars()["rl"]["seeds"]][:V3_CELLS.get(a.group, (0, None))[1]]
    if a.group not in V3_CELLS:
        plans: Dict[Any, Dict[str, Any]] = {s: plan_v3(a.group, s, a.commit, a.dense) for s in seeds}
    elif a.sft_chain:
        plans = {"sft": plan_v3_cell_sft(a.group, a.commit)}
    elif not seeds:
        raise SystemExit(f"{a.group} has no RL chains (--sft-chain only)")
    else:
        plans = {s: plan_v3_cell_rl(a.group, s, a.commit, a.start, a.dense) for s in seeds}
    for s, p in plans.items():
        text = json.dumps(p, indent=1, ensure_ascii=False)
        if a.out:
            out = Path(a.out.format(seed=s, tr=p["tr"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n", encoding="utf-8")
            if a.group in V3_CELLS:
                v3_prepare_sets(p)
            print(out)
        else:
            print(text)
        if dense_pending(p):
            print(f"note: {', '.join(dense_pending(p))} is the dense placeholder {DENSE_PENDING}; the chain stops at "
                  "rl_config until the plan is written again with --dense", file=sys.stderr)


if __name__ == "__main__":
    main()

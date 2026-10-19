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
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

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
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((MATRIX / f"{a.group}.yaml").read_text(encoding="utf-8"))
    if "train_chain_plan" not in cfg:
        print(json.dumps({"group": a.group, "reuse": reuse(a.group, a.rl_package)}, indent=1, ensure_ascii=False))
        return
    build = plan_with_chain_patch if a.chain_patch else plan
    text = json.dumps(build(a.group, a.rl_package, a.commit), indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

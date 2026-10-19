"""The train-chain plan of one matrix group (configs/matrix_v2/<group>.yaml) with its variables filled in.

    python -m data_gen.sft_mixture.matrix_plan --group D [--rl-package v10_p2] [--commit SHA] [--out plan.json]

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
    a = ap.parse_args(argv)
    cfg = yaml.safe_load((MATRIX / f"{a.group}.yaml").read_text(encoding="utf-8"))
    if "train_chain_plan" not in cfg:
        print(json.dumps({"group": a.group, "reuse": reuse(a.group, a.rl_package)}, indent=1, ensure_ascii=False))
        return
    text = json.dumps(plan(a.group, a.rl_package, a.commit), indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

"""Ablation configs (task spec 6.6) and the exactly-one-switch check.

    python -m eval.ablation ladder --base configs/L0_exec_only.yaml --steps plan,check,goal,recover \
                                   --from configs/L4_recover.yaml --out configs/ladder/
    python -m eval.ablation loo --final configs/L4_recover.yaml --out configs/loo/
    python -m eval.ablation check A.yaml B.yaml [--slot plan]

ladder: forward ladder. Step k inherits from step k-1 (step 1 from --base) and enables exactly the k-th
        slot, copying that slot's resolved block from the --from config.
loo:    leave-one-out. One config per slot enabled in --final, inheriting from it and setting only
        ``modules.<slot>: {enabled: false}`` (the registry then builds the slot's null impl).
check:  the two RESOLVED configs (agent.config.load_config, ``base:`` chains followed) must differ in
        exactly one ``modules.<slot>`` block. Outside ``modules`` only label keys (run.name) may differ.

Generated files carry ``base:`` inheritance, the switched module block and a ``run.name`` label (so
result tables can tell the configs apart), with a comment header naming the parent and the slot.
ladder and loo re-load every file they write and run ``check`` against its parent; a failure undoes
that call's writes (files it created are removed, files it overwrote are restored) and exits non-zero.

Status: written and unit-tested against the real configs/ files; no ablation has run on 105 yet.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional, Sequence

import yaml

from agent.config import REPO_ROOT, load_config
from agent.interfaces import SLOT_PROTOCOLS

LABEL_KEYS = frozenset({"run.name"})


def resolved(path: str) -> dict:
    cfg = load_config(path)
    cfg.pop("_meta", None)
    return cfg


def _flatten(x: Any, prefix: str = "") -> dict:
    if isinstance(x, dict) and x:
        out = {}
        for k, v in x.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
        return out
    return {prefix: x}


_ABSENT = "<absent>"


def diff_configs(a: dict, b: dict) -> list[tuple[str, Any, Any]]:
    fa, fb = _flatten(a), _flatten(b)
    pairs = ((k, fa.get(k, _ABSENT), fb.get(k, _ABSENT)) for k in sorted(set(fa) | set(fb)))
    return [(k, va, vb) for k, va, vb in pairs if va != vb]


def switch_report(a: dict, b: dict) -> dict:
    diffs = diff_configs(a, b)
    slots = sorted({k.split(".")[1] for k, _, _ in diffs if k.startswith("modules.")})
    other = [d for d in diffs if not d[0].startswith("modules.") and d[0] not in LABEL_KEYS]
    labels = [d for d in diffs if d[0] in LABEL_KEYS]
    return {"ok": len(slots) == 1 and not other, "slots": slots, "diffs": diffs, "other": other, "labels": labels}


def format_report(rep: dict, a_name: str, b_name: str) -> str:
    lines = [f"A: {a_name}", f"B: {b_name}"]
    for k, va, vb in rep["diffs"]:
        tag = "label" if k in LABEL_KEYS else ("module" if k.startswith("modules.") else "OTHER")
        lines.append(f"  [{tag}] {k}: {va!r} -> {vb!r}")
    if rep["ok"]:
        lines.append(f"OK: exactly one module slot switched: {rep['slots'][0]}")
    else:
        why = []
        if len(rep["slots"]) != 1:
            why.append(f"{len(rep['slots'])} module slots differ {rep['slots']}")
        if rep["other"]:
            why.append(f"{len(rep['other'])} non-module key(s) differ")
        lines.append("FAIL: " + "; ".join(why))
    return "\n".join(lines)


def check(a_path: str, b_path: str, slot: Optional[str] = None) -> tuple[bool, str, dict]:
    rep = switch_report(resolved(a_path), resolved(b_path))
    if slot is not None and rep["slots"] != [slot]:
        rep["ok"] = False
    text = format_report(rep, _show(a_path), _show(b_path))
    if slot is not None and rep["slots"] != [slot]:
        text += f"\nFAIL: expected the switched slot to be {slot!r}"
    return rep["ok"], text, rep


def _show(path: str) -> str:
    p = os.path.abspath(path)
    return os.path.relpath(p, REPO_ROOT) if p.startswith(REPO_ROOT + os.sep) else p


def _flow(x: Any) -> str:
    return yaml.safe_dump(x, default_flow_style=True, sort_keys=False, width=10_000).strip()


def render_config(parent_rel: str, run_name: str, slot: str, block: dict, header: Sequence[str]) -> str:
    lines = [f"# {h}" for h in header]
    lines += [f"base: {parent_rel}", f"run: {_flow({'name': run_name})}", "modules:", f"  {slot}: {_flow(block)}"]
    return "\n".join(lines) + "\n"


def _write(path: str, text: str, written: list, backup: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path not in backup:
        backup[path] = None
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                backup[path] = f.read()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    written.append(path)


def _fail(backup: dict, msg: str) -> None:
    """Undo one generation: remove the files it created, restore the ones it overwrote (e.g. configs/ladder)."""
    for p, old in backup.items():
        if old is None:
            if os.path.exists(p):
                os.remove(p)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(old)
    raise SystemExit(msg)


def _command(argv: Optional[Sequence[str]]) -> str:
    return "python -m eval.ablation " + " ".join(argv) if argv else "eval.ablation"


def stale_files(out_dir: str, prefix: str, written: Sequence[str]) -> list[str]:
    """Generated-looking files in ``out_dir`` that this call did not write (left by an earlier generation)."""
    if not os.path.isdir(out_dir):
        return []
    mine = {os.path.basename(p) for p in written}
    return sorted(os.path.join(out_dir, f) for f in os.listdir(out_dir)
                  if f.startswith(prefix) and f.endswith(".yaml") and f not in mine)


def _note_stale(out_dir: str, prefix: str, written: Sequence[str]) -> None:
    for p in stale_files(out_dir, prefix, written):
        print(f"note: {_show(p)} was not written by this generation (stale?); remove it if it is", file=sys.stderr)


def ladder(base: str, final: str, steps: Sequence[str], out_dir: str,
           argv: Optional[Sequence[str]] = None) -> list[str]:
    unknown = [s for s in steps if s not in SLOT_PROTOCOLS]
    if unknown or len(set(steps)) != len(steps) or not steps:
        raise SystemExit(f"--steps must be distinct slots from {list(SLOT_PROTOCOLS)}, got {list(steps)}")
    base_cfg, final_cfg = resolved(base), resolved(final)
    for s in steps:
        if (base_cfg["modules"].get(s) or {}).get("enabled"):
            raise SystemExit(f"slot {s!r} is already enabled in the base config {_show(base)}")
        if not (final_cfg["modules"].get(s) or {}).get("enabled"):
            raise SystemExit(f"slot {s!r} is not enabled in the final config {_show(final)}")
    out_dir = os.path.abspath(out_dir)
    written: list[str] = []
    backup: dict = {}
    parent = os.path.abspath(base)
    for k, slot in enumerate(steps, 1):
        name = f"ladder{k}_{slot}"
        path = os.path.join(out_dir, f"{name}.yaml")
        block = final_cfg["modules"][slot]
        header = [f"GENERATED by `{_command(argv)}`; regenerate instead of editing.",
                  f"Forward ladder step {k}/{len(steps)}: parent {_show(parent)}",
                  f"switched slot: modules.{slot} (off -> enabled, block copied from {_show(final)})"]
        _write(path, render_config(os.path.relpath(parent, out_dir), name, slot, block, header), written, backup)
        ok, text, _ = check(parent, path, slot)
        if not ok or resolved(path)["modules"][slot] != block:
            _fail(backup, f"self-check failed for {_show(path)}:\n{text}")
        parent = path
    rest = switch_report(resolved(parent), final_cfg)
    left = [d for d in rest["diffs"] if d[0] not in LABEL_KEYS]
    if left:
        print(f"note: the last ladder step differs from {_show(final)} in {len(left)} key(s):", file=sys.stderr)
        for k, va, vb in left:
            print(f"  {k}: {va!r} -> {vb!r}", file=sys.stderr)
    _note_stale(out_dir, "ladder", written)
    return written


def loo(final: str, out_dir: str, argv: Optional[Sequence[str]] = None) -> list[str]:
    final_cfg = resolved(final)
    enabled = [s for s in SLOT_PROTOCOLS if (final_cfg["modules"].get(s) or {}).get("enabled")]
    if not enabled:
        raise SystemExit(f"no module enabled in {_show(final)}")
    out_dir = os.path.abspath(out_dir)
    written: list[str] = []
    backup: dict = {}
    for slot in enabled:
        name = f"loo_no_{slot}"
        path = os.path.join(out_dir, f"{name}.yaml")
        header = [f"GENERATED by `{_command(argv)}`; regenerate instead of editing.",
                  f"Leave-one-out: parent {_show(final)}",
                  f"switched slot: modules.{slot} (enabled -> off; the registry builds the null impl)"]
        _write(path, render_config(os.path.relpath(os.path.abspath(final), out_dir), name, slot, {"enabled": False},
                                   header), written, backup)
        ok, text, _ = check(final, path, slot)
        if not ok:
            _fail(backup, f"self-check failed for {_show(path)}:\n{text}")
    _note_stale(out_dir, "loo_", written)
    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ladder", help="forward ladder: enable one slot per step")
    p.add_argument("--base", required=True)
    p.add_argument("--steps", required=True, help="comma list of slots, in ladder order")
    p.add_argument("--from", dest="final", required=True, help="config whose module blocks the steps copy")
    p.add_argument("--out", required=True)
    p = sub.add_parser("loo", help="leave-one-out: disable one enabled slot per config")
    p.add_argument("--final", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("check", help="assert two configs differ in exactly one modules.<slot> block")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--slot", default=None, help="also require that this is the switched slot")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        ok, text, _ = check(args.a, args.b, args.slot)
        print(text)
        return 0 if ok else 1
    if args.cmd == "ladder":
        paths = ladder(args.base, args.final, [s.strip() for s in args.steps.split(",") if s.strip()], args.out, argv)
    else:
        paths = loo(args.final, args.out, argv)
    for path in paths:
        print(f"wrote {_show(path)} (self-check OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

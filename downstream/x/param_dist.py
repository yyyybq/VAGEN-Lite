#!/usr/bin/env python3
"""Parameter distances (the user's decision of 2026-10-30, "按你拟的做", plus "E1 four vs their own starts" per
layer; design reports/X0B_PD_LP_DESIGN_20261028.md).

    python downstream/x/param_dist.py compute --group qwen25 --out PD_QWEN25.json   # B, S, RS, RD and E1's four
    python downstream/x/param_dist.py compute --group qwen3  --out PD_QWEN3.json    # Qwen3 base, SFT, RS, RD
    python downstream/x/param_dist.py plot --json PD_QWEN25.json [PD_QWEN3.json] --out-dir DIR   # matplotlib

Every weight is read by path, one tensor at a time (safetensors of the HF folders; the E1 final checkpoints'
pytorch_model.pt memory-mapped), compared in float32 and dropped; nothing is written but the results.  Names are
brought to one form (StarVLA's "qwen_vl_interface.model." prefix and transformers' "model." / "language_model."
removed), and each tensor is put in a module: vision block i, the vision patch / position embedding, the merger(s),
language-model layer i, the final norm, the token embedding, lm_head (the StarVLA action head is not compared).
Per module and in total, for every difference A - B: the relative distance |A - B| / |B|; for pairs of differences
with the same base, the cosine of the two.  Groups:
    qwen25  S - B, RS - B, RD - B (what the upstream training changed); RS - S, RD - S (the RL part, S being the
            round-11 SFT the RL started from); E1-X - X for X in B, S, RS, RD (what downstream training changed in
            each start); cosines S-B vs RS-B, S-B vs RD-B, RS-B vs RD-B, RS-S vs RD-S, and the four E1 differences
            pairwise
    qwen3   Q3S - Q3B, Q3RS - Q3B, Q3RD - Q3B; Q3RS - Q3S, Q3RD - Q3S (the RL started from SFT epoch 8); cosines
            as for qwen25
Tensors whose shapes differ between the two sides are skipped and listed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import q0_ask  # noqa: E402

E1 = "/mnt/umm/users/yinbaiqiao/downstream_runs/experiments/DS-{}-100-s1/run/final_model/pytorch_model.pt"
GROUPS = {
    "qwen25": {
        "sources": {"B": None, "S": None, "RS": None, "RD": None,
                    "E1-B": E1.format("B"), "E1-S": E1.format("S"), "E1-RS": E1.format("RS"), "E1-RD": E1.format("RD")},
        "deltas": [("S-B", "S", "B"), ("RS-B", "RS", "B"), ("RD-B", "RD", "B"), ("RS-S", "RS", "S"), ("RD-S", "RD", "S"),
                   ("E1B-B", "E1-B", "B"), ("E1S-S", "E1-S", "S"), ("E1RS-RS", "E1-RS", "RS"), ("E1RD-RD", "E1-RD", "RD")],
        "cosines": [("S-B", "RS-B"), ("S-B", "RD-B"), ("RS-B", "RD-B"), ("RS-S", "RD-S"),
                    ("E1B-B", "E1S-S"), ("E1B-B", "E1RS-RS"), ("E1B-B", "E1RD-RD"), ("E1S-S", "E1RS-RS"),
                    ("E1S-S", "E1RD-RD"), ("E1RS-RS", "E1RD-RD")],
    },
    "qwen3": {
        "sources": {"Q3B": None, "Q3S": None, "Q3RS": None, "Q3RD": None},
        "deltas": [("Q3S-Q3B", "Q3S", "Q3B"), ("Q3RS-Q3B", "Q3RS", "Q3B"), ("Q3RD-Q3B", "Q3RD", "Q3B"),
                   ("Q3RS-Q3S", "Q3RS", "Q3S"), ("Q3RD-Q3S", "Q3RD", "Q3S")],
        "cosines": [("Q3S-Q3B", "Q3RS-Q3B"), ("Q3S-Q3B", "Q3RD-Q3B"), ("Q3RS-Q3B", "Q3RD-Q3B"), ("Q3RS-Q3S", "Q3RD-Q3S")],
    },
}


def canon(name: str):
    """(canonical name, module) of a weight name, or None for weights outside the VLM (the action head)."""
    n = name
    if n.startswith("qwen_vl_interface.model."):
        n = n[len("qwen_vl_interface.model."):]
    elif n.startswith("qwen_vl_interface.") or n.startswith("action_model") or n.startswith("module."):
        return None
    if n.startswith("model."):
        n = n[len("model."):]
    if n.startswith("language_model."):
        n = n[len("language_model."):]
    m = re.match(r"visual\.blocks\.(\d+)\.", n)
    if m:
        return n, f"vision.{int(m.group(1)):02d}"
    if n.startswith("visual.merger") or n.startswith("visual.deepstack_merger"):
        return n, "merger"
    if n.startswith("visual."):
        return n, "vision.embed"
    m = re.match(r"layers\.(\d+)\.", n)
    if m:
        return n, f"llm.{int(m.group(1)):02d}"
    if n.startswith("embed_tokens"):
        return n, "embed"
    if n.startswith("norm."):
        return n, "llm.norm"
    if n.startswith("lm_head"):
        return n, "lm_head"
    return n, "other"


def coarse(module: str) -> str:
    return module.split(".")[0] if module.split(".")[0] in ("vision", "llm") else module


class HFReader:
    def __init__(self, folder: str):
        self.folder = Path(folder)
        idx = self.folder / "model.safetensors.index.json"
        if idx.is_file():
            wm = json.loads(idx.read_text())["weight_map"]
        else:
            from safetensors import safe_open

            with safe_open(str(self.folder / "model.safetensors"), framework="pt") as f:
                wm = {k: "model.safetensors" for k in f.keys()}
        self.names = {}
        for raw, shard in wm.items():
            c = canon(raw)
            if c:
                self.names[c[0]] = (raw, shard)
        self._open = {}

    def get(self, cname: str):
        from safetensors import safe_open

        raw, shard = self.names[cname]
        if shard not in self._open:
            self._open[shard] = safe_open(str(self.folder / shard), framework="pt")
        return self._open[shard].get_tensor(raw)


class PTReader:
    def __init__(self, path: str):
        import torch

        sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        for key in ("state_dict", "module", "model"):
            if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
                sd = sd[key]
        self.sd = sd
        self.names = {}
        for raw in sd:
            c = canon(raw)
            if c:
                self.names[c[0]] = (raw, None)

    def get(self, cname: str):
        return self.sd[self.names[cname][0]]


def reader(name: str, path):
    if path is None:
        return HFReader(q0_ask.model_path(name))
    return PTReader(path) if str(path).endswith(".pt") else HFReader(path)


def compute(group: dict, readers: dict, log=print) -> dict:
    """Sums per module for every delta and cosine of the group (see the module text)."""
    import torch

    deltas, cosines = group["deltas"], group["cosines"]
    bases = sorted({b for _, _, b in deltas})
    names = sorted(set.intersection(*[set(r.names) for r in readers.values()]))
    only = {s: sorted(set(r.names) - set(names)) for s, r in readers.items()}
    acc, skipped, modules = {}, [], {}
    t0 = time.time()
    for i, cname in enumerate(names):
        module = canon(cname)[1]
        modules.setdefault(module, 0)
        tensors = {s: r.get(cname).to(torch.float32) for s, r in readers.items()}
        shapes = {tuple(t.shape) for t in tensors.values()}
        if len(shapes) != 1:
            skipped.append([cname, {s: list(t.shape) for s, t in tensors.items()}])
            continue
        modules[module] += tensors[bases[0]].numel()
        a = acc.setdefault(module, {"base_sq": {b: 0.0 for b in bases}, "diff_sq": {d[0]: 0.0 for d in deltas},
                                    "dot": {f"{x}|{y}": 0.0 for x, y in cosines}})
        for b in bases:
            a["base_sq"][b] += float(torch.sum(tensors[b] * tensors[b]))
        diff = {}
        for label, x, y in deltas:
            diff[label] = tensors[x] - tensors[y]
            a["diff_sq"][label] += float(torch.sum(diff[label] * diff[label]))
        for x, y in cosines:
            a["dot"][f"{x}|{y}"] += float(torch.sum(diff[x] * diff[y]))
        del tensors, diff
        if i % 100 == 0:
            log(f"[param_dist] {i}/{len(names)} tensors, {time.time() - t0:.0f} s")
    return {"modules": modules, "sums": acc, "skipped": skipped, "only_in": {k: v for k, v in only.items() if v},
            "tensors": len(names)}


def summarize(group: dict, raw: dict) -> dict:
    """Relative distances and cosines per module, per coarse module (vision, merger, llm, embed, lm_head) and
    overall, from compute()'s sums."""
    import math

    def finish(s):
        rel = {lab: math.sqrt(s["diff_sq"][lab]) / math.sqrt(s["base_sq"][b]) if s["base_sq"][b] else float("nan")
               for lab, _, b in group["deltas"]}
        cos = {}
        for x, y in group["cosines"]:
            den = math.sqrt(s["diff_sq"][x] * s["diff_sq"][y])
            cos[f"{x}|{y}"] = s["dot"][f"{x}|{y}"] / den if den else float("nan")
        return {"relative": rel, "cosine": cos}

    def add(into, s):
        for part in ("base_sq", "diff_sq", "dot"):
            for k, v in s[part].items():
                into[part][k] = into[part].get(k, 0.0) + v

    out = {"per_module": {}, "coarse": {}, "all": None}
    total = {"base_sq": {}, "diff_sq": {}, "dot": {}}
    coarse_sums = {}
    for module, s in sorted(raw["sums"].items()):
        out["per_module"][module] = finish(s)
        add(total, s)
        add(coarse_sums.setdefault(coarse(module), {"base_sq": {}, "diff_sq": {}, "dot": {}}), s)
    out["coarse"] = {c: finish(s) for c, s in coarse_sums.items()}
    out["all"] = finish(total)
    return out


def compute_cmd(args) -> int:
    import torch

    torch.set_num_threads(args.threads)
    group = GROUPS[args.group]
    readers = {}
    for s, path in group["sources"].items():
        readers[s] = reader(s, path)
        print(f"[param_dist] {s}: {len(readers[s].names)} tensors", flush=True)
    raw = compute(group, readers, log=lambda m: print(m, flush=True))
    res = {"group": args.group, "sources": {s: str(p or q0_ask.model_path(s)) for s, p in group["sources"].items()},
           "deltas": group["deltas"], "cosines": group["cosines"], "parameters_per_module": raw["modules"],
           "skipped": raw["skipped"], "only_in": raw["only_in"], "tensors": raw["tensors"],
           **summarize(group, raw)}
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"all": res["all"], "skipped": len(raw["skipped"])}), flush=True)
    return 0


def module_order(modules) -> list:
    def key(m):
        head = m.split(".")[0]
        rank = {"vision": 0, "merger": 1, "llm": 2, "embed": 3, "lm_head": 4}.get(head, 5)
        tail = m.split(".")[1] if "." in m else ""
        return (rank, 0 if tail.isdigit() else 1, int(tail) if tail.isdigit() else 0, m)
    return sorted(modules, key=key)


def plot_cmd(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels = [("qwen25", "e1_vs_starts", ["E1B-B", "E1S-S", "E1RS-RS", "E1RD-RD"],
               "E1 (downstream 100%, 200k steps) vs its own start"),
              ("qwen25", "upstream_vs_B", ["S-B", "RS-B", "RD-B"], "Qwen2.5 upstream training vs B"),
              ("qwen3", "qwen3_vs_base", ["Q3S-Q3B", "Q3RS-Q3B", "Q3RD-Q3B"], "Qwen3 upstream training vs base")]
    data = {}
    for p in args.json:
        r = json.loads(Path(p).read_text())
        data[r["group"]] = r
    for group, name, labels, title in panels:
        if group not in data:
            continue
        r = data[group]
        mods = [m for m in module_order(r["per_module"]) if m not in ("other",)]
        fig, ax = plt.subplots(figsize=(13, 4.2))
        for lab in labels:
            ax.plot(range(len(mods)), [r["per_module"][m]["relative"][lab] for m in mods], marker=".", label=lab)
        ticks = [i for i, m in enumerate(mods) if not m.split(".")[-1].isdigit() or int(m.split(".")[-1]) % 4 == 0]
        ax.set_xticks(ticks)
        ax.set_xticklabels([mods[i] for i in ticks], rotation=60, fontsize=7)
        ax.set_yscale("log")
        ax.set_ylabel("|A - B| / |B|")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / f"param_dist_{name}.png", dpi=130)
        plt.close(fig)
        print(f"[param_dist] {out / f'param_dist_{name}.png'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--group", choices=sorted(GROUPS), required=True)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--out", required=True)
    p = sub.add_parser("plot")
    p.add_argument("--json", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    return {"compute": compute_cmd, "plot": plot_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())

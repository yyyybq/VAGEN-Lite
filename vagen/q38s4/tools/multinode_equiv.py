"""C line (q38 s4), stage 2 test 5: the same small run on 1 node x 4 GPUs and on 2 nodes x 2 GPUs.

  python -m vagen.q38s4.tools.multinode_equiv build --root ROOT
      ROOT/model: a tiny random Qwen3.5 (tests/tiny.py config, 2 layers, hidden 64) with Qwen3.8-27B's processor
                  files, so LF loads it like the 27B;
      ROOT/media + ROOT/vsi.jsonl: 16 synthetic videos (4-10 s, 64x64) with 4 QA rows each, 4 image rows (one with
                  4 turns), in VSI-590K's format;
      ROOT/data:  vagen.q38s4.data_build of that source (--probe, 60 NFP rows, nothing held out).
  then vagen.q38s4.launch: a tokenize run, and two train runs of the same arm and global batch 8 (world 4,
  accumulation 2: the same samples on the same ranks in the same order), one with --nodes 1 --gpus-per-node 4, one
  with --nodes 2 --gpus-per-node 2.
  python -m vagen.q38s4.tools.multinode_equiv compare --a RUN_A --b RUN_B [--rtol 1e-2]
      reads each run's log history (ckpt/trainer_state.json) and compares every logged number step by step: the
      first step's loss must agree to 1e-6 (same weights, same inputs; the forward has no cross-rank sum), every
      number of every step to --rtol (the gradient reduce-scatter adds in another order across nodes); the step
      counts must match.  Writes compare.json next to RUN_B and exits 1 on a failure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

KEYS = ("loss", "grad_norm", "learning_rate", "text_loss", "nfp_mse", "nfp_cos", "copy_mse", "copy_cos", "nfp_pairs",
        "nfp_videos")
Q = "<image>\nThese are frames of a video.\nHow many {x}?\nOptions:\nA. 1\nB. 2\nC. 3\nAnswer with the letter."


def build(root: Path) -> None:
    import llamafactory.data.mm_plugin  # noqa: F401  before PyAV (see lf_plugin)
    import av
    import numpy as np
    from PIL import Image

    from vagen.q38s4 import data_build as D
    from vagen.q38s4.tests import tiny

    if root.exists():
        raise SystemExit(f"{root} exists")
    (root / "media" / "syn").mkdir(parents=True)
    rng = np.random.default_rng(0)
    model = tiny.build(seed=0)
    model.save_pretrained(root / "model")
    tiny.processor().save_pretrained(root / "model")
    rows: List[Dict[str, Any]] = []
    for i in range(16):
        n = 40 + 4 * i  # 4.0 .. 10.0 s at 10 fps
        rel = f"syn/v{i:02d}.mp4"
        c = av.open(str(root / "media" / rel), "w")
        s = c.add_stream("mpeg4", rate=10)
        s.width, s.height, s.pix_fmt, s.bit_rate = 64, 64, "yuv420p", 4_000_000
        for _ in range(n):
            arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
        c.close()
        for j, x in enumerate(("chairs", "tables", "lamps", "doors")):
            rows.append({"conversations": [{"from": "human", "value": Q.format(x=x)},
                                           {"from": "gpt", "value": "ABC"[(i + j) % 3]}],
                         "video": rel, "question_type": "object_counting"})
    for k in range(4):
        rel = f"syn/img{k}.png"
        Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(root / "media" / rel)
        conv = [{"from": "human", "value": Q.format(x="beds")}, {"from": "gpt", "value": "B"}]
        if k == 0:
            conv += [{"from": "human", "value": "And sofas?"}, {"from": "gpt", "value": "A"}]
        rows.append({"conversations": conv, "image": rel, "question_type": "object_counting"})
    (root / "vsi.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    m = D.build(D.parse_args(["--src", str(root / "vsi.jsonl"), "--media-root", str(root / "media"),
                              "--out", str(root / "data"), "--nfp-rows", "60", "--heldout-frac", "0", "--probe",
                              "--video-maxlen", "32"]))
    print(json.dumps({"model": str(root / "model"), "rows_out": m["rows_out"]}, indent=1))


def history(run: Path) -> List[Dict[str, Any]]:
    p = run / "ckpt" / "trainer_state.json"
    if not p.is_file():
        ck = sorted((run / "ckpt").glob("checkpoint-*/trainer_state.json"), key=lambda x: int(x.parent.name[11:]))
        if not ck:
            raise SystemExit(f"{run}: no trainer_state.json")
        p = ck[-1]
    return [h for h in json.loads(p.read_text())["log_history"] if "loss" in h]


def compare(a: List[Dict[str, Any]], b: List[Dict[str, Any]], rtol: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"steps_a": len(a), "steps_b": len(b), "keys": {}, "failures": []}
    if len(a) != len(b) or not a:
        out["failures"].append(f"step counts {len(a)} vs {len(b)}")
        return out
    for k in KEYS:
        pairs = [(x.get(k), y.get(k)) for x, y in zip(a, b)]
        if all(p == (None, None) for p in pairs):
            continue
        if any((x is None) != (y is None) for x, y in pairs):
            out["failures"].append(f"{k}: logged in one run only")
            continue
        rel = [abs(x - y) / max(abs(x), abs(y), 1e-12) if x != y else 0.0 for x, y in pairs if x is not None]
        out["keys"][k] = {"max_abs": max(abs(x - y) for x, y in pairs if x is not None), "max_rel": max(rel),
                          "first_a": pairs[0][0], "first_b": pairs[0][1]}
        if max(rel) > rtol or any(math.isnan(x) or math.isnan(y) for x, y in pairs if x is not None):
            out["failures"].append(f"{k}: max relative difference {max(rel):.3g} > {rtol}")
    l0a, l0b = a[0]["loss"], b[0]["loss"]
    if abs(l0a - l0b) > 1e-6 * max(abs(l0a), 1e-12):
        out["failures"].append(f"first-step loss {l0a} vs {l0b}")
    if [x.get("step") for x in a] != [y.get("step") for y in b]:
        out["failures"].append("logged steps differ")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--root", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--rtol", type=float, default=1e-2)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(Path(a.root).resolve())
        return
    res = compare(history(Path(a.a)), history(Path(a.b)), a.rtol)
    res.update(a=str(Path(a.a).resolve()), b=str(Path(a.b).resolve()), rtol=a.rtol, ok=not res["failures"])
    (Path(a.b) / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()

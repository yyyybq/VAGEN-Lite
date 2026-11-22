"""C line (q38 s4): VSI-590K -> LLaMA-Factory sharegpt data with D4 sampling marks and the NFP rows.

  python -m vagen.q38s4.data_build --src VSI-590K/vsi_590k.jsonl --media-root MEDIA --out OUT [--probe] ...

Writes into OUT (which must not exist yet; nothing is overwritten or deleted):

- qa.jsonl: every source row (590,667), in source order.  Roles human -> user, gpt -> assistant, every message
  stripped; a video row's ``<image>`` placeholder (the source writes one in the first user turn whatever the medium)
  becomes ``<video>`` -- with the chat template's <|vision_start|> ... <|vision_end|> around it for --video-markers
  outer (what Qwen3.8's processor and vLLM write at evaluation, vagen.q38.sft.lf_data), as it is for none -- and its
  path is MEDIA/<video>|s4|qa (all 256 frames, lf_video); an image row keeps <image> and MEDIA/<image>.  Multi-turn
  rows stay multi-turn (all assistant turns trained, as in the 8B SFT: mask_history false);
- nfp.jsonl: --nfp-rows rows (290,000 by default) drawn without replacement, with --seed, from the video rows whose
  video is not held out (and, with --probe, is long enough for an NFP window); each is the user turn's video
  placeholder alone and the assistant turn "." (lf_nfp ignores every label of an NFP sample), the video
  MEDIA/<video>|s4|nfp:u=<u> with u uniform in [0, 1) from the same seed;
- nfp_heldout.jsonl: --heldout-windows NFP rows per held-out video (u = (k + 0.5) / windows), for the NFP loss on
  unseen videos; never trained on;
- heldout_videos.txt: the --heldout-frac of the distinct videos (sorted names, drawn with --seed); with
  --heldout-scope all their QA rows go to qa_heldout.jsonl instead of qa.jsonl, with nfp (the default) only the NFP
  rows avoid them;
- dataset_info.json: s4_qa, s4_nfp, s4_nfp_heldout (and s4_qa_heldout);
- build_manifest.json: arguments, source sha256, output sha256 and line counts, row counts per source / medium,
  the probe's counts and every video it found too short or unreadable.

A row that does not have the expected shape (roles alternating from user, exactly one <image> in the first user
turn and none elsewhere, one medium) stops the build.  --probe opens every distinct video's container (metadata
only, no decoding): a video that cannot be read stops the build; one too short for an NFP window (under 4 frames at
--video-fps) leaves the NFP pool and is listed.  Without --probe nothing is checked on disk (tests, dry runs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from vagen.q38s4.lf_video import VideoSpec, encode, nfp_indices

VIDEO_MARKERS = ("outer", "none")
VIDEO_OUTER = "<|vision_start|><video><|vision_end|>"
ROLES = {"human": "user", "gpt": "assistant"}
NFP_ANSWER = "."
HELDOUT_SCOPES = ("nfp", "all")


class BadRow(ValueError):
    pass


def video_placeholder(markers: str) -> str:
    if markers not in VIDEO_MARKERS:
        raise ValueError(f"video markers {markers!r} not in {VIDEO_MARKERS}")
    return VIDEO_OUTER if markers == "outer" else "<video>"


def medium(rec: Dict[str, Any]) -> Tuple[str, str]:
    has_v, has_i = "video" in rec, "image" in rec
    if has_v == has_i:
        raise BadRow("a row needs exactly one of video / image")
    kind = "video" if has_v else "image"
    path = rec[kind]
    if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
        raise BadRow(f"bad {kind} path {path!r}")
    return kind, path


def convert_row(rec: Dict[str, Any], media_root: str, markers: str) -> Dict[str, Any]:
    """One source row -> an LF sharegpt row (messages + videos or images)."""
    kind, path = medium(rec)
    conv = rec.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2 or len(conv) % 2:
        raise BadRow("conversations must be (human, gpt)+")
    msgs = []
    for i, turn in enumerate(conv):
        want = "human" if i % 2 == 0 else "gpt"
        if turn.get("from") != want or not isinstance(turn.get("value"), str):
            raise BadRow(f"turn {i}: expected {want}")
        text = turn["value"]
        n = text.count("<image>")
        if "<video>" in text or n != (1 if i == 0 else 0):
            raise BadRow(f"turn {i}: {n} <image> placeholders")
        if i == 0 and kind == "video":
            text = text.replace("<image>", video_placeholder(markers))
        text = text.strip()
        if not text:
            raise BadRow(f"turn {i}: empty")
        msgs.append({"role": ROLES[want], "content": text})
    full = os.path.join(media_root, path)
    if kind == "video":
        return {"messages": msgs, "videos": [encode(full, VideoSpec("qa"))]}
    return {"messages": msgs, "images": [full]}


def nfp_row(media_root: str, video: str, u: float, markers: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": video_placeholder(markers)},
                         {"role": "assistant", "content": NFP_ANSWER}],
            "videos": [encode(os.path.join(media_root, video), VideoSpec("nfp", u))]}


def pick_heldout(videos: Iterable[str], frac: float, seed: int) -> List[str]:
    names = sorted(set(videos))
    k = int(round(frac * len(names)))
    if frac > 0 and k == 0:
        k = 1
    return sorted(random.Random(f"heldout:{seed}").sample(names, k))


def probe_videos(media_root: str, videos: Iterable[str], fps: float, maxlen: int) -> Dict[str, Dict[str, Any]]:
    """Per distinct video: total frames, fps, duration, whether an NFP window fits (metadata only)."""
    import av

    from vagen.q38s4.lf_video import stream_info

    out = {}
    for v in sorted(set(videos)):
        path = os.path.join(media_root, v)
        try:
            c = av.open(path, "r")
        except (av.FFmpegError, OSError) as e:
            out[v] = {"error": f"{type(e).__name__}: {e}"}
            continue
        try:
            s = next((x for x in c.streams if x.type == "video"), None)
            if s is None:
                out[v] = {"error": "no video stream"}
                continue
            try:
                info = stream_info(c, s)
            except ValueError as e:
                out[v] = {"error": str(e)}
                continue
            try:
                nfp_indices(info["total_frames"], info["average_fps"], 0.0, fps, maxlen)
                info["nfp_ok"] = True
            except ValueError:
                info["nfp_ok"] = False
            out[v] = info
        finally:
            c.close()
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dataset_entry(file_name: str, media: str) -> Dict[str, Any]:
    return {"file_name": file_name, "formatting": "sharegpt",
            "columns": {"messages": "messages", media: media},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
                     "system_tag": "system"}}


def dataset_info(with_qa_heldout: bool) -> Dict[str, Any]:
    info = {}
    for name, fn in (("s4_qa", "qa.jsonl"), ("s4_nfp", "nfp.jsonl"), ("s4_nfp_heldout", "nfp_heldout.jsonl")):
        info[name] = dataset_entry(fn, "videos")
        if name == "s4_qa":
            info[name]["columns"]["images"] = "images"
    if with_qa_heldout:
        info["s4_qa_heldout"] = dataset_entry("qa_heldout.jsonl", "videos")
        info["s4_qa_heldout"]["columns"]["images"] = "images"
    return info


def build(args: argparse.Namespace) -> Dict[str, Any]:
    src, out = Path(args.src), Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: pick a new folder (the build never overwrites)")
    media_root = os.path.abspath(args.media_root)
    video_placeholder(args.video_markers)
    if args.heldout_scope not in HELDOUT_SCOPES:
        raise SystemExit(f"--heldout-scope must be one of {HELDOUT_SCOPES}")

    rows: List[Tuple[str, str]] = []  # (kind, path) per source row
    counts: Counter = Counter()
    with open(src, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            rec = json.loads(line)
            try:
                convert_row(rec, media_root, args.video_markers)
                kind, path = medium(rec)
            except BadRow as e:
                raise SystemExit(f"{src}:{ln}: {e}") from None
            rows.append((kind, path))
            counts[f"{kind}/{path.split('/')[0]}"] += 1
            counts[f"turns/{len(rec['conversations'])}"] += 1
    videos = sorted({p for k, p in rows if k == "video"})
    heldout = set(pick_heldout(videos, args.heldout_frac, args.seed))

    probe: Optional[Dict[str, Dict[str, Any]]] = None
    if args.probe:
        probe = probe_videos(media_root, videos, args.video_fps, args.video_maxlen)
        bad = {v: p["error"] for v, p in probe.items() if "error" in p}
        if bad:
            raise SystemExit(f"{len(bad)} videos cannot be read, e.g. {sorted(bad.items())[:5]}")
    too_short = sorted(v for v, p in (probe or {}).items() if not p["nfp_ok"])
    excluded = heldout | set(too_short)

    pool = [i for i, (k, p) in enumerate(rows) if k == "video" and p not in excluded]
    if args.nfp_rows > len(pool):
        raise SystemExit(f"--nfp-rows {args.nfp_rows} > {len(pool)} eligible video rows")
    rng = random.Random(f"nfp:{args.seed}")
    picked = rng.sample(pool, args.nfp_rows)
    us = [rng.random() for _ in picked]

    tmp = out.with_name(out.name + ".partial")
    if tmp.exists():
        raise SystemExit(f"{tmp} exists (an earlier build stopped): look at it and remove it yourself")
    tmp.mkdir(parents=True)
    n_qa = n_qa_held = 0
    split_qa = args.heldout_scope == "all"
    with open(src, encoding="utf-8") as f, open(tmp / "qa.jsonl", "w", encoding="utf-8") as fq, \
            open(tmp / "qa_heldout.jsonl", "w", encoding="utf-8") if split_qa else open(os.devnull, "w") as fh:
        for (kind, path), line in zip(rows, f):
            row = convert_row(json.loads(line), media_root, args.video_markers)
            if split_qa and kind == "video" and path in heldout:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_qa_held += 1
            else:
                fq.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_qa += 1
    with open(tmp / "nfp.jsonl", "w", encoding="utf-8") as fo:
        for i, u in zip(picked, us):
            fo.write(json.dumps(nfp_row(media_root, rows[i][1], u, args.video_markers), ensure_ascii=False) + "\n")
    held_sorted = sorted(heldout - set(too_short))
    with open(tmp / "nfp_heldout.jsonl", "w", encoding="utf-8") as fo:
        for v in held_sorted:
            for k in range(args.heldout_windows):
                u = (k + 0.5) / args.heldout_windows
                fo.write(json.dumps(nfp_row(media_root, v, u, args.video_markers), ensure_ascii=False) + "\n")
    (tmp / "heldout_videos.txt").write_text("".join(v + "\n" for v in sorted(heldout)))
    (tmp / "dataset_info.json").write_text(json.dumps(dataset_info(split_qa), indent=2) + "\n")

    files = sorted(p.name for p in tmp.iterdir())
    manifest = {
        "args": {k: v for k, v in vars(args).items()},
        "media_root": media_root,
        "source": {"path": str(src.resolve()), "sha256": sha256(src), "rows": len(rows)},
        "counts": dict(sorted(counts.items())),
        "distinct_videos": len(videos),
        "heldout_videos": len(heldout),
        "nfp_pool_rows": len(pool),
        "rows_out": {"qa": n_qa, "qa_heldout": n_qa_held, "nfp": len(picked),
                     "nfp_heldout": len(held_sorted) * args.heldout_windows},
        "nfp_videos_used": len({rows[i][1] for i in picked}),
        "probe": None if probe is None else {
            "videos": len(probe), "too_short_for_nfp": too_short,
            "duration_s": _quantiles([p["duration"] for p in probe.values()]),
            "average_fps": dict(Counter(round(p["average_fps"], 3) for p in probe.values()).most_common(10))},
        "outputs": {n: {"sha256": sha256(tmp / n), "lines": _lines(tmp / n)} for n in files},
    }
    (tmp / "build_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return manifest


def _lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _quantiles(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    xs = sorted(xs)
    pick = lambda q: round(xs[min(len(xs) - 1, int(q * (len(xs) - 1)))], 2)  # noqa: E731
    return {"min": round(xs[0], 2), "p10": pick(0.1), "p50": pick(0.5), "p90": pick(0.9), "max": round(xs[-1], 2)}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, help="vsi_590k.jsonl")
    ap.add_argument("--media-root", required=True, help="folder the source's video / image paths are relative to")
    ap.add_argument("--out", required=True)
    ap.add_argument("--video-markers", choices=VIDEO_MARKERS, default="outer")
    ap.add_argument("--nfp-rows", type=int, default=290_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-frac", type=float, default=0.01)
    ap.add_argument("--heldout-scope", choices=HELDOUT_SCOPES, default="nfp")
    ap.add_argument("--heldout-windows", type=int, default=4)
    ap.add_argument("--probe", action="store_true", help="open every video (needs the media on disk)")
    ap.add_argument("--video-fps", type=float, default=2.0)
    ap.add_argument("--video-maxlen", type=int, default=256)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    m = build(parse_args(argv))
    print(json.dumps({k: m[k] for k in ("rows_out", "distinct_videos", "heldout_videos", "nfp_pool_rows",
                                        "nfp_videos_used")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""Live-server template parity for TVR (CONTRACT_wave2 5.2.4; DECISIONS B5 on TVR demos).  PY_Q3.

vllm_parity(start, dry_json, base_url, out): for the dry run's first `demos` parity demos (n2_train_vpd
--dry-run output, PY_TRAIN), `per_demo` turn prefixes each (vpd.walkerbench.pilot.parity_cases / pick_turns,
imported read-only), the live server's usage.prompt_tokens of the OpenAI-style request (system / assistant as
strings, user content as text / image parts in placeholder order = vpd.train.parity_check.oai_messages; every
image a data:image/png;base64 URL of the demo's PNG file, as the eval harness sends) must equal the training
tokenization's prompt_tokens of the same prefix + generation prompt.  equal iff every case is equal and >= 2
demos are covered.  The server is the caller's (run_tvr's session); no own VLLMServer.

    PY_Q3 -B -m vpd.tvr.parity_tvr --start S --dry-run-json F --base-url URL --out F
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCHEMA = "vpd.tvr.vllm_parity/1"


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def png_data_url(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def oai_request(msgs: list, images: list, served: str) -> dict:
    """The chat request an OpenAI-style client sends for a training prefix (PNG data URLs)."""
    from vpd.train import parity_check as PC

    out, i = [], 0
    for m in PC.oai_messages(msgs):
        c = m["content"]
        if isinstance(c, list):
            parts = []
            for part in c:
                if part.get("type") == "image":
                    parts.append({"type": "image_url", "image_url": {"url": png_data_url(images[i])}})
                    i += 1
                else:
                    parts.append({"type": "text", "text": part["text"]})
            out.append({"role": m["role"], "content": parts})
        else:
            out.append({"role": m["role"], "content": c})
    if i != len(images):
        raise ValueError(f"{i} image parts for {len(images)} images")
    return {"model": served, "messages": out, "max_tokens": 1, "temperature": 0.0}


def _post(base_url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def served_name(start: str) -> str:
    from vpd import starts

    return starts.resolve(start)["code"] or start


def vllm_parity(start: str, dry_json, base_url: str, out, demos: int = 3, per_demo: int = 4, *,
                served: Optional[str] = None, post: Optional[Callable] = None) -> Dict[str, Any]:
    from vpd.train import n2_train_vpd as T
    from vpd.walkerbench import pilot

    dry = json.loads(Path(dry_json).read_text())
    if dry.get("start") != start:
        raise ValueError(f"{dry_json} is a dry run of {dry.get('start')!r}, not {start!r}")
    records = {r.get("id"): r for r in T.load_records(Path(dry["data"]))}
    cases = pilot.parity_cases(dry, records, demos, per_demo)
    served = served or served_name(start)
    post = post or _post
    rows: List[Dict[str, Any]] = []
    for c in cases:
        body = oai_request(c["msgs"], c["images"], served)
        try:
            got = (post(base_url, body).get("usage") or {}).get("prompt_tokens")
            err = None
        except Exception as e:  # noqa: BLE001  (recorded; the check fails)
            got, err = None, f"{type(e).__name__}: {e}"[:300]
        row = {k: c[k] for k in ("id", "turn", "n_images", "train_prompt_tokens")}
        row.update(server_prompt_tokens=got, equal=(got == c["train_prompt_tokens"]), error=err)
        rows.append(row)
    covered = sorted({r["id"] for r in rows})
    rep = {"schema": SCHEMA, "check": "B5 live vLLM prompt tokens vs training tokenization (PNG data URLs)",
           "start": start, "served": served, "base_url": base_url, "dry_run": str(dry_json), "demos": covered,
           "n": len(rows), "rows": rows,
           "equal": bool(rows) and len(covered) >= min(2, demos) and all(r["equal"] for r in rows), "utc": utc()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    tmp.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--dry-run-json", required=True)
    ap.add_argument("--base-url", required=True, help="http://127.0.0.1:<port>/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--served", default=None, help="served model name (default: the start's code)")
    ap.add_argument("--demos", type=int, default=3)
    ap.add_argument("--per-demo", type=int, default=4)
    a = ap.parse_args(argv)
    rep = vllm_parity(a.start, a.dry_run_json, a.base_url, a.out, a.demos, a.per_demo, served=a.served)
    print(json.dumps({"equal": rep["equal"], "n": rep["n"], "demos": rep["demos"]}))
    return 0 if rep["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())

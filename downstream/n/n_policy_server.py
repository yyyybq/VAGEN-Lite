#!/usr/bin/env python3
"""N series: one VLM behind a socket for n_episode.py (run with the StarVLA environment's Python on one GPU).

    python downstream/n/n_policy_server.py --model B --port 5800
    python downstream/n/n_policy_server.py --model B --port 5800 --adapter RUN/adapters/100/adapter.pt   (N2)

A request is length-prefixed JSON {"messages": chat-template messages (image parts without data), "images": PNG
base64 in order, "max_new_tokens": N}; the answer {"reply": text, "prompt_tokens": the prompt's length}: the model's
chat template with the generation prompt, greedy decoding.  With n_episode.py --history a request carries every earlier
turn of the episode and all of its images.  Models and processors as q0_ask.py (read by path).  One request at a time.
--adapter (N2, the user's decision 1 of 2026-11-03): a LoRA adapter of n2_train.py added to the model's weights in
memory after loading (n2_train.merge_adapter; nothing written).
The matrix v3 starts of nm_starts.json (nm_starts.py, 2026-11-03) are models too, by their names there.
"""
from __future__ import annotations

import argparse
import base64
import io
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "x"))
import n_episode as E  # noqa: E402
import nm_starts  # noqa: E402
import q0_ask  # noqa: E402

nm_starts.register(q0_ask)


def answer(processor, model, req: dict) -> tuple:
    """(the reply, the prompt's length in tokens)."""
    import torch
    from PIL import Image

    images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in req["images"]]
    text = processor.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    n_prompt = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=int(req.get("max_new_tokens", 384)), do_sample=False)
    return processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0], n_prompt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=q0_ask.MODELS)
    ap.add_argument("--port", type=int, default=5800)
    ap.add_argument("--adapter", default=None, help="N2: a LoRA adapter (n2_train.py adapter.pt) merged in memory")
    args = ap.parse_args(argv)
    t0 = time.time()
    processor, model, proc_dir = q0_ask.load(args.model)
    if args.adapter:
        import n2_train

        print(f"[n_policy] adapter {args.adapter}: {n2_train.merge_adapter(model, args.adapter)} modules merged", flush=True)
    print(f"[n_policy] {args.model} loaded in {time.time() - t0:.0f} s (processor {proc_dir})", flush=True)
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", args.port))
    srv.listen(64)
    n = 0
    while True:
        conn, _ = srv.accept()
        with conn:
            try:
                req = E.recv_msg(conn)
                reply, n_prompt = answer(processor, model, req)
                E.send_msg(conn, {"reply": reply, "prompt_tokens": n_prompt})
                n += 1
                if n % 20 == 0:
                    print(f"[n_policy] {n} answers", flush=True)
            except Exception as exc:  # one bad request does not stop the server
                try:
                    E.send_msg(conn, {"reply": "", "error": f"{type(exc).__name__}: {exc}"[:300]})
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())

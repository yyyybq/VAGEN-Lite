#!/usr/bin/env python3
"""Render QA states at their exact serialized pose using the existing renderer."""
from __future__ import annotations
import argparse, asyncio, json, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
from data_gen.active_spatial_qa.image_contract_v2 import image_stats


def _stats(img: Image.Image):
    return image_stats(img)


async def render(args):
    rows = [json.loads(x) for x in Path(args.bank).read_text().splitlines() if x.strip()]
    rows = rows[: args.limit or None]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rendered, errors, panels = [], [], []
    renderer = None
    try:
        for idx, row in enumerate(rows):
            try:
                pose = np.asarray(row["state_pose_c2w"], dtype=np.float32)
                if pose.shape != (4, 4): raise ValueError(f"invalid pose shape {pose.shape}")
                camera = row.get("camera", {})
                K = np.asarray(camera.get("intrinsics"), dtype=np.float32)
                if K.shape != (3, 3): raise ValueError("missing 3x3 camera intrinsics")
                scene = str(row["scene_id"])
                if renderer is None or renderer.cfg.scene_id != scene:
                    if renderer is not None: await renderer.close()
                    renderer = UnifiedRenderGS(render_backend=args.backend, gs_root=args.gs_root, client_url=args.renderer_url, scene_id=scene, gpu_device=args.gpu_device)
                image = await renderer.render_image_from_cam_param(K, np.linalg.inv(pose), args.width, args.height)
                if image.size != (args.width, args.height):
                    raise ValueError(f"renderer returned {image.size}, expected {(args.width, args.height)}")
                stats = _stats(image)
                name = f"{idx:05d}_{row['sample_id']}.png"; image.save(out / name)
                rec = dict(row)
                rec["public_observation"] = dict(row.get("public_observation", {}), image_path=str((out / name).resolve()))
                rec["render_status"] = "success"
                rec["observability_validity"] = "valid" if stats["nonblank"] else "invalid_low_information"
                rec["render"] = {
                    "image_path": str((out / name).resolve()), "scene_id": scene,
                    "pose_c2w": pose.tolist(), "extrinsics_w2c": np.linalg.inv(pose).tolist(),
                    "request": {"intrinsics_K": K.tolist(), "width": args.width, "height": args.height,
                                "pose_c2w": pose.tolist(), "pose_w2c": np.linalg.inv(pose).tolist()},
                    "resolution": [args.width, args.height], "rgb_stats": stats,
                    "backend": args.backend, "renderer_url": args.renderer_url,
                }
                rendered.append(rec)
                thumb = image.convert("RGB").resize((180, 140)); d = ImageDraw.Draw(thumb); d.rectangle((0,0,179,18), fill=(0,0,0)); d.text((3,3), f"{row['task_type']} {row['private_answer']}", fill=(255,255,0)); panels.append(thumb)
            except Exception as exc:
                errors.append({"sample_id": row.get("sample_id"), "error": str(exc), "traceback": traceback.format_exc()})
    finally:
        if renderer is not None: await renderer.close()
    with (out / "manifest.jsonl").open("w") as f:
        for rec in rendered: f.write(json.dumps(rec, default=str) + "\n")
    if panels:
        sheet = Image.new("RGB", (180 * min(5, len(panels)), 140 * ((len(panels)+4)//5)), "white")
        for i, panel in enumerate(panels): sheet.paste(panel, ((i%5)*180, (i//5)*140))
        sheet.save(out / "contact_sheet.png")
    (out / "errors.jsonl").write_text("\n".join(json.dumps(e) for e in errors) + ("\n" if errors else ""))
    summary = {"source_bank": str(args.bank), "candidates": len(rows), "rendered": len(rendered), "errors": len(errors), "eligible": sum(r.get("observability_validity") == "valid" for r in rendered), "backend": args.backend, "renderer_url": args.renderer_url, "resolution": [args.width,args.height]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n"); print(json.dumps(summary, indent=2)); return summary


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bank", required=True); ap.add_argument("--output-dir", required=True); ap.add_argument("--backend", choices=("http","client","local"), default="http"); ap.add_argument("--renderer-url", default=""); ap.add_argument("--gs-root", default=""); ap.add_argument("--gpu-device", type=int, default=None); ap.add_argument("--width", type=int, default=256); ap.add_argument("--height", type=int, default=256); ap.add_argument("--limit", type=int, default=0); asyncio.run(render(ap.parse_args()))


if __name__ == "__main__": main()

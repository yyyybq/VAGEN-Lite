"""vpd-tvr-debug probe entry point (EXPLORATORY; gate tvr): one subcommand per probe or bookkeeping step.

Run as `PY -I -u vpd/tvr/debug_probe.py SUBCOMMAND ...` by vpd/tvr/debug_job.sh (the job payload).  -I drops the
script's directory from sys.path, so this file puts vpd/tvr (our code only) back in front and imports the
helpers from vpd/tvr/debug/.  Probes (job only, they touch the GPU):
  gl-tri    V1  EGL + GL triangle                      (debug/gl_tri.py)
  vk-tri    V2  Vulkan 1.1 offscreen triangle          (debug/vk_tri.py; the login-node smoke uses lavapipe only)
  thor      V3-V8, V10 AI2-THOR diagnostics            (debug/thor_diag.py)
  open3d    V9  Open3D OffscreenRenderer               (debug/open3d_egl.py)
Bookkeeping (stdlib only, any interpreter, no GPU):
  facts       V0 node facts -> result.json with metrics.caps_honored
  post-thor   parse a thor variant's side outputs (LD_DEBUG, nvidia-smi samples, shader cache, loader log)
  finalize    add rc / secs / cap / timed_out (or a skip reason) to a variant's result.json
  summarize   OUT/summary.json with the decision-table rows that matched
  layers-json write the V7 implicit-layer manifest (VK_LAYER_NV_optimus only, absolute library_path)
  get         print one dotted key of a JSON file as JSON (for the shell)
Each probe writes OUT/result.json {variant, pass, stage, metrics, notes, ...}; exit 0 iff pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; keep vpd/tvr/debug free of __pycache__

from debug import common  # noqa: E402


def _outdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _saver(out: str):
    path = os.path.join(out, "result.json")

    def save(res):
        common.write_json_atomic(path, res)
    return save


def _finish(out: str, res: dict) -> int:
    common.write_json_atomic(os.path.join(out, "result.json"), res)
    print(json.dumps(res, indent=1, default=str)[:6000], flush=True)
    return 0 if res.get("pass") else 1


def cmd_gl_tri(a) -> int:
    from debug import gl_tri
    out = _outdir(a.out)
    return _finish(out, gl_tri.run(out, a.egl_lib, a.gl_lib, a.device, a.size, on_stage=_saver(out)))


def cmd_vk_tri(a) -> int:
    from debug import vk_tri
    out = _outdir(a.out)
    return _finish(out, vk_tri.run(out, a.loader, a.want, a.spv_dir or vk_tri.SHADERS, a.size,
                                   on_stage=_saver(out)))


def cmd_thor(a) -> int:
    from debug import thor_diag
    return thor_diag.main_args(a)


def cmd_open3d(a) -> int:
    from debug import open3d_egl
    out = _outdir(a.out)
    return _finish(out, open3d_egl.run(out, a.width, a.height))


def cmd_facts(a) -> int:
    from debug import facts
    out = _outdir(a.out)
    res = facts.collect(out, sys_lib=a.sys_lib, vk_dirs=a.vk_dirs.split(":"),
                        egl_vendor_dirs=a.egl_vendor_dirs.split(":"), dev_dir=a.dev_dir, driver=a.driver,
                        driver_file=a.driver_file, nvgl=a.nvgl, env_dir=a.env_dir, build=a.build,
                        lvp_icd=a.lvp_icd, o3d_py=a.o3d_py)
    _finish(out, res)
    return 0


def cmd_post_thor(a) -> int:
    from debug import post
    post.merge_into_result(a.dir)
    return 0


def cmd_finalize(a) -> int:
    from debug import summarize
    summarize.finalize(a.dir, a.name, rc=a.rc, secs=a.secs, cap=a.cap, skipped=a.skipped)
    return 0


def cmd_summarize(a) -> int:
    from debug import summarize
    s = summarize.summarize(a.out)
    print(json.dumps({k: s[k] for k in ("variants_pass", "caps_honored", "matched")}, indent=1, default=str))
    return 0


def cmd_layers_json(a) -> int:
    from debug import thor_diag
    thor_diag.write_layers_json(a.out, a.lib)
    return 0


def cmd_get(a) -> int:
    obj = common.read_json(a.file)
    for k in a.key.split("."):
        obj = obj.get(k) if isinstance(obj, dict) else None
    print(json.dumps(obj))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("gl-tri")
    p.add_argument("--out", required=True)
    p.add_argument("--egl-lib", required=True, help="absolute path of glvnd libEGL.so.1")
    p.add_argument("--gl-lib", required=True, help="absolute path of glvnd libOpenGL.so.0")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--size", type=int, default=common.SIZE)
    p.set_defaults(fn=cmd_gl_tri)

    p = sub.add_parser("vk-tri")
    p.add_argument("--out", required=True)
    p.add_argument("--loader", required=True, help="path of libvulkan.so.1 (absolute = that loader exactly)")
    p.add_argument("--want", choices=["nvidia", "cpu", "any"], default="any")
    p.add_argument("--spv-dir")
    p.add_argument("--size", type=int, default=common.SIZE)
    p.set_defaults(fn=cmd_vk_tri)

    p = sub.add_parser("thor")
    from debug import thor_diag  # stdlib-only at import time
    thor_diag.add_arguments(p)
    p.set_defaults(fn=cmd_thor)

    p = sub.add_parser("open3d")
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.set_defaults(fn=cmd_open3d)

    p = sub.add_parser("facts")
    p.add_argument("--out", required=True)
    p.add_argument("--sys-lib", default="/usr/lib/x86_64-linux-gnu")
    p.add_argument("--vk-dirs", default="/etc/vulkan:/usr/share/vulkan")
    p.add_argument("--egl-vendor-dirs", default="/usr/share/glvnd/egl_vendor.d:/etc/glvnd/egl_vendor.d")
    p.add_argument("--dev-dir", default="/dev")
    p.add_argument("--driver", default="595.71.05")
    p.add_argument("--driver-file", default="/proc/driver/nvidia/version")
    p.add_argument("--nvgl")
    p.add_argument("--env-dir")
    p.add_argument("--build")
    p.add_argument("--lvp-icd")
    p.add_argument("--o3d-py")
    p.set_defaults(fn=cmd_facts)

    p = sub.add_parser("post-thor")
    p.add_argument("--dir", required=True)
    p.set_defaults(fn=cmd_post_thor)

    p = sub.add_parser("finalize")
    p.add_argument("--dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--rc", type=int)
    p.add_argument("--secs", type=int)
    p.add_argument("--cap", type=int)
    p.add_argument("--skipped")
    p.set_defaults(fn=cmd_finalize)

    p = sub.add_parser("summarize")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("layers-json")
    p.add_argument("--out", required=True)
    p.add_argument("--lib", required=True, help="absolute path of the staged libGLX_nvidia.so.0")
    p.set_defaults(fn=cmd_layers_json)

    p = sub.add_parser("get")
    p.add_argument("file")
    p.add_argument("key")
    p.set_defaults(fn=cmd_get)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if os.environ.get("VPD_TVR_PARSE_ONLY") == "1":  # tests: does the job's argv parse, and to what
        print(json.dumps({k: v for k, v in vars(a).items() if k != "fn"}, default=str))
        return 0
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

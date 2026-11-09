"""vpd-tvr-debug: probes of the TVR render debug job (EXPLORATORY; gate tvr).

Entry point: vpd/tvr/debug_probe.py (subcommands); job payload: vpd/tvr/debug_job.sh.  Modules:
  common     result.json helpers, pixel checks, /proc/self/maps
  facts      V0 node facts (caps_honored)
  gl_tri     V1 EGL + GL 3.3 triangle (ctypes)
  vk_tri     V2 Vulkan 1.1 offscreen triangle (ctypes, SPIR-V from shaders/)
  thor_diag  V3-V8, V10 AI2-THOR CloudRendering diagnostics (RGB / depth / segmentation statistics)
  post       parsers of the per-variant side outputs (LD_DEBUG, nvidia-smi samples, shader cache, loader log)
  open3d_egl V9 Open3D OffscreenRenderer positive control
  summarize  OUT/summary.json with the decision-table rows that matched
Every module imports only the standard library at import time (numpy / PIL / ai2thor / open3d lazily).
"""

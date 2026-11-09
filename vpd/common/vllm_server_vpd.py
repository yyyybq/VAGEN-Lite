"""A vLLM OpenAI-compatible server on one model, for the VPD closed-loop evals (inside a one-card sco job).

Copied from eval_external/closed_loop/vllm_server.py @ bf4b2553 (that file is not edited).  Changes:
1. model_dir: serve a local folder (merged LoRA weights on job-local disk) instead of the code's eval view; the
   served model name stays the code, so the agent and the manifest name the start.
2. extra_args: appended to the command; default ["--structured-outputs-config", '{"backend": "xgrammar"}'] so that
   per-request extra_body={"structured_outputs": {"regex": R}} is decoded by xgrammar (vLLM 0.11 field names,
   checked in vllm/entrypoints/openai/protocol.py and vllm/sampling_params.py StructuredOutputsParams).
3. Defaults: max_images 80 (the longest s2test episode needs 66 images), max_model_len 128000,
   gpu_memory_utilization 0.85 (pass 0.80 for E3VS), and the pixel cap of DECISIONS B16 as --mm-processor-kwargs
   {"max_pixels": 1024000, "size": {"longest_edge": 1024000, "shortest_edge": <the model's own>}}.
   vLLM 0.11 builds Qwen3VLProcessor.from_pretrained(model, **mm_processor_kwargs) and sizes images (and its
   start-up profiling) from image_processor.size["longest_edge"] (vllm/model_executor/models/qwen3_vl.py
   _get_vision_info).  With transformers 4.57.6 (eval-external-q3) max_pixels alone only sets an unused attribute
   (size.longest_edge stays 16,777,216; a 2000x1200 image keeps grid [1, 76, 124]); "size" does cap it
   ([1, 48, 80]) and leaves 640x400 ([1, 24, 40]) and 1280x800 ([1, 50, 80]) unchanged (login-node probe,
   2026-11-08).  max_pixels is kept so the flag still states the B16 value; "size" is what takes effect, the same
   field vpd.starts.set_max_pixels sets in the training processor.  mm_processor_kwargs="auto" (default) builds
   this; pass a dict to override or None to send no flag.
4. port=None picks a free local port.
5. .cfg, .cmd and .manifest() expose the final settings for the run manifest.
6. env(): the original tweaks (conda gcc for Triton, VLLM_USE_FLASHINFER_SAMPLER=0, NO_PROXY) plus
   OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS = paths.VLLM_THREADS (4) for the vLLM process (DECISIONS B9).

    with VLLMServer("c0q3_sft_e8", log=run_dir / "vllm.log", model_dir="/tmp/vpd_merge/Q3S_200") as srv:
        srv.base_url   # http://127.0.0.1:<port>/v1, served model name = the code

The server ignores the model's generation_config.json (--generation-config vllm): every request states its own
sampling parameters, so all models decode the same way.  Nothing here runs on the login node (GPU only).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from vpd import paths

SOURCE = "eval_external/closed_loop/vllm_server.py@bf4b2553"
# enforce_eager: no CUDA graphs (the original's note: vLLM 0.11 hit an illegal memory access with them on
# multi-image chats).
SERVER_DEFAULTS = {"max_model_len": 128000, "max_images": 80, "gpu_memory_utilization": 0.85, "seed": 0,
                   "enforce_eager": True, "mm_processor_kwargs": "auto"}
DEFAULT_EXTRA_ARGS = ("--structured-outputs-config", json.dumps({"backend": "xgrammar"}))
MODEL_DIR_REQUIRED = ("config.json", "tokenizer_config.json", "preprocessor_config.json")
READY_TIMEOUT_S = 1200
QWEN3_SHORTEST_EDGE = 65536  # Qwen3-VL-8B preprocessor_config.json size.shortest_edge (Q3B, Q3S views)


def pixel_cap_kwargs(model_path: Path | None, max_pixels: int = paths.MAX_PIXELS) -> dict:
    """The --mm-processor-kwargs that cap images at max_pixels in vLLM 0.11 + transformers 4.57.6 (docstring 3).
    shortest_edge is the model's own (from its preprocessor_config.json), so only the upper cap changes."""
    shortest = QWEN3_SHORTEST_EDGE
    try:
        cfg = json.loads((Path(model_path) / "preprocessor_config.json").read_text())
        shortest = int((cfg.get("size") or {}).get("shortest_edge") or cfg.get("min_pixels") or shortest)
    except (OSError, TypeError, ValueError):
        pass
    return {"max_pixels": int(max_pixels), "size": {"longest_edge": int(max_pixels), "shortest_edge": shortest}}


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def view_dir(code: str) -> Path:
    """The code's eval view (eval_external.adapters.models.view_dir), imported lazily so that the vpd package
    does not need the registry to build commands in tests."""
    from eval_external.adapters import models as M

    return Path(M.view_dir(code))


class VLLMServer:
    def __init__(self, code: str, log: Path | str, port: int | None = None, model_dir: Path | str | None = None,
                 extra_args: list[str] | tuple[str, ...] | None = None, **overrides):
        unknown = set(overrides) - set(SERVER_DEFAULTS)
        if unknown:
            raise TypeError(f"unknown server settings {sorted(unknown)}; known: {sorted(SERVER_DEFAULTS)}")
        self.code = code
        self.port = int(port) if port else free_port()
        self.log = Path(log)
        self.model_dir = Path(model_dir) if model_dir is not None else None
        if self.model_dir is not None:
            missing = [f for f in MODEL_DIR_REQUIRED if not os.path.lexists(self.model_dir / f)]
            if missing:
                raise FileNotFoundError(f"model_dir {self.model_dir} lacks {missing} (merge.py copies the processor "
                                        f"and tokenizer files next to the weights)")
        self.extra_args = list(DEFAULT_EXTRA_ARGS if extra_args is None else extra_args)
        self.cfg = {**SERVER_DEFAULTS, **overrides}
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def model_path(self) -> Path:
        return self.model_dir if self.model_dir is not None else view_dir(self.code)

    def command(self) -> list[str]:
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
               "--model", str(self.model_path), "--served-model-name", self.code,
               "--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
               "--max-model-len", str(self.cfg["max_model_len"]),
               "--limit-mm-per-prompt", json.dumps({"image": self.cfg["max_images"]}),
               "--gpu-memory-utilization", str(self.cfg["gpu_memory_utilization"]),
               "--seed", str(self.cfg["seed"]), "--generation-config", "vllm"]
        if self.cfg["enforce_eager"]:
            cmd.append("--enforce-eager")
        mm = self.mm_processor_kwargs()
        if mm:
            cmd += ["--mm-processor-kwargs", json.dumps(mm)]
        return cmd + self.extra_args

    def mm_processor_kwargs(self) -> dict | None:
        mm = self.cfg["mm_processor_kwargs"]
        return pixel_cap_kwargs(self.model_path) if mm == "auto" else mm

    @property
    def cmd(self) -> list[str]:
        return self.command()

    def manifest(self) -> dict:
        """What the run manifest records about the server."""
        return {"source": SOURCE, "code": self.code, "model_dir": None if self.model_dir is None else str(self.model_dir),
                "model_path": str(self.model_path), "cfg": dict(self.cfg),
                "mm_processor_kwargs": self.mm_processor_kwargs(), "extra_args": list(self.extra_args),
                "cmd": self.cmd, "port": self.port, "threads": paths.VLLM_THREADS}

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @staticmethod
    def env() -> dict:
        """Triton builds a small C helper at start-up; the sco image has no system compiler, so use the env's conda
        gcc, as examples/train/active_spatial/sco_interiorgs_v10_train.sh does."""
        env = dict(os.environ)
        prefix = Path(sys.executable).resolve().parent.parent
        for var, tool in (("CC", "x86_64-conda-linux-gnu-gcc"), ("CXX", "x86_64-conda-linux-gnu-g++")):
            if not env.get(var) and (prefix / "bin" / tool).exists():
                env[var] = str(prefix / "bin" / tool)
        env.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")
        # FlashInfer's sampler is compiled at start-up and fails without the CUDA header set-up of the training
        # scripts; with greedy decoding the sampler only takes the argmax, so PyTorch's own sampler is used.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        # vLLM 0.11 warms up DeepGEMM at start-up when the package is importable; inside sco jobs its init asserts
        # because no CUDA toolkit is found (no nvcc / CUDA_HOME / /usr/local/cuda), and every engine start failed
        # (wb-smoke pt-v9mmly5e, 2026-11-09). DeepGEMM serves only block-FP8 linears; our bf16 models never use it.
        env["VLLM_USE_DEEP_GEMM"] = "0"
        env.setdefault("NO_PROXY", "*")
        env.setdefault("no_proxy", "*")
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(paths.VLLM_THREADS)  # DECISIONS B9: vLLM processes get 4
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    def _ready(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/models", timeout=5) as r:
                return self.code in r.read().decode()
        except Exception:
            return False

    def __enter__(self) -> "VLLMServer":
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log, "w")
        self._log_fh.write("[vllm_server_vpd] " + json.dumps(self.cmd) + "\n")
        self._log_fh.flush()
        self.started = time.time()
        self.proc = subprocess.Popen(self.command(), stdout=self._log_fh, stderr=subprocess.STDOUT, env=self.env())
        deadline = time.time() + READY_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}; see {self.log}")
            if self._ready():
                self.ready_seconds = round(time.time() - self.started, 1)
                return self
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"vLLM not ready after {READY_TIMEOUT_S // 60} minutes; see {self.log}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log_fh", None):
            self._log_fh.close()

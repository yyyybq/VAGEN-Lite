"""A vLLM OpenAI-compatible server on one model view, for the closed-loop runs (inside a one-card sco job).

    with VLLMServer("base-7B", log=run_dir / "vllm.log") as srv:
        srv.base_url   # http://127.0.0.1:8001/v1, served model name = the model code

The server ignores the model's generation_config.json (--generation-config vllm): every request states its own
sampling parameters, so all models decode the same way.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from eval_external.adapters import models as M

SERVER_DEFAULTS = {"max_model_len": 128000, "max_images": 64, "gpu_memory_utilization": 0.85, "seed": 0}


class VLLMServer:
    def __init__(self, code: str, log: Path, port: int = 8001, **overrides):
        self.code = code
        self.port = port
        self.log = Path(log)
        self.cfg = {**SERVER_DEFAULTS, **overrides}
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def command(self) -> list[str]:
        return [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                "--model", str(M.view_dir(self.code)), "--served-model-name", self.code,
                "--host", "127.0.0.1", "--port", str(self.port), "--dtype", "bfloat16",
                "--max-model-len", str(self.cfg["max_model_len"]),
                "--limit-mm-per-prompt", json.dumps({"image": self.cfg["max_images"]}),
                "--gpu-memory-utilization", str(self.cfg["gpu_memory_utilization"]),
                "--seed", str(self.cfg["seed"]), "--generation-config", "vllm"]

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
        env.setdefault("NO_PROXY", "*")
        env.setdefault("no_proxy", "*")
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
        self.proc = subprocess.Popen(self.command(), stdout=self._log_fh, stderr=subprocess.STDOUT, env=self.env())
        deadline = time.time() + 1200
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited with {self.proc.returncode}; see {self.log}")
            if self._ready():
                return self
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"vLLM not ready after 20 minutes; see {self.log}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_log_fh", None):
            self._log_fh.close()

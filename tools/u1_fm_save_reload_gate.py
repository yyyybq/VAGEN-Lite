#!/usr/bin/env python3
"""Independent-process save/reload gate for the bounded U1 FM overfit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "verl") not in sys.path:
    sys.path.insert(0, str(ROOT / "verl"))

from tools.u1_fm_backprop_ab_gate import (  # noqa: E402
    DEFAULT_FINAL,
    DEFAULT_MODEL,
    DEFAULT_OUT,
    jdump,
    load_model_processor,
    load_rows,
    prepare_sample,
    seed_everything,
    select_fm_parameters,
)
from tools.u1_fm_fixed_overfit_gate import (  # noqa: E402
    decode_merged_pixels,
    fixed_prediction,
    sampled_signature,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_compare(left: Path, right: Path) -> dict[str, Any]:
    a = np.asarray(Image.open(left).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(right).convert("RGB"), dtype=np.int16)
    diff = np.abs(a - b)
    return {
        "left": str(left),
        "right": str(right),
        "shape_equal": a.shape == b.shape,
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": int(diff.max()),
        "exact": bool(np.array_equal(a, b)),
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
    }


def update_final(path: Path, report: dict[str, Any], overfit: dict[str, Any]) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["phase3_fm_fixed_transition_overfit"] = {
        "status": overfit["status"],
        "report": "../fm_only/phase3_fm_fixed_overfit_report.json",
        "initial_fm_loss": overfit.get("initial", {}).get("fm_loss_raw"),
        "final_fm_loss": overfit.get("final", {}).get("fm_loss_raw"),
        "loss_ratio": overfit.get("loss_ratio"),
        "initial_latent_rmse": overfit.get("initial", {}).get("latent_rmse"),
        "final_latent_rmse": overfit.get("final", {}).get("latent_rmse"),
        "latent_ratio": overfit.get("latent_ratio"),
        "parameter_delta_l2": overfit.get("parameter_delta_l2"),
        "checkpoint": overfit.get("checkpoint"),
    }
    data["phase3_fm_save_reload"] = {
        "status": report["status"],
        "report": "../fm_only/phase3_fm_save_reload_report.json",
        "checkpoint": report.get("checkpoint"),
        "signature_match": report.get("signature_match"),
        "formal_loss_close": report.get("formal_loss_close"),
        "latent_close": report.get("latent_close"),
        "decoded_output_exact": report.get("decoded_output", {}).get("exact"),
    }
    upstream = (
        data.get("phase3_fm_data_gate", {}).get("status") == "PASS"
        and data.get("phase3_fm_forward_only", {}).get("status") == "PASS"
        and data.get("phase3_fm_backprop_ab", {}).get("status") == "PASS"
        and data.get("phase3_fsdp_empty_storage", {}).get("status") == "PASS"
        and overfit["status"] == "PASS"
        and report["status"] == "PASS"
    )
    data["phase3_fm_only"] = {
        "status": "PASS" if upstream else "FAIL",
        "report": "../fm_only/phase3_fm_save_reload_report.json",
        "gates": {
            "data": data.get("phase3_fm_data_gate", {}).get("status"),
            "forward_only": data.get("phase3_fm_forward_only", {}).get("status"),
            "backprop_ab": data.get("phase3_fm_backprop_ab", {}).get("status"),
            "fsdp_empty_storage": data.get("phase3_fsdp_empty_storage", {}).get("status"),
            "overfit_visualization": overfit["status"],
            "save_reload": report["status"],
        },
    }
    data["overall_status"] = "NOT_READY_FOR_TRAINING"
    data["next_gate"] = "phase4_protocol_live_test" if upstream else "blocked_on_phase3_fm_save_reload"
    jdump(data, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--final-delivery", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    report_path = args.out / "phase3_fm_save_reload_report.json"
    overfit_path = args.out / "phase3_fm_fixed_overfit_report.json"
    checkpoint = args.checkpoint or (args.out / "fm_only_overfit_trainable_state.pt")
    started = time.time()
    report: dict[str, Any] = {
        "phase": "phase3_fm_save_reload",
        "status": "RUNNING",
        "scope": "independent Python process; same fixed transition/seed/config; no training/rollout/Phase4",
        "model": str(args.model),
        "checkpoint": str(checkpoint),
    }
    jdump(report, report_path)

    try:
        overfit = json.loads(overfit_path.read_text(encoding="utf-8"))
        loss_ratio = float(overfit["loss_ratio"])
        latent_ratio = float(overfit["latent_ratio"])
        overfit_pass = (
            math.isfinite(float(overfit["final"]["fm_loss_raw"]))
            and loss_ratio <= 0.7
            and latent_ratio <= 0.9
            and float(overfit["parameter_delta_l2"]) > 0.0
            and int(overfit["skipped_steps"]) == 0
            and not bool(overfit["amp"]["overflow"])
            and checkpoint.is_file()
            and checkpoint.stat().st_size > 0
        )
        overfit["status"] = "PASS" if overfit_pass else "FAIL"
        overfit["pass_criteria"] = {
            "fm_loss_ratio_max": 0.7,
            "latent_rmse_ratio_max": 0.9,
            "rationale": "FM loss must drop substantially; target/prediction latent distance must also clearly decrease.",
        }
        overfit["next_gate"] = "phase3_fm_save_reload" if overfit_pass else "blocked_on_phase3_fm_overfit"
        jdump(overfit, overfit_path)
        if not overfit_pass:
            raise RuntimeError("saved overfit evidence does not satisfy the corrected explicit criteria")

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_state: dict[str, torch.Tensor] = payload["state_dict"]
        metadata = payload["metadata"]
        seed = int(metadata["seed"])
        sample_index = int(metadata["sample_index"])

        seed_everything(seed)
        torch.cuda.reset_peak_memory_stats()
        model, processor, dtype = load_model_processor(args.model, args.device)
        model.train()
        for param in model.parameters():
            param.requires_grad_(False)
        named = select_fm_parameters(model)
        named_map = dict(named)
        missing = sorted(set(named_map) - set(saved_state))
        extra = sorted(set(saved_state) - set(named_map))
        if missing or extra:
            raise RuntimeError(f"FM state key mismatch: missing={missing[:5]} extra={extra[:5]}")
        with torch.no_grad():
            for name, param in named:
                source = saved_state[name]
                if source.shape != param.shape:
                    raise RuntimeError(f"shape mismatch for {name}: {source.shape} != {param.shape}")
                param.copy_(source.to(device=param.device, dtype=param.dtype))
        signature = sampled_signature(named)
        signature_match = signature == metadata["signature"]

        row = load_rows(args.out / "fixed_transitions.jsonl")[sample_index]
        sample = prepare_sample(processor, row, args.device, dtype)
        os.environ["U1_FM_USE_UND_KV"] = "0"
        os.environ["U1_FM_BACKPROP"] = "1"
        seed_everything(seed)
        with torch.no_grad():
            forward_output = model(**sample, labels=None, return_dict=True, use_cache=False)
            pending_created = bool(getattr(model, "_u1_pending_fm", None))
            formal_loss = model.compute_pending_fm_aux_loss()
        if formal_loss is None:
            raise RuntimeError("reload formal pending loss is missing")

        with torch.no_grad():
            diag = fixed_prediction(model, sample, seed)
        reload_prediction = args.out / "visualizations" / "reload_prediction.png"
        reload_meta = decode_merged_pixels(diag["pred_x"], diag["token_hw"], reload_prediction)
        expected_prediction = Path(overfit["final"]["prediction"]["path"])
        decoded = image_compare(expected_prediction, reload_prediction)

        expected_loss = float(overfit["final"]["fm_loss_raw"])
        expected_latent = float(overfit["final"]["latent_rmse"])
        formal_value = float(formal_loss.detach().float().item())
        diagnostic_value = float(diag["fm_loss"].item())
        latent_value = float(diag["latent_rmse"].item())
        formal_close = abs(formal_value - expected_loss) <= 1e-6
        diagnostic_close = abs(diagnostic_value - expected_loss) <= 1e-6
        latent_close = abs(latent_value - expected_latent) <= 1e-6
        passed = (
            signature_match
            and pending_created
            and forward_output.loss is None
            and formal_close
            and diagnostic_close
            and latent_close
            and decoded["exact"]
            and signature["finite"]
        )
        report.update(
            {
                "status": "PASS" if passed else "FAIL",
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_metadata": metadata,
                "reloaded_signature": signature,
                "signature_match": signature_match,
                "pending_created": pending_created,
                "pending_cleared": not bool(getattr(model, "_u1_pending_fm", None)),
                "forward_loss_is_none": forward_output.loss is None,
                "expected_final_loss": expected_loss,
                "reloaded_formal_loss": formal_value,
                "reloaded_diagnostic_loss": diagnostic_value,
                "formal_loss_close": formal_close,
                "diagnostic_loss_close": diagnostic_close,
                "expected_latent_rmse": expected_latent,
                "reloaded_latent_rmse": latent_value,
                "latent_close": latent_close,
                "decoded_output": decoded,
                "reload_prediction": reload_meta,
                "gpu_memory": {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "peak_allocated": torch.cuda.max_memory_allocated(),
                    "peak_reserved": torch.cuda.max_memory_reserved(),
                },
                "elapsed_seconds": time.time() - started,
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "phase4_protocol_live_test" if passed else "blocked_on_phase3_fm_save_reload",
            }
        )
    except Exception as exc:
        overfit = json.loads(overfit_path.read_text(encoding="utf-8")) if overfit_path.exists() else {"status": "FAIL"}
        report.update(
            {
                "status": "FAIL",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.time() - started,
                "overall_status": "NOT_READY_FOR_TRAINING",
                "next_gate": "blocked_on_phase3_fm_save_reload",
            }
        )

    jdump(report, report_path)
    update_final(args.final_delivery, report, overfit)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

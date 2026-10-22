"""A strong-model run must not silently become L0: server preflight, launcher env check, aggregate flag,
and the provider probe. All MOCK (scripted providers, synthetic files); no endpoint is contacted."""

import json
import os

import pytest

from conftest import ROOT
from agent.config import load_config
from agent.server import main as server_main, resolve_providers
from eval import aggregate as agg
from eval.launch import missing_provider_env
from providers import build_providers
from providers.probe import probe

ASTRA_ENV = ("ASTRA_BASE_URL", "ASTRA_API_KEY", "ASTRA_MODEL")
S7 = os.path.join(ROOT, "configs", "S7_min_loop.yaml")


def _desc(provider="astra"):
    return {"plan": {"enabled": True, "uses_provider": True, "provider": provider},
            "check": {"enabled": False, "uses_provider": True, "provider": "other"},
            "goal": {"enabled": True, "uses_provider": False, "provider": None}}


def test_resolve_providers_flags_missing_env(monkeypatch):
    for k in ASTRA_ENV:
        monkeypatch.delenv(k, raising=False)
    provs = build_providers({"astra": {"kind": "openai_compat", "base_url_env": "ASTRA_BASE_URL",
                                       "api_key_env": "ASTRA_API_KEY", "model_env": "ASTRA_MODEL"}})
    out = resolve_providers(provs, _desc())
    assert set(out) == {"astra"}  # only providers of ENABLED modules
    assert "missing" in out["astra"]["unavailable"]


def test_resolve_providers_records_model_and_host_never_key(monkeypatch):
    monkeypatch.setenv("ASTRA_BASE_URL", "https://llm.example.org/v1")
    monkeypatch.setenv("ASTRA_API_KEY", "sk-secret-123456789")
    monkeypatch.setenv("ASTRA_MODEL", "some-model-id")
    provs = build_providers({"astra": {"kind": "openai_compat", "base_url_env": "ASTRA_BASE_URL",
                                       "api_key_env": "ASTRA_API_KEY", "model_env": "ASTRA_MODEL"}})
    out = resolve_providers(provs, _desc())
    assert out["astra"]["host"] == "llm.example.org" and out["astra"]["model"] == "some-model-id"
    assert "unavailable" not in out["astra"] and "sk-secret" not in json.dumps(out)


def test_resolve_providers_undefined_name():
    assert "not defined" in resolve_providers({}, _desc("nope"))["nope"]["unavailable"]


def test_server_refuses_to_start_without_provider(monkeypatch, tmp_path):
    for k in ASTRA_ENV:
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(SystemExit) as e:
        server_main(["--config", S7, "-o", f"run.local_log_root={tmp_path}", "-o", "run.log_root=null",
                     "--run-id", "preflight-test", "--max-steps", "100"])
    assert "astra" in str(e.value) and "L0" in str(e.value)
    assert not os.path.exists(tmp_path / "preflight-test" / "run_manifest.json")


def test_launch_missing_provider_env():
    cfg = load_config(S7)
    assert missing_provider_env(cfg, {}) == {"astra": ["ASTRA_BASE_URL", "ASTRA_MODEL", "ASTRA_API_KEY"]}
    assert missing_provider_env(cfg, {k: "x" for k in ASTRA_ENV}) == {}
    l0 = load_config(os.path.join(ROOT, "configs", "L0_exec_only.yaml"))
    assert missing_provider_env(l0, {}) == {}  # no enabled provider-backed module


def test_aggregate_flags_all_failed_provider_calls(tmp_path):
    from test_eval_analysis_aggregate import make_run

    run = make_run(str(tmp_path / "runs"), "L1_plan", "turning_on_radio", {301: 0.5, 302: 0.0},
                   summary_extra={301: {"provider_calls": 4, "provider_errors": 4}})
    res = agg.aggregate([run], None, n_boot=0)
    rows = {r["instance_id"]: r for r in res["rows"]}
    assert rows[301].get("all_provider_calls_failed") is True and not rows[302].get("all_provider_calls_failed")
    assert any("effectively L0" in w for w in res["warnings"])


def test_probe_records_stop_reason_usage_and_realistic_latency():
    replies = {"probe.text": [{"x": "ready"}], "probe.json": [{"answer": "five", "confidence": 1.0}],
               "probe.vision": [{"red_side": "left"}], "probe.latency": [{"n": 0}],
               "probe.realistic": [{"status": "in_progress", "confidence": 0.5, "evidence": "e"}]}
    p = build_providers({"mock": {"kind": "scripted", "replies": replies}})["mock"]
    out = probe(p, n_latency=3)
    assert out["max_tokens"] == 256 and out["realistic_max_tokens"] == 1024
    assert out["checks"]["json_schema"]["ok"] and out["checks"]["vision"]["ok"]
    assert out["checks"]["latency_realistic"]["n"] == 3 and out["checks"]["latency_realistic"]["errors"] == 0
    assert "stop_reason" in out["checks"]["text"] and "usage" in out["checks"]["vision"]
    assert out["models_reported"] == ["scripted"]
    realistic = [r for r in p.inner.requests if r.purpose == "probe.realistic"]
    assert len(realistic) == 3 and all(r.images and r.max_tokens == 1024 for r in realistic)

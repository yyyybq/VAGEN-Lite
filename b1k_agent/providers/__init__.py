"""Strong-model providers (configurable; keys only from environment variables).

    providers:
      astra:   {kind: openai_compat, base_url_env: ASTRA_BASE_URL, api_key_env: ASTRA_API_KEY, model_env: ASTRA_MODEL}
      claude:  {kind: anthropic, model: claude-opus-5-5, effort: low}     # substitute; label results
      mock:    {kind: scripted, replies: {...}}                            # tests only

``build_providers`` never contacts an endpoint; misconfiguration surfaces on the first call as an
error response (modules then take their documented fallback), and providers/probe.py checks an
endpoint's real capabilities.
"""

from __future__ import annotations

from typing import Any, Mapping

from providers.base import LoggedProvider, Provider, ProviderRequest, ProviderResponse, ProviderUnavailable


def _make(name: str, cfg: Mapping[str, Any]):
    kind = str(cfg.get("kind", ""))
    if kind == "openai_compat":
        from providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(name, cfg)
    if kind == "anthropic":
        from providers.anthropic_sdk import AnthropicProvider

        return AnthropicProvider(name, cfg)
    if kind == "replay":
        from providers.replay import ReplayProvider

        return ReplayProvider(name, cfg)
    if kind == "scripted":
        from providers.replay import ScriptedProvider

        return ScriptedProvider(name, cfg)
    raise ValueError(f"provider '{name}': unknown kind {kind!r}")


def build_providers(cfg: Mapping[str, Any], recorder: Any = None) -> dict[str, LoggedProvider]:
    out = {}
    for name, pcfg in (cfg or {}).items():
        pcfg = dict(pcfg or {})
        out[name] = LoggedProvider(
            _make(name, pcfg),
            price_in_per_mtok=pcfg.get("price_in_per_mtok", 0.0),
            price_out_per_mtok=pcfg.get("price_out_per_mtok", 0.0),
            recorder=recorder,
            label=pcfg.get("label", name),
        )
    return out


__all__ = ["build_providers", "Provider", "ProviderRequest", "ProviderResponse", "ProviderUnavailable", "LoggedProvider"]

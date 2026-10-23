"""Builds module objects from config. The loop gets a ModuleSet and never imports a module package.

Config block per slot (configs/*.yaml):

    modules:
      plan:
        enabled: true
        impl: llm            # an IMPLS key of modules/plan, or "<package>:<impl>" (e.g. "distill:small_vlm")
        provider: astra      # name in the top-level providers: block (only for provider-backed impls)
        prompt_version: v1
        params: {...}

A disabled slot (the default) gets the slot's "null" implementation, so the loop runs unchanged.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, Callable, Mapping, Optional

from agent.interfaces import SLOT_PROTOCOLS
from agent.prompts import PromptLoader


@dataclasses.dataclass(frozen=True)
class ImplSpec:
    slot: str
    factory: Callable[["ModuleDeps"], Any]
    uses_provider: bool = False
    description: str = ""


@dataclasses.dataclass
class ModuleDeps:
    slot: str
    impl: str
    params: Mapping[str, Any]
    provider: Any = None  # providers.base.Provider or None
    prompts: Optional[PromptLoader] = None
    prompt_version: Optional[str] = None
    recorder: Any = None  # agent.recorder.RunRecorder or None


@dataclasses.dataclass
class ModuleEntry:
    slot: str
    enabled: bool
    package: str
    impl: str
    obj: Any
    uses_provider: bool
    provider_name: Optional[str]
    prompt_version: Optional[str]
    params: Mapping[str, Any]
    prompts: Optional[PromptLoader]


class ModuleSet:
    def __init__(self, entries: Mapping[str, ModuleEntry]):
        missing = set(SLOT_PROTOCOLS) - set(entries)
        if missing:
            raise ValueError(f"ModuleSet missing slots {sorted(missing)}")
        self._entries = dict(entries)

    def get(self, slot: str) -> Any:
        return self._entries[slot].obj

    def enabled(self, slot: str) -> bool:
        return self._entries[slot].enabled

    def uses_provider(self, slot: str) -> bool:
        return self._entries[slot].uses_provider

    def slots(self):
        return tuple(self._entries)

    def describe(self) -> dict:
        out = {}
        for slot, e in self._entries.items():
            out[slot] = {
                "enabled": e.enabled,
                "impl": f"{e.package}:{e.impl}",
                "uses_provider": e.uses_provider,
                "provider": e.provider_name,
                "prompt_version": e.prompt_version,
                "prompt_files": dict(e.prompts.loaded) if e.prompts else {},
            }
        return out


def _resolve(slot: str, impl: str) -> tuple[str, str]:
    if ":" in impl:
        package, name = impl.split(":", 1)
    else:
        package, name = slot, impl
    return package, name


def load_impl_spec(package: str, name: str) -> ImplSpec:
    mod = importlib.import_module(f"modules.{package}")
    impls = getattr(mod, "IMPLS", None)
    if not isinstance(impls, dict) or name not in impls:
        raise KeyError(f"modules.{package} has no impl '{name}' (available: {sorted(impls or {})})")
    return impls[name]


def build_modules(modules_cfg: Mapping[str, Any], providers: Mapping[str, Any], recorder: Any = None) -> ModuleSet:
    entries = {}
    unknown = set(modules_cfg or {}) - set(SLOT_PROTOCOLS)
    if unknown:
        raise ValueError(f"unknown module slots in config: {sorted(unknown)}")
    for slot, protocol in SLOT_PROTOCOLS.items():
        mcfg = dict((modules_cfg or {}).get(slot) or {})
        enabled = bool(mcfg.get("enabled", False))
        impl = str(mcfg.get("impl", "null")) if enabled else "null"
        package, name = _resolve(slot, impl)
        if not enabled:
            package = slot
        spec = load_impl_spec(package, name)
        if spec.slot != slot:
            raise ValueError(f"impl {package}:{name} is for slot '{spec.slot}', configured under '{slot}'")
        provider_name = mcfg.get("provider") if spec.uses_provider else None
        provider = None
        if spec.uses_provider:
            if not provider_name or provider_name not in providers:
                raise ValueError(f"slot '{slot}' impl '{impl}' needs a provider; got {provider_name!r}")
            provider = providers[provider_name]
        prompt_version = mcfg.get("prompt_version")
        prompts = PromptLoader(package) if prompt_version else None
        deps = ModuleDeps(
            slot=slot,
            impl=name,
            params=dict(mcfg.get("params") or {}),
            provider=provider,
            prompts=prompts,
            prompt_version=prompt_version,
            recorder=recorder,
        )
        obj = spec.factory(deps)
        if not isinstance(obj, protocol):
            raise TypeError(f"{package}:{name} does not implement {protocol.__name__}")
        entries[slot] = ModuleEntry(
            slot=slot,
            enabled=enabled,
            package=package,
            impl=name,
            obj=obj,
            uses_provider=spec.uses_provider,
            provider_name=provider_name,
            prompt_version=prompt_version,
            params=deps.params,
            prompts=prompts,
        )
    return ModuleSet(entries)

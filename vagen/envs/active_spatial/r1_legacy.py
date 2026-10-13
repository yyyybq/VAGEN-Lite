"""R1 runs keep R1's success rule.

Since 2026-10-13 the env judges success with the v10 rules (target_visibility.py) unless
``r1_legacy_success`` is set.  Every R1 training, evaluation and diagnostic entry point sets it (decision of
2026-10-14), so R1 results stay comparable with earlier ones.  R1's frozen packages are hash-protected, so
the runners write legacy copies of their env YAMLs next to the run instead of editing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

R1_ENV_OVERRIDES: Dict[str, Any] = {"r1_legacy_success": True}


def env_entries(doc: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The env config dicts of an env YAML: ``{envs: [{config: ...}]}`` or ``{env1: {env_config: ...}}``."""
    if isinstance(doc.get("envs"), list):
        return [e.setdefault("config", {}) for e in doc["envs"]]
    return [v.setdefault("env_config", {}) for v in doc.values() if isinstance(v, dict) and "env_config" in v]


def legacy_env_yaml(src: Any, out_dir: Any) -> Path:
    """A copy of the env YAML ``src`` in ``out_dir`` with ``r1_legacy_success: true`` in every env entry
    (OmegaConf interpolations such as ``${oc.env:R1_RENDER_URL}`` stay as written)."""
    src, out_dir = Path(src), Path(out_dir)
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    entries = env_entries(doc)
    if not entries:
        raise ValueError(f"{src}: no env entry to set r1_legacy_success on")
    for entry in entries:
        entry.update(R1_ENV_OVERRIDES)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{src.stem}.r1_legacy{src.suffix or '.yaml'}"
    out.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out


def force_r1_legacy(cfg: Any, out_dir: Any) -> List[str]:
    """Point ``cfg.data.train_files`` / ``val_files`` (a path or a list of paths) at legacy copies.
    Returns the paths written.  ``cfg`` is an OmegaConf config opened for writing."""
    written: List[str] = []
    for key in ("train_files", "val_files"):
        value = cfg.data.get(key)
        if value is None:
            continue
        paths = [value] if isinstance(value, str) else list(value)
        new = [str(legacy_env_yaml(p, out_dir)) for p in paths]
        written.extend(new)
        cfg.data[key] = new[0] if isinstance(value, str) else new
    return written

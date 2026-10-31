"""Per-module, per-version prompt files (red line 5: modules never share prompt text).

Layout: configs/prompts/<module>/<version>.txt. A file is split into named sections by lines of the
form "### SECTION_NAME". A loader is bound to one module and cannot read another module's folder.
Every load is recorded (path + sha256) so the run manifest can pin exact prompt versions.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Dict

_SECTION = re.compile(r"^###\s+([A-Za-z0-9_]+)\s*$")

DEFAULT_PROMPT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "prompts")


class PromptLoader:
    def __init__(self, module: str, root: str = DEFAULT_PROMPT_ROOT):
        if not re.fullmatch(r"[a-z_]+", module):
            raise ValueError(f"bad module name {module!r}")
        self.module = module
        self.root = os.path.abspath(root)
        self.loaded: Dict[str, str] = {}  # relative path -> sha256

    def path(self, version: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", version) or ".." in version:
            raise ValueError(f"bad prompt version {version!r}")
        p = os.path.abspath(os.path.join(self.root, self.module, f"{version}.txt"))
        if not p.startswith(os.path.join(self.root, self.module) + os.sep):
            raise ValueError("prompt path escapes the module folder")
        return p

    def load(self, version: str) -> Dict[str, str]:
        p = self.path(version)
        with open(p, "r", encoding="utf-8") as f:
            text = f.read()
        self.loaded[os.path.relpath(p, self.root)] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return split_sections(text)


def split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, list] = {}
    current = "BODY"
    for line in text.splitlines():
        m = _SECTION.match(line)
        if m:
            current = m.group(1).upper()
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def render(template: str, **values: str) -> str:
    """Minimal {name} substitution that leaves unknown braces (e.g. JSON examples) untouched."""

    def sub(m: re.Match) -> str:
        key = m.group(1)
        return str(values[key]) if key in values else m.group(0)

    return re.sub(r"\{([a-z_][a-z0-9_]*)\}", sub, template)

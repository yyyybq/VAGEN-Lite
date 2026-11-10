"""Verbatim copies of the TVRBench text objects the py3.12 side needs without importing ai2thor (CONTRACT_wave2 5.1.1).

TVRBench (REPO, commit ad296bb7) defines the system prompt and the per-turn user text in
tvrbench/online_rl/thor_view_env.py (which imports ai2thor at module level) and the official lenient action parser
in scripts/eval_va.py (which imports thor_view_env).  The eval harness, the demo builder, kin_env and the blind arm
need these objects in interpreters without ai2thor, so they are copied here with byte-identical source semantics:

  SYSTEM_PROMPT            thor_view_env.py:51-74 (942 chars)
  IMAGE_PH                 thor_view_env.py:42
  build_concat_user_text   thor_view_env.py:126-141 (copied with its docstring and annotations)
  _ACTION_RE, _VALID_ACTIONS, parse_action   eval_va.py:103-124 (the secondary arm's parser)
  ACTION_NAMES, HORIZON_MIN, HORIZON_MAX     thor_env.py:83-97 (ACTION_NAMES = pilot_cfg.ACTIONS)

source_check(repo) proves the copies: it parses the pinned files with `ast` (no import, so no ai2thor), compares
ast.dump of the extracted nodes with ast.dump of the same nodes of THIS file, execs the extracted parse_action in a
fresh namespace and compares it with ours on a fixed corpus, and checks the pinned files' sha256 against pilot_cfg.

REGEX / regex_action: the primary (constrained) arm's format (pilot_cfg.REGEX; fullmatch, group 1).

Python 3.10 and 3.12, stdlib only (CONTRACT R3).  CLI (prints the detail JSON, exit 0 iff every check is true):
    python -B -m vpd.tvr.tvr_text check [--repo REPO]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

from vpd.tvr import lvp, pilot_cfg

IMAGE_PH = "<image>"

ACTION_NAMES = pilot_cfg.ACTIONS
HORIZON_MIN = -30.0
HORIZON_MAX = 30.0

REGEX = pilot_cfg.REGEX
_REGEX_C = re.compile(REGEX)

# --- thor_view_env.py:51-74 (verbatim) ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a navigation agent in an indoor environment. "
    "Your task is to navigate and adjust your viewpoint to PRECISELY match a target image. "
    "You must match the exact position, orientation, and camera angle "
    "— the goal is for your observation to look identical to the target.\n\n"
    "Available actions:\n"
    "- MoveAhead: Move forward 0.25m\n"
    "- MoveBack: Move backward 0.25m\n"
    "- MoveLeft: Move left 0.25m\n"
    "- MoveRight: Move right 0.25m\n"
    "- RotateRight: Rotate clockwise 45°\n"
    "- RotateLeft: Rotate counter-clockwise 45°\n"
    "- LookUp: Tilt camera up 30°\n"
    "- LookDown: Tilt camera down 30°\n"
    "- Stop: Declare that you have reached the target viewpoint\n\n"
    "You will receive:\n"
    "1. Your current observation\n"
    "2. Your recent action history (if available)\n"
    "3. The target viewpoint you need to match\n\n"
    "Use your action history to avoid repeating ineffective actions "
    "(e.g. if MoveAhead caused a collision, try a different direction).\n\n"
    "You MUST respond in exactly this format:\n"
    "Action: <action name>"
)


# --- thor_view_env.py:126-141 (verbatim) --------------------------------------------------------------------------
def build_concat_user_text(valid_actions: List[str]) -> str:
    """Per-turn user text in concat multi-turn mode (history is the chat —
    do NOT inject a "Your recent actions" text block; that's the no_concat /
    SFT-aligned path).

    Used by both `ThorViewEnv._build_obs_str` at RL rollout time and by
    `scripts/build_sft_online_data.py` at SFT data build time. Single source
    of truth — they cannot drift.
    """
    return "\n".join([
        "Your CURRENT observation:",
        IMAGE_PH,
        "TARGET viewpoint you must match:",
        IMAGE_PH,
        f"Valid actions at this step: {', '.join(valid_actions)}",
    ])


# --- eval_va.py:103-124 (verbatim) --------------------------------------------------------------------------------
_ACTION_RE = re.compile(r"Action\s*:\s*([A-Za-z]+)", re.IGNORECASE)

_VALID_ACTIONS = {
    "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
    "RotateLeft", "RotateRight", "LookUp", "LookDown", "Stop",
}


def parse_action(text: str) -> Optional[str]:
    """Return the canonical action name, or None if unparseable."""
    if not text:
        return None
    text = text.replace("</think>", "").strip()
    m = _ACTION_RE.search(text)
    if not m:
        return None
    cand = m.group(1)
    # Case-normalize against known actions
    for v in _VALID_ACTIONS:
        if cand.lower() == v.lower():
            return v
    return None


# --- ours -----------------------------------------------------------------------------------------------------------
def regex_action(reply: str) -> Optional[str]:
    """The primary / blind arms' parser: re.fullmatch(pilot_cfg.REGEX, reply).group(1), else None."""
    if not isinstance(reply, str):
        return None
    m = _REGEX_C.fullmatch(reply)
    return m.group(1) if m else None


VALID_PREFIX = "Valid actions at this step: "


def valid_from_user_text(text: str) -> List[str]:
    """Inverse of build_concat_user_text: the valid list of a user text (ValueError if the text is not exactly
    build_concat_user_text(list))."""
    last = text.rsplit("\n", 1)[-1]
    if not last.startswith(VALID_PREFIX):
        raise ValueError(f"no valid-actions line in {text!r}")
    rest = last[len(VALID_PREFIX):]
    valid = rest.split(", ") if rest else []
    if build_concat_user_text(valid) != text:
        raise ValueError(f"not a TVRBench concat user text: {text!r}")
    return valid


# --- source_check ---------------------------------------------------------------------------------------------------
VIEW_ENV_REL = "tvrbench/online_rl/thor_view_env.py"
EVAL_VA_REL = "scripts/eval_va.py"

# >= 40 replies for the parse_action equality check: well-formed, case variants, </think> artefacts, junk
PARSE_CORPUS = (
    "Action: MoveAhead", "Action: MoveBack", "Action: MoveLeft", "Action: MoveRight", "Action: RotateRight",
    "Action: RotateLeft", "Action: LookUp", "Action: LookDown", "Action: Stop", "action: stop", "ACTION: LOOKDOWN",
    "Action:Stop", "Action :  rotateleft", "Action:\tMoveAhead", "</think>\n\nAction: MoveRight",
    "<think>go</think>Action: LookUp", "</think>Action: Stop</think>", "Action: Move Ahead", "Action: MoveAhead.",
    "Action: MoveAheadX", "Action: Jump", "Action: InvalidAction", "MoveAhead", "Stop", "", " ", "\n",
    "I think Action: RotateRight is best", "Action: RotateRight\nAction: Stop", "Action:", "Action: 123",
    "Actions: Stop", "action:stop!", "Action: lookup now", "Action: Stop.", "Action\n:\nStop", "Action - Stop",
    "Answer: MoveAhead", "Action: MOVEBACK", "</think>", "Action: Rotate_Left", "Action: Look-Up", "Action: É",
    "Action: StopStop", "Action: stop and go", "\u200bAction: Stop", "action : moveright\n",
)


def _sha256_file(p) -> Optional[str]:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


def _top_nodes(tree: ast.Module) -> dict:
    """name -> node for top-level FunctionDef and single-target Assign statements."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            out[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = node
    return out


def _dump(node) -> Optional[str]:
    return None if node is None else ast.dump(node)


def source_check_detail(repo=None) -> dict:
    """The full check (see source_check); every comparison with its evidence."""
    repo = Path(lvp.TVR_REPO if repo is None else repo)
    ours = _top_nodes(ast.parse(Path(__file__).read_text(encoding="utf-8")))
    out: dict = {"repo": str(repo), "errors": []}
    try:
        tv = _top_nodes(ast.parse((repo / VIEW_ENV_REL).read_text(encoding="utf-8")))
        ev_src = (repo / EVAL_VA_REL).read_text(encoding="utf-8")
        ev = _top_nodes(ast.parse(ev_src))
    except (OSError, SyntaxError) as e:
        out["errors"].append(repr(e))
        tv, ev, ev_src = {}, {}, ""
    sp_node = tv.get("SYSTEM_PROMPT")
    sp_value = None
    if sp_node is not None:
        try:
            sp_value = ast.literal_eval(sp_node.value)
        except ValueError as e:
            out["errors"].append(f"SYSTEM_PROMPT literal: {e!r}")
    out["system_prompt"] = {"dump_equal": _dump(sp_node) == _dump(ours.get("SYSTEM_PROMPT")),
                            "value_equal": sp_value == SYSTEM_PROMPT, "len": len(SYSTEM_PROMPT),
                            "sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()}
    ut = {k: _dump(tv.get(k)) == _dump(ours.get(k)) and tv.get(k) is not None
          for k in ("build_concat_user_text", "IMAGE_PH")}
    out["user_text"] = ut
    pa = {k: _dump(ev.get(k)) == _dump(ours.get(k)) and ev.get(k) is not None
          for k in ("parse_action", "_ACTION_RE", "_VALID_ACTIONS")}
    # exec the extracted official parser in a fresh namespace and compare on the corpus
    corpus_equal = False
    diffs = []
    if all(ev.get(k) is not None for k in ("parse_action", "_ACTION_RE", "_VALID_ACTIONS")):
        mod = ast.Module(body=[ev["_ACTION_RE"], ev["_VALID_ACTIONS"], ev["parse_action"]], type_ignores=[])
        ns: dict = {"re": re, "Optional": Optional}
        try:
            exec(compile(mod, str(repo / EVAL_VA_REL), "exec"), ns)  # noqa: S102 (pinned read-only source)
            for s in PARSE_CORPUS:
                a, b = ns["parse_action"](s), parse_action(s)
                if a != b:
                    diffs.append({"text": s, "official": a, "ours": b})
            corpus_equal = not diffs
        except Exception as e:  # noqa: BLE001
            out["errors"].append(f"exec parse_action: {e!r}")
    pa.update(corpus_n=len(PARSE_CORPUS), corpus_equal=corpus_equal, corpus_diffs=diffs)
    out["parse_action"] = pa
    want = pilot_cfg.FROZEN["tvrbench"]["files_sha256"]
    got = {rel: _sha256_file(repo / rel) for rel in want}
    out["files_sha256"] = {"expected": dict(want), "got": got, "unequal": sorted(r for r in want if got[r] != want[r])}
    out["result"] = {
        "system_prompt_equal": bool(out["system_prompt"]["dump_equal"] and out["system_prompt"]["value_equal"]),
        "user_text_equal": bool(ut["build_concat_user_text"] and ut["IMAGE_PH"]),
        "parse_action_equal": bool(pa["parse_action"] and pa["_ACTION_RE"] and pa["_VALID_ACTIONS"]
                                   and corpus_equal),
        "files_sha256_equal": not out["files_sha256"]["unequal"],
    }
    return out


def source_check(repo=None) -> dict:
    """{"system_prompt_equal", "user_text_equal", "parse_action_equal", "files_sha256_equal"} (all bool) for the
    pinned TVRBench checkout `repo` (default lvp.TVR_REPO).  No import of the checked files."""
    return dict(source_check_detail(repo)["result"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vpd.tvr.tvr_text", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="source_check on the pinned checkout; prints the detail JSON")
    c.add_argument("--repo", default=str(lvp.TVR_REPO))
    a = ap.parse_args(argv)
    d = source_check_detail(a.repo)
    print(json.dumps(d, indent=1, ensure_ascii=False))
    return 0 if all(d["result"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())

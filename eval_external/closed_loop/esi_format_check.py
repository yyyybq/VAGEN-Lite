"""ESI-Bench first-step format check (no simulator): can a model answer in ESI-Bench's own format?

    python -m eval_external.closed_loop.esi_format_check prepare          # login node: pick questions, fetch renders
    python -m eval_external.closed_loop.esi_format_check run --models ours-SFT base-7B ...   # one-card job
    python -m eval_external.closed_loop.esi_format_check summary

ESI-Bench runs in Isaac Sim, which needs RT cores; our H800s have none, so no episode can run.  What can run is the
first model call of an episode, as src/active_explore/pipeline.py makes it:
- the question: dataset/json_clean of ESI-Bench@9f6f15e, two per big category (10 categories, 20 questions; the
  first question of the first two subtasks in name order, or the first two questions when a category has one);
- the system text "Return exactly one valid JSON object and nothing else." and the user contents
  ["[CURRENT VIEW - step 1]", image, task prompt], the task prompt from the task module's build_system_prompt
  (threshold 0.8, min_steps 3, camera_info from the module's initial_camera, as the pipeline does);
- the image: the question's render in ESI-Bench/esi-bench-passive-single (the official offline single-view set),
  standing in for the first simulator capture;
- the call: ESI-Bench's own GPTModel.generate_json against a local vLLM server (response_format json_object,
  max 1024 tokens), greedy (temperature 0, top_p 1) as everywhere in eval_external; ESI-Bench's default is 0.2 / 0.9.
The task modules import omnigibson at load time; it is replaced by an inert module (only prompt building and
parsing run, neither touches the simulator).  A reply counts as invalid when (a) no JSON object parses (the
pipeline would silently act move_forward), (b) "action" is none of the ten camera moves, "stop", or an object
command the prompt offers (pick up / place ... on top of / put back <object>, pick_up_obj_near_<x>) -- the
pipeline would do nothing -- or (c) "answer" or a numeric "confidence" is missing.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.machinery
import json
import re
import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
ESI = ROOT / "repos" / "ESI-Bench"
SRC = ESI / "src" / "active_explore"
QUESTIONS = ESI / "dataset" / "json_clean"
RENDERS_REPO, RENDERS_REV = "ESI-Bench/esi-bench-passive-single", "159806e86b5ef49b5cd88625ec9978a9fa0a18a2"
OUT = ROOT / "runs" / "esi_format_check"
SYSTEM = "Return exactly one valid JSON object and nothing else."
STUBBED = ("omnigibson", "cv2", "bddl")


class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUBBED:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name)
        m.__path__, m.__spec__, m.__name__ = [], spec, spec.name
        return m

    def exec_module(self, module):
        pass


def esi_imports():
    if not any(isinstance(f, _Stub) for f in sys.meta_path):
        sys.meta_path.insert(0, _Stub())
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from models.gpt import GPTModel, parse_json_response  # noqa: F401  (ESI-Bench's own client and parser)
    from tasks._registry import module_for_name, module_for_payload

    return GPTModel, parse_json_response, module_for_name, module_for_payload


def payload_of(row: dict) -> dict:
    """pipeline.payload_from_hf_row, unchanged in substance."""
    def dec(v):
        try:
            return json.loads(v) if isinstance(v, str) and v.strip() else v
        except json.JSONDecodeError:
            return v

    meta = dec(row.get("metadata_json"))
    p = dict(meta) if isinstance(meta, dict) else {}
    p["scene"], p["room"] = row.get("scene") or p.get("scene"), row.get("room") or p.get("room")
    if row.get("question") and not p.get("_question"):
        p["_question"] = row["question"]
    ans = dec(row.get("answer"))
    p.setdefault("_ground_truth", ans)
    p.setdefault("answer", ans)
    p.update({"_hf_id": row.get("id"), "_hf_big_task": row.get("big_task"), "_hf_small_task": row.get("small_task"),
              "_hf_runner_task": row.get("runner_task"), "_hf_options": dec(row.get("options_json")) or [],
              "_hf_image_paths": dec(row.get("image_paths_json")) or []})
    return p


def select() -> list[Path]:
    """Two questions per big category that have a render: the first (by path) of the first two subtasks that have
    one, or the first two of a category's only subtask."""
    renders = renders_by_id()
    picked = []
    for big in sorted(p for p in QUESTIONS.iterdir() if p.is_dir()):
        subs = sorted(p for p in big.iterdir() if p.is_dir())
        usable = [[q for q in sorted(s.glob("*/*/*.json")) if json.loads(q.read_text())["id"] in renders] for s in subs]
        usable = [u for u in usable if u]
        picked += [u[0] for u in usable[:2]] if len(usable) >= 2 else usable[0][:2]
    return picked


def renders_by_id() -> dict[str, str]:
    """Question id -> its render in the offline set (file names start with the id); the listing is cached."""
    cache = OUT / "render_files.json"
    if not cache.exists():
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(RENDERS_REPO, revision=RENDERS_REV)
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".png"))))
    out = {}
    for f in json.loads(cache.read_text()):
        out.setdefault(f.rsplit("/", 1)[1].split("_", 1)[0], f)
    return out


CAMERA_ACTIONS = {"move_forward", "move_backward", "move_left", "move_right", "move_up", "move_down", "turn_left",
                  "turn_right", "turn_up", "turn_down"}  # pipeline.CAMERA_ACTIONS; "stop" ends the episode
OBJECT_COMMANDS = ("pick up ", "place ", "put back ")  # offered as "pick up <object>" etc. by some tasks
_LIST_HEAD = re.compile(r"(actions?|ACTIONS)\s*(\([^)]*\))?\s*:\s*$")


def allowed_actions(prompt: str) -> set[str]:
    """Actions the prompt lists beyond the camera moves and stop: the lines under a header ending in "actions:"
    (up to the next blank line), named pickup actions, and object commands ("pick up <object>")."""
    out, lines = set(), prompt.splitlines()
    for i, line in enumerate(lines):
        if not _LIST_HEAD.search(line.strip()):
            continue
        for item in lines[i + 1:]:
            if not item.strip():
                break
            for part in re.split(r"\s*\|\s*|,\s+", re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", item.strip())):
                part = part.strip().strip("'\"`.")
                if part and len(part) < 80:
                    out.add(part.lower())
                    out.add(re.split(r"\s*(?::|\s-\s|\s—\s|\()", part)[0].strip().strip("'\"`").lower())  # "name: what"
    out |= {n.strip().lower() for n in re.findall(r"pick_up_obj_near_[^,.\n|]+", prompt)}
    out |= {verb.strip() + " <object>" for verb in OBJECT_COMMANDS if f"{verb}<" in prompt}
    return out


def action_ok(action, prompt: str) -> bool:
    a = " ".join(str(action or "").split()).lower()
    if a in CAMERA_ACTIONS or a == "stop" or a in allowed_actions(prompt):
        return True
    templates = [t for t in allowed_actions(prompt) if "<" in t]  # e.g. "place <object> on top of <object>"
    return any(re.fullmatch(re.sub(r"<[^>]+>", ".+", re.escape(t).replace(r"\<", "<").replace(r"\>", ">")), a)
               for t in templates) or any(f"{v}<" in prompt and a.startswith(v) and len(a) > len(v)
                                          for v in OBJECT_COMMANDS)


def answer_keys(prompt: str) -> set[str]:
    """The fields of the prompt's JSON template other than action / reasoning / confidence (answer, choice_letter...)."""
    keys = set(re.findall(r'"(\w+)"\s*:', prompt)) - {"action", "reasoning", "confidence"}
    return keys or {"answer"}


def build_prompt(module, payload: dict, source_json: Path | None = None) -> tuple[str, dict]:
    import inspect

    def call(fn, *args, **kw):
        params = inspect.signature(fn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
        return fn(*args, **kw)

    state, camera_info = {}, None
    _, _, camera_info = module.initial_camera(payload)  # as pipeline.main: pose and camera_info from the question
    if hasattr(module, "preprocess"):
        cfg = types.SimpleNamespace(threshold=0.8, min_steps=3, max_steps=30, history_image_window=5)
        try:
            s = call(module.preprocess, payload, source_json=source_json, config=cfg)
            state = s if isinstance(s, dict) else {}
        except Exception as e:  # recorded; the prompt is still built
            state = {"_preprocess_error": f"{type(e).__name__}: {e}"}
    prompt = call(module.build_system_prompt, payload, 0.8, 3, camera_info, task_state=state)
    return prompt, state


def classify(parsed: dict, prompt: str) -> list[str]:
    """What ESI-Bench would replace with a default (its task parsers fill a missing action with move_forward, a
    missing answer with "unsure", a missing confidence with 0.0; the pipeline does nothing for an action it does
    not know)."""
    bad = []
    if parsed.get("error") == "JSON parse failed" or "raw_response" in parsed:
        bad.append("no_json")
    if not action_ok(parsed.get("action"), prompt):
        bad.append("action_not_offered")
    if not answer_keys(prompt) & set(parsed):
        bad.append("no_answer")
    try:
        float(parsed.get("confidence"))
    except (TypeError, ValueError):
        bad.append("no_confidence")
    return bad


def cmd_prepare(_args) -> None:
    from huggingface_hub import hf_hub_download

    _, _, module_for_name, module_for_payload = esi_imports()
    OUT.mkdir(parents=True, exist_ok=True)
    renders = renders_by_id()
    items = []
    for qf in select():
        row = json.loads(qf.read_text())
        payload = payload_of(row)
        mod_name = module_for_payload(payload) or module_for_name(row.get("runner_task"))
        module = importlib.import_module(mod_name)
        prompt, state = build_prompt(module, payload, qf)
        local = hf_hub_download(RENDERS_REPO, renders[row["id"]], repo_type="dataset", revision=RENDERS_REV,
                                local_dir=str(ROOT / "data" / "esi_passive_single"))
        items.append({"id": row["id"], "big_task": row["big_task"], "small_task": row["small_task"],
                      "question_json": str(qf.relative_to(ESI)), "module": mod_name, "image": local,
                      "prompt": prompt, "allowed_actions": sorted(allowed_actions(prompt)),
                      "preprocess_error": state.get("_preprocess_error")})
    (OUT / "items.json").write_text(json.dumps(items, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(items)} questions, {len({i['big_task'] for i in items})} categories -> {OUT / 'items.json'}")


def cmd_run(args) -> None:
    from eval_external.closed_loop.vllm_server import VLLMServer

    GPTModel, parse_json_response, _, _ = esi_imports()
    items = json.loads((OUT / "items.json").read_text())
    for code in args.models:
        out = OUT / f"{code}.jsonl"
        out.unlink(missing_ok=True)
        with VLLMServer(code, log=OUT / f"{code}.vllm.log") as srv:
            client = GPTModel(api_key="EMPTY", model=code, base_url=srv.base_url)
            for it in items:
                contents = ["[CURRENT VIEW - step 1]", Path(it["image"]), it["prompt"]]
                try:
                    parsed, raw, finish = client.generate_json(contents=contents, system_instruction=SYSTEM,
                                                               max_output_tokens=1024, temperature=0.0, top_p=1.0)
                    err = None
                except Exception as e:  # a server-side rejection is recorded, not scored
                    parsed, raw, finish, err = {}, "", None, f"{type(e).__name__}: {e}"
                rec = {"id": it["id"], "big_task": it["big_task"], "small_task": it["small_task"], "raw": raw,
                       "finish_reason": finish, "parsed": parsed, "call_error": err,
                       "invalid": None if err else classify(parsed, it["prompt"])}
                with open(out, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_summary(_args) -> None:
    """Re-scores every saved reply with classify() (so a change of the rule needs no rerun), then counts."""
    prompts = {i["id"]: i["prompt"] for i in json.loads((OUT / "items.json").read_text())}
    rows = []
    for f in sorted(OUT.glob("*.jsonl")):
        recs = [json.loads(line) for line in open(f)]
        for r in recs:
            if r["call_error"] is None:
                r["invalid"] = classify(r["parsed"], prompts[r["id"]])
        scored = [r for r in recs if r["invalid"] is not None]
        inv = [r for r in scored if r["invalid"]]
        kinds = {}
        for r in inv:
            for k in r["invalid"]:
                kinds[k] = kinds.get(k, 0) + 1
        rows.append({"model": f.stem, "n": len(recs), "scored": len(scored), "invalid": len(inv),
                     "invalid_pct": round(100 * len(inv) / max(len(scored), 1), 1), "kinds": kinds,
                     "call_errors": len(recs) - len(scored)})
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    for r in rows:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("run")
    run.add_argument("--models", nargs="+", required=True)
    sub.add_parser("summary")
    args = ap.parse_args()
    {"prepare": cmd_prepare, "run": cmd_run, "summary": cmd_summary}[args.cmd](args)


if __name__ == "__main__":
    main()

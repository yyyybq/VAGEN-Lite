"""Gate 0 of docs/PREREG_PPO2.md (frozen 2026-11-13): the representation probe.  A diagnosis, no gate.

Question: in C0 SFT e8 (the RL start model; every critic so far started from it), at the position where the critic
reads a turn's value, how much value-relevant ground truth is linearly decodable?

1. The episodes: ``evaluation run --critic-probe TABLES_DIR`` (one sampled episode per b3000 task, temperature 0.8,
   top-p 0.92, history on, the V4 env with critic_features), through an eval-only training chain.  Each row keeps the
   system prompt; each turn the frame (PNG), the user text, the answer, the unrounded reward, the env's truth at the
   pose of the turn's observation and the digest of the prompt's token ids vLLM ran.
2. ``extract`` (one card, bf16, no grad; the Qwen3-VL runtime's Python, transformers >= 4.57):

       python -m data_gen.interiorgs_v10.repr_probe extract --results EVAL/NAME --out FEAT [--shard i/n] [--limit N]

   The sequence is the TRAINING one ("拼接方式同训练（concat）"): the ids RL's concat agent loop
   (vagen/agent_loop/gym_agent_loop.py, GymAgentLoop) builds for the episode, mirrored by training_layout: the first
   prompt (system + first user turn, the chat template with its generation prompt, the processor with the first frame),
   then per turn the answer's tokens as the generated response ids (the answer text tokenised alone, plus the end of
   turn <|im_end|> the generation stops on; none for an answer cut by the token limit), then the next user turn's suffix
   (the chat template of [placeholder system, user] with the generation prompt, the processor with that turn's frame,
   the placeholder system's ids removed).  One forward per episode with every frame's pixel_values / image_grid_thw in
   order (the model computes M-RoPE on the whole sequence).  Turn t's position: the length of the ids before its answer
   - 1, the last token of its generation prompt (where verl's critic reads the value of the turn's first response
   token).  Each turn's prompt in the eval's own layout (evaluation.chat_messages with the history, the frames from
   their PNGs) must equal what vLLM ran (prompt_tokens, prompt_ids_sha256): the check that the recorded text and frames
   are the run's; the first prompt is the same call in both layouts and must agree.  A difference stops the extraction
   (--on-mismatch skip records the episode in skipped.jsonl and counts it).  A run with --sft-text is refused (training
   shows the env's text).  Five readings at each position: the outputs of decoder layers 9, 18, 27 and 36 (hooks on the
   layers; layers 9 / 18 / 27 are HF's hidden_states[9] / [18] / [27]; HF's last hidden_states entry is already after
   the final norm, so layer 36 is read by its own hook) and the output of the final norm; the first episode of each
   call also runs with output_hidden_states and stops the extraction unless HF's entries equal the hooks' and the norm
   of layer 36 gives the final-norm reading.  The layer count must be 36; non-finite values stop the extraction.
   Written per shard: chunk_*.npz (one float16 array per reading, float32 for a chunk whose values pass float16's
   range), chunk_*.jsonl (one row per turn, the npz rows' order), manifest.json.  Resumable: a rerun skips the episodes
   already written.
3. ``fit`` (GPU when there is one, else CPU; deterministic, fixed seeds):

       python -m data_gen.interiorgs_v10.repr_probe fit --features FEAT --out FIT

   Linear probes per reading: ridge regression for G_t (the V4 Monte-Carlo return-to-go, the turn rewards from t to the
   end, gamma 1; the main target), d (the remaining minimum primitive actions; turns without d left out) and the
   turns left (t to the end, t included); L2 logistic regression for the episode's success, the bearing sector
   (front / right / back / left; no bearing left out; multinomial) and the distance condition (too_close / ok /
   too_far; absolute_positioning turns with a condition; multinomial).  5 folds grouped by scene_id (assign_folds:
   deterministic, balanced by turns); the regularisation chosen inside each training fold by an inner 3-fold grouped
   split over a fixed grid (ridge alpha 1e-2 .. 1e7 by decades, sum-of-squares loss as sklearn's Ridge; logistic
   C = 1 / alpha, sklearn's convention: 0.5 |w|^2 + C * summed log-loss; the inner choice by the pooled squared error /
   the mean log-loss; a choice at the grid's edge is flagged in results.md), the features standardised with the
   training fold's statistics.  Reported out of fold:
   EV = 1 - var(y - y_hat) / var(y) for G_t, d and the turns left; AUC and accuracy for success; accuracy and the
   majority-class baseline (the training fold's majority) for bearing, condition and success.
   Baselines on the same turns and folds: the privileged-feature MLP (vagen/feature_critic.py: the 29-number encode,
   the same three-layer MLP with its success-logit and turns-left heads, the read-out a*p + b*turns_left + c + e*phi
   fitted on the training fold's G_t; MLP_EPOCHS passes) and ridge on the turn index and family one-hot.
   The reading (pre-registered): the best reading's out-of-fold EV of G_t >= 0.3 "linearly readable", < 0.1 "not
   readable", otherwise "weak"; the other targets numbers only.  results.json, results.md.  Next to it, reported only:
   the turns past RL's concat response budget, the answers cut at the eval's token cap, and the best reading's EV of
   G_t on the turns within the budget (budget_report).

The agent loop's chat-template arguments: extract --ppo-config PPO_YAML reads the run's data.apply_chat_template_kwargs,
else TEMPLATE_KWARGS (the PPO template's); the manifest records them and their source, and a shard keeps one.

Every turn with the env's truth enters the probes; a turn whose features the env could not compute
(critic_features None) is left out of every probe and counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10 import evaluation as E

VERSION = "repr_probe_v1"
NUM_LAYERS = 36  # Qwen3-VL-8B's decoder layers; extract refuses another count
LAYERS = (9, 18, 27, 36)  # outputs of these decoder layers (1-based: layer i is model.layers[i - 1])
READINGS = ("layer9", "layer18", "layer27", "layer36", "final_norm")
READING_SOURCES = {"layer9": "output of decoder layer 9 (HF hidden_states[9])",
                   "layer18": "output of decoder layer 18 (HF hidden_states[18])",
                   "layer27": "output of decoder layer 27 (HF hidden_states[27])",
                   "layer36": "output of decoder layer 36 before the final norm (forward hook on the last layer)",
                   "final_norm": "output of the final norm (HF hidden_states[-1])"}
ALPHAS = tuple(10.0 ** k for k in range(-2, 8))  # ridge: 1e-2 .. 1e7
CS = tuple(1.0 / a for a in ALPHAS)  # logistic C = 1 / alpha: 1e2 .. 1e-7
# the concat agent loop's settings (vagen/agent_loop/gym_agent_loop.py, data.apply_chat_template_kwargs of
# data_gen/interiorgs_v10/configs/ppo_template_v10.yaml) and the end of a generated turn (the Qwen chat templates')
TEMPLATE_KWARGS = {"enable_thinking": False}
END_OF_TURN = "<|im_end|>"
PLACEHOLDER_SYSTEM = {"role": "system", "content": "placeholder"}  # GymAgentLoop's: its ids are cut off each suffix
OUTER_FOLDS, INNER_FOLDS = 5, 3
SEED = 20261019
READABLE, NOT_READABLE = 0.3, 0.1  # the pre-registered reading (docs/PREREG_PPO2.md, gate 0)
MLP_EPOCHS = 32  # the privileged MLP's passes over a training fold (the RL critic sees a row 4 passes x 8 updates)
LOGISTIC_MAX_ITER = 300
FORWARD_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")  # model_batch -> the model
# per-episode counts summed in the shard manifest: the eval-layout check against the run, how the answers ended
EPISODE_COUNTS = ("prompt_checked", "prompt_unchecked", "eos_appended", "length_cut", "finish_unknown")
FP16_MAX = 65504.0
CHUNK_EPISODES = 25
AP = "absolute_positioning"
BEARINGS = ("front", "right", "back", "left")
CONDITIONS = ("too_close", "ok", "too_far")


class Mismatch(RuntimeError):
    """A rebuilt prompt differs from the episode's sequence or from the prompt the run gave the model."""


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _num(x: Any, nd: int = 6) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ---------------------------------------------------------------------------
# the episodes
# ---------------------------------------------------------------------------
def read_run(result: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """model.json and results of a finished critic-probe result folder (refused otherwise)."""
    result = Path(result)
    model = json.loads((result / "model.json").read_text())
    if model.get("critic_probe") != E.CRITIC_PROBE_VERSION:
        raise SystemExit(f"{result}: model.json critic_probe {model.get('critic_probe')!r}, this reads "
                         f"{E.CRITIC_PROBE_VERSION!r} (evaluation run --critic-probe)")
    if not model.get("history"):
        raise SystemExit(f"{result}: run without history; the probe teacher-forces each episode as one sequence")
    if model.get("grounding_probe"):
        raise SystemExit(f"{result}: a grounding-probe run")
    if model.get("sft_text"):
        raise SystemExit(f"{result}: run with --sft-text; RL training shows the env's text as it is")
    if not (result / "done.json").is_file():
        raise SystemExit(f"{result}: no done.json (the run has not finished)")
    rows = _jsonl(result / "results.jsonl")
    keys = Counter((r["index"], r.get("sample")) for r in rows)
    twice = sorted(k for k, n in keys.items() if n > 1)
    if twice:
        raise SystemExit(f"{result}: episodes recorded twice: {twice[:5]}")
    return model, rows


def episode_key(result: Path, row: Dict[str, Any]) -> str:
    return f"{Path(result).resolve()}#{int(row['index'])}#{row.get('sample')}"


def list_episodes(results: Sequence[Path]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every recorded episode of the result folders (in the given order, then by index and sample; infrastructure
    errors left out) and what the folders say; the folders must agree on the model and the chat layout."""
    episodes: List[Dict[str, Any]] = []
    folders, errors = [], 0
    for res in results:
        res = Path(res)
        model, rows = read_run(res)
        folders.append({"path": str(res.resolve()), "model_json_sha256": _sha(res / "model.json"),
                        "results_sha256": _sha(res / "results.jsonl"),
                        **{k: model.get(k) for k in ("name", "model", "history", "sft_text", "max_turns", "greedy",
                                                     "temperature", "top_p", "sample_seed", "critic_probe",
                                                     "env_switches", "prompt_format")}})
        for row in sorted(rows, key=lambda r: (int(r["index"]), r.get("sample") or 0)):
            if "error" in row:
                errors += 1
                continue
            episodes.append({"key": episode_key(res, row), "result": res, "model": model, "row": row})
    for field in ("model", "history", "sft_text", "max_turns"):
        values = {json.dumps(f[field]) for f in folders}
        if len(values) > 1:
            raise SystemExit(f"the result folders differ in model.json {field}: {sorted(values)}")
    return episodes, {"folders": folders, "run_errors": errors, "episodes": len(episodes)}


def parse_shard(text: str) -> Tuple[int, int]:
    i, _, n = str(text).partition("/")
    i, n = int(i), int(n)
    if not 0 <= i < n:
        raise SystemExit(f"--shard {text}: want i/n with 0 <= i < n")
    return i, n


def shard_of(episodes: Sequence[Dict[str, Any]], i: int, n: int) -> List[Dict[str, Any]]:
    """Episode k (in list_episodes' order) goes to shard k % n."""
    return [e for k, e in enumerate(episodes) if k % n == i]


# ---------------------------------------------------------------------------
# the chat: the eval's layout (the check against the run) and the training layout (the readings)
# ---------------------------------------------------------------------------
def image_placeholder(_image: Any) -> Dict[str, Any]:
    """An image part as vLLM's chat hands it to the chat template (the frames go to the processor separately)."""
    return {"type": "image"}


def conversation(system: str, turns: Sequence[Tuple[str, Any, str]], upto: int, as_sft: bool = False
                 ) -> List[Dict[str, Any]]:
    """The eval layout: the messages of turn ``upto``'s model call, exactly as the run built them
    (evaluation.chat_messages with every earlier turn: the history), or with ``upto == len(turns)`` the whole episode
    with every answer.  Only to check the recorded text and frames against the run (check_prompts); the readings come
    from training_layout."""
    if not turns or not 0 <= upto <= len(turns):
        raise ValueError(f"conversation: turn {upto} of {len(turns)}")
    if upto < len(turns):
        user, image, _ = turns[upto]
        return E.chat_messages(system, list(turns[:upto]), user, image, image_placeholder, as_sft)
    *earlier, (user, image, answer) = turns
    messages = E.chat_messages(system, earlier, user, image, image_placeholder, as_sft)
    return messages + [{"role": "assistant", "content": E.sft_text(answer) if as_sft else answer}]


def _ids(batch: Any) -> List[int]:
    """The token ids of a processor's batch of one text (input_ids [1, L], a tensor or a list)."""
    ids = batch["input_ids"][0]
    return [int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else ids)]


def encode(processor: Any, messages: List[Dict[str, Any]], images: Sequence[Any], generation_prompt: bool
           ) -> Tuple[List[int], Any]:
    """Token ids (and the processor's batch) of a chat as vLLM's chat builds the prompt: the chat template, then the
    processor on the text and the frames in their order (it expands each image placeholder to the frame's tokens)."""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
    batch = processor(text=[text], images=list(images) if images else None, return_tensors="pt")
    return _ids(batch), batch


def check_prompts(trajectory: Sequence[Dict[str, Any]], prompts: Sequence[Sequence[int]]) -> Dict[str, int]:
    """Each rebuilt eval-layout prompt against the run's record of the prompt vLLM ran: its length (prompt_tokens) and
    its token ids (prompt_ids_sha256, evaluation.ids_digest).  A difference: Mismatch; a turn without a digest is
    counted."""
    checked = unchecked = 0
    for t, (turn, prompt) in enumerate(zip(trajectory, prompts)):
        if turn.get("prompt_tokens") is not None and int(turn["prompt_tokens"]) != len(prompt):
            raise Mismatch(f"turn {t}: the run's prompt had {turn['prompt_tokens']} tokens, the rebuilt one "
                           f"{len(prompt)}")
        digest = turn.get("prompt_ids_sha256")
        if digest is None:
            unchecked += 1
            continue
        if digest != E.ids_digest(prompt):
            raise Mismatch(f"turn {t}: the rebuilt prompt's token ids differ from the run's ({len(prompt)} tokens)")
        checked += 1
    return {"prompt_checked": checked, "prompt_unchecked": unchecked}


def obs_content(text: str, image_placeholder_text: str = "<image>", video_placeholder: str = "<video>"
                ) -> List[Dict[str, Any]]:
    """gym_agent_loop.convert_obs_to_content (line by line, without its count check): an observation's text split at
    its placeholders into text and image parts, empty pieces dropped."""
    import re

    pattern = f"({re.escape(image_placeholder_text)}|{re.escape(video_placeholder)})"
    content: List[Dict[str, Any]] = []
    for seg in re.split(pattern, text):
        if not seg:
            continue
        if seg == image_placeholder_text:
            content.append({"type": "image"})
        elif seg == video_placeholder:
            content.append({"type": "video"})
        else:
            content.append({"type": "text", "text": seg})
    return content


def training_layout(processor: Any, system: str, turns: Sequence[Tuple[str, Any, str]],
                    finishes: Optional[Sequence[Optional[str]]] = None,
                    template_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The episode's ids as RL's concat agent loop builds them (vagen/agent_loop/gym_agent_loop.py, GymAgentLoop),
    mirrored step by step; ``turns``: (user text, frame, answer text) per turn, ``finishes``: each answer's finish
    reason (None: unknown, taken as stopped).

      init_class            system_prompt_prefix: the ids of the chat template of [placeholder system] without the
                            generation prompt, through the processor without images
      _handle_pending_state prompt_ids: the chat template of [system, first user] with the generation prompt, through
                            the processor with the first frame (the system's and users' content as
                            convert_obs_to_content builds it)
      _handle_generating    prompt_ids += the response ids: here the answer text tokenised alone (add_special_tokens
                            False) plus the end-of-turn token the generation stops on (RL has no stop string: it runs
                            to <|im_end|>, which vLLM keeps in the token ids); none after an answer cut by the limit
      _handle_env_state     (not after the last turn) prompt_ids += the chat template of [placeholder system, user] with
                            the generation prompt, through the processor with the turn's frame, less the first
                            len(system_prompt_prefix) ids

    Returns the ids, each turn's position (the ids before its answer, less one: its generation prompt's last token),
    the processor's batches in order (their frames' pixel_values / image_grid_thw) and the first prompt's length."""
    kw = dict(TEMPLATE_KWARGS if template_kwargs is None else template_kwargs)
    tok = processor.tokenizer
    eos = tok.convert_tokens_to_ids(END_OF_TURN)
    prefix_text = processor.apply_chat_template([PLACEHOLDER_SYSTEM], add_generation_prompt=False, tokenize=False,
                                                **kw)
    prefix = _ids(processor(text=[prefix_text], return_tensors="pt"))
    user0, frame0, _ = turns[0]
    messages = [{"role": "system", "content": obs_content(system)}, {"role": "user", "content": obs_content(user0)}]
    raw = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kw)
    first = processor(text=[raw], images=[frame0], return_tensors="pt")
    ids, parts = _ids(first), [first]
    prompt0 = len(ids)
    positions: List[int] = []
    ends = {"eos_appended": 0, "length_cut": 0, "finish_unknown": 0}
    for t, (user, frame, answer) in enumerate(turns):
        if t > 0:  # the user suffix of turn t, after turn t - 1's answer
            raw = processor.apply_chat_template([PLACEHOLDER_SYSTEM, {"role": "user", "content": obs_content(user)}],
                                                add_generation_prompt=True, tokenize=False, **kw)
            batch = processor(text=[raw], images=[frame], return_tensors="pt")
            suffix = _ids(batch)
            if suffix[:len(prefix)] != prefix:  # the agent loop cuts len(prefix) ids without looking
                raise Mismatch(f"turn {t}: the user suffix does not start with the placeholder system's ids")
            ids += suffix[len(prefix):]
            parts.append(batch)
        positions.append(len(ids) - 1)
        finish = finishes[t] if finishes is not None and t < len(finishes) else None
        response = [int(i) for i in tok.encode(answer, add_special_tokens=False)]
        if finish == "length":
            ends["length_cut"] += 1
        else:
            response.append(int(eos))
            ends["eos_appended"] += 1
            ends["finish_unknown"] += int(finish is None)
        ids += response
    return {"input_ids": ids, "positions": positions, "parts": parts, "prompt0_len": prompt0, **ends}


def model_batch(ids: Sequence[int], parts: Sequence[Any]) -> Dict[str, Any]:
    """The forward's inputs: the whole sequence (batch 1, attention everywhere) and every frame's pixel_values /
    image_grid_thw in the sequence's order."""
    import torch

    out: Dict[str, Any] = {"input_ids": torch.tensor([list(ids)], dtype=torch.long),
                           "attention_mask": torch.ones(1, len(ids), dtype=torch.long)}
    for key in ("pixel_values", "image_grid_thw"):
        values = [p[key] for p in parts if key in p and p[key] is not None]
        if values:
            out[key] = torch.cat([torch.as_tensor(v) for v in values], 0)
    return out


def load_png(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


def episode_inputs(episode: Dict[str, Any], processor: Any, load_image: Callable[[Path], Any] = load_png,
                   run_check: bool = True, template_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The episode's training-layout sequence (model_batch; ``template_kwargs``: the chat-template arguments of the
    agent loop, None: TEMPLATE_KWARGS), each turn's position, the eval-layout prompt checks (``run_check`` False: the
    run's record of its prompts is not compared, every turn counted unchecked) and the first prompt's agreement
    between the layouts (always checked)."""
    row, model, result = episode["row"], episode["model"], Path(episode["result"])
    traj = row.get("trajectory") or []
    if not traj:
        raise Mismatch("no recorded turn")
    if "system" not in row:
        raise Mismatch("the row has no system prompt (not a critic-probe row)")
    if model.get("sft_text"):
        raise SystemExit("a run with --sft-text: RL training shows the env's text as it is")
    turns = [(t["user"], load_image(result / t["frame"]), t["answer"]) for t in traj]
    images = [im for _, im, _ in turns]
    if run_check:
        prompts = [encode(processor, conversation(row["system"], turns, t), images[:t + 1], True)[0]
                   for t in range(len(turns))]
        checks = check_prompts(traj, prompts)
    else:
        prompts = [encode(processor, conversation(row["system"], turns, 0), images[:1], True)[0]]
        checks = {"prompt_checked": 0, "prompt_unchecked": len(turns)}
    layout = training_layout(processor, row["system"], turns, [t.get("finish_reason") for t in traj], template_kwargs)
    if layout["input_ids"][:layout["prompt0_len"]] != prompts[0] or layout["prompt0_len"] != len(prompts[0]):
        raise Mismatch("the first prompt differs between the training and the eval layout")
    return {"batch": model_batch(layout["input_ids"], layout["parts"]), "positions": layout["positions"],
            "seq_len": len(layout["input_ids"]), "prompt_lens": [p + 1 for p in layout["positions"]],
            "prompt0_len": layout["prompt0_len"], "eval_prompt_lens": [len(p) for p in prompts] if run_check else None,
            **{k: layout[k] for k in ("eos_appended", "length_cut", "finish_unknown")}, **checks}


# ---------------------------------------------------------------------------
# the targets
# ---------------------------------------------------------------------------
def episode_targets(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per turn: the turn reward (unrounded), G_t (the turn rewards from t to the end, gamma 1), the episode's
    success, the turns left (t to the end, t included), d, the bearing sector, the distance condition
    (absolute_positioning only), phi, from the env's truth at the pose of the turn's observation."""
    traj = row.get("trajectory") or []
    lacking = [t for t, turn in enumerate(traj) if "reward_exact" not in turn or "critic_features" not in turn]
    if lacking:
        raise ValueError(f"episode {row.get('index')}: turns {lacking} lack reward_exact / critic_features "
                         "(not a critic-probe row)")
    rewards = [float(turn["reward_exact"]) for turn in traj]
    returns, acc = [0.0] * len(rewards), 0.0
    for t in reversed(range(len(rewards))):
        acc += rewards[t]
        returns[t] = acc
    out = []
    for t, turn in enumerate(traj):
        f = turn["critic_features"]
        out.append({"turn": t, "reward": rewards[t], "G_t": returns[t], "success": bool(row["success"]),
                    "turns_left": len(traj) - t,
                    "d": None if f is None or f.get("d") is None else int(f["d"]),
                    "bearing": None if f is None else f.get("bearing"),
                    "condition": f.get("condition") if f is not None and row.get("task_type") == AP else None,
                    "phi": None if f is None or f.get("phi") is None else float(f["phi"]),
                    "critic_features": f, "critic_features_error": turn.get("critic_features_error")})
    return out


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------
def shard_dir(out: Path, i: int, n: int) -> Path:
    return Path(out) / f"shard_{i:03d}_of_{n:03d}"


def _atomic_write(path: Path, write: Callable[[Any], None], mode: str = "w") -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, mode) as f:
        write(f)
    os.replace(tmp, path)


def scan_shard(sdir: Path) -> Dict[str, Any]:
    """What a shard folder holds: the episode keys written (chunks whose jsonl exists) and skipped (skipped.jsonl),
    the turns written and the next chunk's number."""
    written, turns, last = set(), 0, -1
    for jl in sorted(Path(sdir).glob("chunk_*.jsonl")):
        last = max(last, int(jl.stem.split("_")[1]))
        rows = _jsonl(jl)
        written |= {r["key"] for r in rows}
        turns += len(rows)
    path = Path(sdir) / "skipped.jsonl"
    skipped = {r["key"] for r in _jsonl(path)} if path.is_file() else set()
    return {"written": written, "skipped": skipped, "turns": turns, "next": last + 1}


def write_chunk(sdir: Path, k: int, arrays: Dict[str, List[np.ndarray]], metas: List[Dict[str, Any]]) -> List[str]:
    """chunk_k.npz (one array per reading, float16; float32 for a reading past float16's range) then chunk_k.jsonl
    (the chunk counts as written once its jsonl exists).  Returns the readings kept as float32."""
    data, wide = {}, []
    for name in READINGS:
        a = np.concatenate(arrays[name]).astype(np.float32)
        if a.size and float(np.abs(a).max()) > FP16_MAX:
            data[name] = a
            wide.append(name)
        else:
            data[name] = a.astype(np.float16)
        if len(data[name]) != len(metas):
            raise RuntimeError(f"chunk {k}: {len(data[name])} rows of {name} for {len(metas)} turns")
    _atomic_write(Path(sdir) / f"chunk_{k:05d}.npz", lambda f: np.savez(f, **data), "wb")
    _atomic_write(Path(sdir) / f"chunk_{k:05d}.jsonl", lambda f: f.write("".join(json.dumps(m) + "\n" for m in metas)))
    return wide


def extract_shard(episodes: Sequence[Dict[str, Any]], sdir: Path, processor: Any,
                  read: Callable[[Any, List[int]], Dict[str, np.ndarray]], manifest: Dict[str, Any],
                  on_mismatch: str = "fail", limit: Optional[int] = None, chunk_episodes: int = CHUNK_EPISODES,
                  load_image: Callable[[Path], Any] = load_png, log: Callable[[str], None] = print,
                  run_check: bool = True, template_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extract one shard's episodes not yet written; ``read(batch, positions)`` gives each reading's [turns, hidden]
    array (the GPU forward in cmd_extract); ``template_kwargs`` the agent loop's chat-template arguments (None:
    TEMPLATE_KWARGS; a shard keeps the manifest's).  Writes chunks, skipped.jsonl and manifest.json; returns the
    counts."""
    sdir = Path(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    have = scan_shard(sdir)
    done, k = have["written"] | have["skipped"], have["next"]
    todo = [e for e in episodes if e["key"] not in done]
    if limit is not None:
        todo = todo[: int(limit)]
    man_path = sdir / "manifest.json"
    old = json.loads(man_path.read_text()) if man_path.is_file() else {}
    if old and old.get("version") != manifest.get("version"):
        raise SystemExit(f"{sdir}: written by {old.get('version')!r}, this is {manifest.get('version')!r}")
    if old and old.get("model") != manifest.get("model"):
        raise SystemExit(f"{sdir}: written from {old.get('model')!r}, this run reads {manifest.get('model')!r}")
    if old and old.get("template_kwargs") != manifest.get("template_kwargs"):  # one layout per shard
        raise SystemExit(f"{sdir}: written with the chat-template arguments {old.get('template_kwargs')!r}, this run "
                         f"gives {manifest.get('template_kwargs')!r}")
    old_counts = old.get("counts") or {}
    counts = {"episodes_written": len(have["written"]), "turns_written": have["turns"],
              "skipped": len(have["skipped"]),
              **{key: int(old_counts.get(key, 0)) for key in EPISODE_COUNTS}}
    wide = dict(old.get("float32_readings") or {})
    man = {**manifest, "shard_episodes": len(episodes), "counts": counts, "float32_readings": wide,
           "status": "running", "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    _atomic_write(man_path, lambda f: f.write(json.dumps(man, indent=1)))
    arrays: Dict[str, List[np.ndarray]] = {name: [] for name in READINGS}
    metas: List[Dict[str, Any]] = []
    in_chunk = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal k, in_chunk, arrays, metas
        if in_chunk:
            for name in write_chunk(sdir, k, arrays, metas):
                wide[f"chunk_{k:05d}"] = wide.get(f"chunk_{k:05d}", []) + [name]
            counts["episodes_written"] += in_chunk
            counts["turns_written"] += len(metas)
            k += 1
        arrays, metas, in_chunk = {name: [] for name in READINGS}, [], 0
        man.update(counts=counts, float32_readings=wide, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
        _atomic_write(man_path, lambda f: f.write(json.dumps(man, indent=1)))

    for n, ep in enumerate(todo):
        row = ep["row"]
        try:
            targets = episode_targets(row)
            inputs = episode_inputs(ep, processor, load_image, run_check, template_kwargs)
        except Mismatch as exc:
            if on_mismatch != "skip":
                flush()
                raise SystemExit(f"episode {ep['key']}: {exc} (--on-mismatch skip records it and goes on)")
            _atomic_append(sdir / "skipped.jsonl", {"key": ep["key"], "reason": str(exc)})
            counts["skipped"] += 1
            continue
        got = read(inputs["batch"], inputs["positions"])
        readings = {name: np.asarray(got[name]) for name in READINGS}
        for name, a in readings.items():  # every reading checked before any of the episode's is kept
            if a.shape[0] != len(targets):
                raise RuntimeError(f"episode {ep['key']}: {a.shape[0]} rows of {name} for {len(targets)} turns")
        bad = {name: int((~np.isfinite(a)).sum()) for name, a in readings.items() if not np.all(np.isfinite(a))}
        if bad:  # never stored; the episodes before it are written
            flush()
            raise SystemExit(f"episode {ep['key']}: non-finite values in {bad}")
        for name, a in readings.items():
            arrays[name].append(a)
        eval_lens = inputs.get("eval_prompt_lens") or [None] * len(targets)
        traj = row.get("trajectory") or []
        for tg, pos, plen, elen in zip(targets, inputs["positions"], inputs["prompt_lens"], eval_lens):
            metas.append({"key": ep["key"], "result": str(Path(ep["result"]).resolve()), "index": row["index"],
                          "sample": row.get("sample"), "task_id": row.get("task_id"), "scene_id": row.get("scene_id"),
                          "task_type": row.get("task_type"), "d_star": row.get("d_star"), **tg,
                          "position": pos, "prompt_tokens": plen, "seq_len": inputs["seq_len"],
                          # the turn's place in RL's response budget (the response starts after the first prompt)
                          "response_offset": pos + 1 - inputs["prompt0_len"], "eval_prompt_tokens": elen,
                          "finish_reason": traj[tg["turn"]].get("finish_reason"),
                          "episode_turns": len(targets)})
        for key in EPISODE_COUNTS:
            counts[key] += int(inputs[key])
        in_chunk += 1
        if in_chunk >= chunk_episodes:
            flush()
            log(f"[repr_probe] {n + 1}/{len(todo)} episodes, {time.time() - t0:.0f} s")
    flush()
    have = scan_shard(sdir)
    remaining = len([e for e in episodes if e["key"] not in have["written"] | have["skipped"]])
    man.update(status="done" if remaining == 0 else "partial", remaining=remaining)
    _atomic_write(man_path, lambda f: f.write(json.dumps(man, indent=1)))
    return man


def _atomic_append(path: Path, record: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


TEMPLATE_KWARGS_SOURCE = ("repr_probe.TEMPLATE_KWARGS (data.apply_chat_template_kwargs of "
                          "data_gen/interiorgs_v10/configs/ppo_template_v10.yaml)")


def template_settings(ppo_config: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
    """The agent loop's chat-template arguments and where they come from: with ``ppo_config`` (a run's ppo.yaml) its
    data.apply_chat_template_kwargs, as GymAgentLoop.init_class reads it (a missing key: {}), else TEMPLATE_KWARGS."""
    if not ppo_config:
        return dict(TEMPLATE_KWARGS), TEMPLATE_KWARGS_SOURCE
    import yaml

    path = Path(ppo_config)
    if not path.is_file():
        raise SystemExit(f"--ppo-config {ppo_config}: no such file")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kwargs = (cfg.get("data") or {}).get("apply_chat_template_kwargs", {})
    kwargs = {} if kwargs is None else kwargs
    if not isinstance(kwargs, dict):
        raise SystemExit(f"--ppo-config {ppo_config}: data.apply_chat_template_kwargs is not a mapping: {kwargs!r}")
    if any(isinstance(v, str) and "${" in v for v in kwargs.values()):
        raise SystemExit(f"--ppo-config {ppo_config}: data.apply_chat_template_kwargs has unresolved interpolations "
                         f"{kwargs!r} (give the resolved config)")
    return dict(kwargs), f"{path.resolve()} data.apply_chat_template_kwargs"


def threads_of(given: Optional[int], default: int) -> int:
    """torch's CPU threads: the option, else OMP_NUM_THREADS / MKL_NUM_THREADS when set, else ``default``."""
    if given:
        return int(given)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        value = os.environ.get(var, "").strip()
        if value.isdigit() and int(value) > 0:
            return int(value)
    return int(default)


def text_model(model: Any) -> Any:  # pragma: no cover - needs the model
    """The decoder stack (``layers`` and the final ``norm``) of a Qwen3-VL model."""
    for path in (("model", "language_model"), ("language_model",), ("model",)):
        m = model
        for name in path:
            m = getattr(m, name, None)
        if m is not None and hasattr(m, "layers") and hasattr(m, "norm"):
            return m
    raise SystemExit("the model has no decoder stack with layers and norm")


NORM_TOLERANCE = 1e-2  # the final norm of the layer-36 reading against the final-norm reading (bf16, rtol and atol)


def check_diagnosis(diag: Dict[str, Any]) -> None:
    """The first forward of an extraction call: HF's hidden_states has NUM_LAYERS + 1 entries, its entries 9 / 18 / 27
    and its last equal the hooks' readings of the same forward, and the final norm applied to the layer-36 reading
    gives the final-norm reading; else the extraction stops."""
    agree = diag.get("hidden_states_agree") or {}
    bad = []
    if diag.get("hidden_states_entries") != NUM_LAYERS + 1:
        bad.append(f"hidden_states has {diag.get('hidden_states_entries')} entries, not {NUM_LAYERS + 1}")
    bad += [f"hidden_states disagree with the hook: {k}" for k in ("layer9", "layer18", "layer27", "final_norm")
            if agree.get(k) is not True]
    if not diag.get("norm_of_layer36_ok"):
        bad.append(f"the final norm of layer 36 is not the final-norm reading (max abs diff "
                   f"{diag.get('norm_of_layer36_max_abs_diff')})")
    if diag.get("hidden_states_error"):
        bad.append(f"the forward with output_hidden_states failed: {diag['hidden_states_error']}")
    if bad:
        raise SystemExit("the readings' check failed: " + "; ".join(bad))


def make_reader(model: Any, device: str, diag: Dict[str, Any]):  # pragma: no cover - needs a GPU
    """read(batch, positions): one forward, the five readings at the positions (float32 numpy).  The first call runs
    its forward with output_hidden_states as well and stops unless check_diagnosis passes on it (diag)."""
    import torch

    lm = text_model(model)
    base = getattr(model, "model", model)
    if len(lm.layers) != NUM_LAYERS:
        raise SystemExit(f"the model has {len(lm.layers)} decoder layers, the probe reads a {NUM_LAYERS}-layer model")

    def run(batch, positions, hidden_states=False):
        inputs = {k: batch[k].to(device) for k in FORWARD_KEYS if k in batch}
        pos = torch.tensor(positions, dtype=torch.long, device=device)
        got: Dict[str, Any] = {}

        def grab(name):
            def hook(_mod, _inp, out):
                h = out[0] if isinstance(out, (tuple, list)) else out
                got[name] = h[0].index_select(0, pos).detach()
            return hook

        handles = [lm.layers[i - 1].register_forward_hook(grab(f"layer{i}")) for i in LAYERS]
        handles.append(lm.norm.register_forward_hook(grab("final_norm")))
        try:
            with torch.no_grad():
                out = base(**inputs, use_cache=False, return_dict=True,
                           **({"output_hidden_states": True} if hidden_states else {}))
        finally:
            for h in handles:
                h.remove()
        return got, out, pos

    def read(batch, positions):
        if diag.get("checked"):
            got, _, _ = run(batch, positions)
        else:  # the same forward: the hooks and HF's hidden_states
            diag["checked"] = True
            try:
                got, out, pos = run(batch, positions, hidden_states=True)
                hs = out.hidden_states
                diag["hidden_states_entries"] = len(hs)
                diag["hidden_states_agree"] = {
                    **{f"layer{i}": bool(torch.equal(hs[i][0].index_select(0, pos), got[f"layer{i}"]))
                       for i in LAYERS[:3]},
                    "final_norm": bool(torch.equal(hs[-1][0].index_select(0, pos), got["final_norm"])),
                    "hidden_states[36]_is_final_norm": bool(torch.equal(hs[NUM_LAYERS][0].index_select(0, pos),
                                                                        got["final_norm"]))}
                with torch.no_grad():
                    renorm = lm.norm(got["layer36"][None])[0]
                diag["norm_of_layer36_max_abs_diff"] = float((renorm.float() - got["final_norm"].float()).abs().max())
                diag["norm_of_layer36_ok"] = bool(torch.allclose(renorm.float(), got["final_norm"].float(),
                                                                 rtol=NORM_TOLERANCE, atol=NORM_TOLERANCE))
            except Exception as exc:  # noqa: BLE001 - recorded, then refused by check_diagnosis
                diag["hidden_states_error"] = f"{type(exc).__name__}: {exc}"
            check_diagnosis(diag)
        return {name: got[name].float().cpu().numpy() for name in READINGS}

    return read


def cmd_extract(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    episodes, info = list_episodes([Path(r) for r in args.results])
    model_path = info["folders"][0]["model"]
    if args.model and os.path.realpath(args.model) != os.path.realpath(model_path):
        raise SystemExit(f"--model {args.model} is not the model the episodes were sampled with ({model_path})")
    model_path = args.model or model_path
    cfg = AutoConfig.from_pretrained(model_path)
    layers = getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers", None)
    if layers != NUM_LAYERS:
        raise SystemExit(f"{model_path}: config num_hidden_layers {layers}, the probe reads a {NUM_LAYERS}-layer model")
    i, n = parse_shard(args.shard)
    template_kwargs, template_source = template_settings(args.ppo_config)
    torch.set_num_threads(threads_of(args.threads, 8))
    processor = AutoProcessor.from_pretrained(model_path)
    eos_id = int(processor.tokenizer.convert_tokens_to_ids(END_OF_TURN))
    gen_path = Path(model_path) / "generation_config.json"
    gen_eos = json.loads(gen_path.read_text()).get("eos_token_id") if gen_path.is_file() else None
    if gen_eos is not None and eos_id not in (gen_eos if isinstance(gen_eos, list) else [gen_eos]):
        raise SystemExit(f"{END_OF_TURN} ({eos_id}) is not among the model's eos_token_id {gen_eos}: the generation "
                         "would not stop on it")
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                                        attn_implementation=args.attn).to("cuda").eval()
    diag: Dict[str, Any] = {}
    manifest = {"version": VERSION, "model": model_path, "readings": list(READINGS), "reading_sources": READING_SOURCES,
                "layers": list(LAYERS), "num_layers": NUM_LAYERS, "hidden_size": cfg.text_config.hidden_size,
                "forward_dtype": "bfloat16", "stored_dtype": "float16", "attn_implementation": args.attn,
                "layout": "training concat (vagen/agent_loop/gym_agent_loop.py GymAgentLoop, mirrored by "
                          "training_layout)", "template_kwargs": template_kwargs,
                "template_kwargs_source": template_source,
                "end_of_turn": {"token": END_OF_TURN, "id": eos_id, "generation_eos": gen_eos},
                "position": "the ids before turn t's answer - 1: the last token of its generation prompt",
                "sft_text_runs": [f["path"] for f in info["folders"] if f.get("sft_text")],  # refused: always []
                "shard": [i, n], "on_mismatch": args.on_mismatch, "run_prompt_check": not args.skip_run_check,
                "max_turns": info["folders"][0]["max_turns"],
                "transformers": transformers.__version__, "torch": torch.__version__,
                "results": info["folders"], "run_errors": info["run_errors"], "episodes_all_shards": info["episodes"],
                "diagnosis": diag}
    sdir = shard_dir(Path(args.out), i, n)
    man = extract_shard(shard_of(episodes, i, n), sdir, processor, make_reader(model, "cuda", diag), manifest,
                        args.on_mismatch, args.limit, int(args.chunk_episodes), run_check=not args.skip_run_check,
                        template_kwargs=template_kwargs)
    print(json.dumps({k: man[k] for k in ("status", "counts", "diagnosis", "shard_episodes")}, indent=1))


# ---------------------------------------------------------------------------
# fit: folds, probes, metrics
# ---------------------------------------------------------------------------
def assign_folds(groups: Sequence[str], k: int, seed: int = SEED) -> np.ndarray:
    """Each group (scene) to one of k folds, deterministic and independent of the rows' order: groups by decreasing
    size (ties by a seeded hash of the name), each to the fold with the fewest rows so far (ties: the lowest)."""
    counts = Counter(str(g) for g in groups)
    if len(counts) < k:
        raise ValueError(f"{len(counts)} groups for {k} folds")
    order = sorted(counts, key=lambda g: (-counts[g], hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()))
    load, fold_of = [0] * k, {}
    for g in order:
        f = min(range(k), key=lambda j: (load[j], j))
        fold_of[g] = f
        load[f] += counts[g]
    return np.array([fold_of[str(g)] for g in groups], dtype=np.int64)


def explained_variance(y: Sequence[float], pred: Sequence[float]) -> Optional[float]:
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if len(y) < 2 or float(np.var(y)) == 0.0:
        return None
    return float(1.0 - np.var(y - pred) / np.var(y))


def auc(y: Sequence[bool], score: Sequence[float]) -> Optional[float]:
    """The area under the ROC curve (Mann-Whitney, ties averaged); None with one class only."""
    y, s = np.asarray(y, dtype=bool), np.asarray(score, dtype=np.float64)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def reading_of(ev: Optional[float]) -> str:
    """The pre-registered reading of the best reading's out-of-fold EV of G_t."""
    if ev is None or not math.isfinite(ev):
        return "undetermined"
    if ev >= READABLE:
        return "linearly readable"
    if ev < NOT_READABLE:
        return "not readable"
    return "weak"


def _standardise(xtr, xte):
    import torch

    mu = xtr.mean(0)
    sd = xtr.std(0, unbiased=False)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    return (xtr - mu) / sd, (xte - mu) / sd


def ridge_path(xtr, ytr, xte, alphas: Optional[Sequence[float]] = None) -> np.ndarray:
    """Ridge predictions [alphas, test rows, targets]: the features standardised with the training rows' statistics,
    the intercept unpenalised (the training mean), loss |y - Xw|^2 + alpha |w|^2; float64, by the eigen-decomposition
    of X'X (or of XX' when there are fewer rows than features)."""
    import torch

    alphas = ALPHAS if alphas is None else alphas
    xtr = torch.as_tensor(xtr, dtype=torch.float64)
    xte = torch.as_tensor(xte, dtype=torch.float64, device=xtr.device)
    a, b = _standardise(xtr, xte)
    y = torch.as_tensor(np.asarray(ytr, dtype=np.float64).reshape(len(a), -1), device=xtr.device)
    ym = y.mean(0)
    yc = y - ym
    n, d = a.shape
    out = []
    if d <= n:
        s, v = torch.linalg.eigh(a.T @ a)
        s = s.clamp_min(0.0)
        z = v.T @ (a.T @ yc)
        bv = b @ v
        for al in alphas:
            out.append(bv @ (z / (s + al)[:, None]) + ym)
    else:
        s, u = torch.linalg.eigh(a @ a.T)
        s = s.clamp_min(0.0)
        z = u.T @ yc
        ku = (b @ a.T) @ u
        for al in alphas:
            out.append(ku @ (z / (s + al)[:, None]) + ym)
    return torch.stack(out).cpu().numpy()


def ridge_cv(x, y, folds: np.ndarray, groups: Sequence[str], alphas: Optional[Sequence[float]] = None,
             seed: int = SEED, device: str = "cpu") -> Dict[str, Any]:
    """Out-of-fold ridge predictions (one column per target), alpha chosen per training fold and target by the pooled
    squared error of an inner grouped split (grid: ALPHAS)."""
    import torch

    alphas = ALPHAS if alphas is None else alphas
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
    y = np.asarray(y, dtype=np.float64).reshape(len(xt), -1)
    groups = np.asarray([str(g) for g in groups])
    oof = np.full(y.shape, np.nan)
    chosen: List[Optional[List[float]]] = []
    for f in range(int(folds.max()) + 1):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        if not len(te):
            chosen.append(None)
            continue
        inner = assign_folds(groups[tr], INNER_FOLDS, seed + 1000 + f)
        sse = np.zeros((len(alphas), y.shape[1]))
        xtr, ytr = xt[torch.as_tensor(tr, device=device)], y[tr]
        for g in range(INNER_FOLDS):
            itr, ite = np.flatnonzero(inner != g), np.flatnonzero(inner == g)
            p = ridge_path(xtr[torch.as_tensor(itr, device=device)], ytr[itr],
                           xtr[torch.as_tensor(ite, device=device)], alphas)
            sse += ((p - ytr[ite][None]) ** 2).sum(1)
        best = sse.argmin(0)
        p = ridge_path(xtr, ytr, xt[torch.as_tensor(te, device=device)], alphas)
        for j, bi in enumerate(best):
            oof[te, j] = p[bi, :, j]
        chosen.append([alphas[bi] for bi in best])
    return {"oof": oof, "alphas": chosen}


def logistic_path(xtr, ytr: np.ndarray, n_classes: int, xte, cs: Optional[Sequence[float]] = None,
                  max_iter: int = LOGISTIC_MAX_ITER) -> np.ndarray:
    """L2 logistic regression class probabilities [cs, test rows, classes] (binary: one logit; else multinomial),
    sklearn's objective 0.5 |W|^2 + C * summed log-loss (intercept unpenalised), standardised features, float64,
    L-BFGS, the C path from the strongest penalty up with warm starts."""
    import torch

    cs = CS if cs is None else cs
    xtr = torch.as_tensor(xtr, dtype=torch.float64)
    xte = torch.as_tensor(xte, dtype=torch.float64, device=xtr.device)
    a, b = _standardise(xtr, xte)
    n, d = a.shape
    y = torch.as_tensor(np.asarray(ytr, dtype=np.int64), device=xtr.device)
    k = 1 if n_classes == 2 else n_classes
    w = torch.zeros(d, k, dtype=torch.float64, device=xtr.device, requires_grad=True)
    c0 = torch.zeros(k, dtype=torch.float64, device=xtr.device, requires_grad=True)
    out: Dict[float, np.ndarray] = {}
    for c in sorted(cs):
        opt = torch.optim.LBFGS([w, c0], lr=1.0, max_iter=int(max_iter), history_size=20, tolerance_grad=1e-7,
                                tolerance_change=1e-11, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = a @ w + c0
            if k == 1:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(z[:, 0], y.double(), reduction="sum")
            else:
                loss = torch.nn.functional.cross_entropy(z, y, reduction="sum")
            loss = (loss + 0.5 * (w ** 2).sum() / float(c)) / n
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            z = b @ w + c0
            p = torch.sigmoid(z[:, 0])
            probs = torch.stack([1 - p, p], 1) if k == 1 else torch.softmax(z, 1)
        out[c] = probs.cpu().numpy()
    return np.stack([out[c] for c in cs])


def logistic_cv(x, y, n_classes: int, folds: np.ndarray, groups: Sequence[str],
                cs: Optional[Sequence[float]] = None, seed: int = SEED, device: str = "cpu",
                max_iter: int = LOGISTIC_MAX_ITER) -> Dict[str, Any]:
    """Out-of-fold class probabilities, C chosen per training fold by the mean log-loss of an inner grouped split
    (grid: CS); and the majority-class baseline (each test fold gets its training fold's most frequent class)."""
    import torch

    cs = CS if cs is None else cs
    xt = torch.as_tensor(np.asarray(x), dtype=torch.float64, device=device)
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray([str(g) for g in groups])
    oof = np.full((len(y), n_classes), np.nan)
    majority = np.full(len(y), -1)
    chosen: List[Optional[float]] = []

    def logloss(p, yy):
        return float(-np.log(np.clip(p[np.arange(len(yy)), yy], 1e-12, 1.0)).sum())

    for f in range(int(folds.max()) + 1):
        tr, te = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        if not len(te):
            chosen.append(None)
            continue
        majority[te] = int(np.bincount(y[tr], minlength=n_classes).argmax())
        inner = assign_folds(groups[tr], INNER_FOLDS, seed + 2000 + f)
        loss = np.zeros(len(cs))
        xtr, ytr = xt[torch.as_tensor(tr, device=device)], y[tr]
        for g in range(INNER_FOLDS):
            itr, ite = np.flatnonzero(inner != g), np.flatnonzero(inner == g)
            p = logistic_path(xtr[torch.as_tensor(itr, device=device)], ytr[itr], n_classes,
                              xtr[torch.as_tensor(ite, device=device)], cs, max_iter)
            loss += [logloss(p[j], ytr[ite]) for j in range(len(cs))]
        best = int(loss.argmin())
        path = [c for c in sorted(cs) if c <= cs[best]]  # warm-started up the C path, as the inner fits
        oof[te] = logistic_path(xtr, ytr, n_classes, xt[torch.as_tensor(te, device=device)], path, max_iter)[-1]
        chosen.append(cs[best])
    return {"oof": oof, "C": chosen, "majority": majority}


def _class_report(y: np.ndarray, res: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    pred = res["oof"].argmax(1)
    return {"n": int(len(y)), "accuracy": _num(np.mean(pred == y)),
            "majority_accuracy": _num(np.mean(res["majority"] == y)),
            "classes": {name: int((y == i).sum()) for i, name in enumerate(names)},
            "C_per_fold": res["C"]}


def mlp_baseline(metas: Sequence[Dict[str, Any]], folds: np.ndarray, max_turns: int, seed: int = SEED,
                 epochs: int = MLP_EPOCHS) -> Dict[str, Any]:
    """The privileged-feature baseline (vagen/feature_critic.py): per training fold the three-layer MLP on the
    29-number encode (success logit: cross-entropy against the episode's outcome; turns left / max_turns: squared
    error), Adam, mini-batches as the RL critic, ``epochs`` passes; then the read-out a*p + b*turns_left + c + e*phi
    (FeatureCritic._fit_readout with readout_phi, ridge 1e-4) on the training fold's G_t; G_t predicted out of fold."""
    import torch

    from vagen import feature_critic as FC

    x = torch.tensor([FC.encode(m["critic_features"], m["turn"], max_turns) for m in metas], dtype=torch.float32)
    s = torch.tensor([float(m["success"]) for m in metas], dtype=torch.float32)
    t = torch.tensor([float(m["turns_left"]) for m in metas], dtype=torch.float32) / float(max_turns)
    g = torch.tensor([float(m["G_t"]) for m in metas], dtype=torch.float32)
    phi = torch.tensor([float(m["phi"]) if m["phi"] is not None else 0.0 for m in metas], dtype=torch.float32)
    oof = np.full(len(metas), np.nan)
    readouts, per_fold = [], []
    for f in range(int(folds.max()) + 1):
        tr = torch.as_tensor(np.flatnonzero(folds != f))
        te = torch.as_tensor(np.flatnonzero(folds == f))
        if not len(te):
            continue
        fc = FC.FeatureCritic({"max_turns": int(max_turns), "readout_phi": True}, int(seed) + f)
        xtr, str_, ttr = x[tr], s[tr], t[tr]
        mb, n = fc.cfg["mini_batch"], len(tr)
        fc.net.train()
        for _ in range(int(epochs)):
            order = torch.randperm(n, generator=fc.gen)
            for k in range(0, n, mb):
                idx = order[k:k + mb]
                bce, mse = fc._losses(xtr[idx], str_[idx], ttr[idx])
                loss = bce + fc.cfg["steps_weight"] * mse
                fc.opt.zero_grad(set_to_none=True)
                loss.backward()
                fc.opt.step()
        fc.net.eval()
        fc._fit_readout(xtr, phi[tr], g[tr])
        _, _, v = fc._predict(x[te], phi[te])
        oof[te.numpy()] = v.numpy()
        readouts.append([_num(r) for r in fc.readout])
        per_fold.append(_num(explained_variance(g[te].numpy(), v.numpy())))
    return {"ev_G_t": _num(explained_variance(g.numpy(), oof)), "ev_per_fold": per_fold,
            "readout_a_b_c_e_per_fold": readouts, "epochs": int(epochs),
            "phi_missing": int(sum(m["phi"] is None for m in metas))}


def turn_family_design(metas: Sequence[Dict[str, Any]], max_turns: int) -> np.ndarray:
    """The turn index one-hot (max_turns slots, a later turn in the last) and the family one-hot."""
    from vagen.feature_critic import FAMILIES

    x = np.zeros((len(metas), int(max_turns) + len(FAMILIES)))
    for i, m in enumerate(metas):
        x[i, min(max(int(m["turn"]), 0), int(max_turns) - 1)] = 1.0
        if m["task_type"] in FAMILIES:
            x[i, int(max_turns) + FAMILIES.index(m["task_type"])] = 1.0
    return x


def probe_reading(x: np.ndarray, metas: Sequence[Dict[str, Any]], folds: np.ndarray, groups: np.ndarray,
                  seed: int = SEED, device: str = "cpu", max_iter: int = LOGISTIC_MAX_ITER) -> Dict[str, Any]:
    """Every probe on one reading's features (rows: the kept turns, in metas' order)."""
    out: Dict[str, Any] = {}
    g = np.array([m["G_t"] for m in metas], dtype=np.float64)
    left = np.array([m["turns_left"] for m in metas], dtype=np.float64)
    r = ridge_cv(x, np.stack([g, left], 1), folds, groups, seed=seed, device=device)
    for j, (name, y) in enumerate((("G_t", g), ("turns_left", left))):
        per = [_num(explained_variance(y[folds == f], r["oof"][folds == f, j])) for f in range(int(folds.max()) + 1)]
        out[name] = {"n": int(len(y)), "ev": _num(explained_variance(y, r["oof"][:, j])), "ev_per_fold": per,
                     "alpha_per_fold": [a[j] if a else None for a in r["alphas"]]}
    out["_oof_G_t"] = r["oof"][:, 0]  # fit_all's reported-only subset (taken out before results.json)
    dm = np.array([m["d"] is not None for m in metas])
    if dm.sum() >= OUTER_FOLDS:
        d = np.array([m["d"] for m in metas if m["d"] is not None], dtype=np.float64)
        rd = ridge_cv(x[dm], d, folds[dm], groups[dm], seed=seed, device=device)
        out["d"] = {"n": int(dm.sum()), "ev": _num(explained_variance(d, rd["oof"][:, 0])),
                    "alpha_per_fold": [a[0] if a else None for a in rd["alphas"]]}
    succ = np.array([int(bool(m["success"])) for m in metas])
    rs = logistic_cv(x, succ, 2, folds, groups, seed=seed, device=device, max_iter=max_iter)
    out["success"] = {**_class_report(succ, rs, ("failure", "success")), "auc": _num(auc(succ, rs["oof"][:, 1]))}
    for name, values, mask in (
            ("bearing", BEARINGS, np.array([m["bearing"] in BEARINGS for m in metas])),
            ("condition", CONDITIONS, np.array([m["task_type"] == AP and m["condition"] in CONDITIONS for m in metas]))):
        if len({m["scene_id"] for m, k in zip(metas, mask) if k}) < OUTER_FOLDS:
            out[name] = {"n": int(mask.sum()), "note": "fewer scenes than folds"}
            continue
        y = np.array([values.index(m[name]) for m, k in zip(metas, mask) if k])
        rc = logistic_cv(x[mask], y, len(values), folds[mask], groups[mask], seed=seed, device=device,
                         max_iter=max_iter)
        out[name] = _class_report(y, rc, values)
    return out


def load_features(features: Path, allow_partial: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray],
                                                                        Dict[str, Any]]:
    """Every shard's turns (sorted by episode key and turn) and readings; the shards must be complete and agree."""
    mans = sorted(Path(features).glob("shard_*/manifest.json"))
    if not mans:
        raise SystemExit(f"{features}: no shard_*/manifest.json")
    loaded = [json.loads(m.read_text()) for m in mans]
    for key in ("version", "model", "readings", "layers", "max_turns"):
        values = {json.dumps(m.get(key)) for m in loaded}
        if len(values) > 1:
            raise SystemExit(f"{features}: the shards differ in {key}: {sorted(values)}")
    ns = {m["shard"][1] for m in loaded}
    have = sorted(m["shard"][0] for m in loaded)
    if len(ns) > 1 or have != list(range(next(iter(ns)))):
        raise SystemExit(f"{features}: shards {have} of {sorted(ns)}: one complete set of shards wanted")
    unfinished = [str(p.parent.name) for p, m in zip(mans, loaded) if m.get("status") != "done"]
    if unfinished and not allow_partial:
        raise SystemExit(f"{features}: shards not done: {unfinished} (--allow-partial to fit what is there)")
    metas: List[Dict[str, Any]] = []
    parts: Dict[str, List[np.ndarray]] = {r: [] for r in READINGS}
    for man in mans:
        for jl in sorted(man.parent.glob("chunk_*.jsonl")):
            rows = _jsonl(jl)
            with np.load(jl.with_suffix(".npz")) as z:
                for r in READINGS:
                    a = z[r]
                    if len(a) != len(rows):
                        raise SystemExit(f"{jl}: {len(rows)} turns, {len(a)} rows of {r}")
                    parts[r].append(a.astype(np.float32))
            metas += rows
    keys = Counter((m["key"], m["turn"]) for m in metas)
    twice = [k for k, n in keys.items() if n > 1]
    if twice:
        raise SystemExit(f"{features}: turns written twice: {twice[:3]}")
    order = sorted(range(len(metas)), key=lambda i: (metas[i]["key"], metas[i]["turn"]))
    arrays = {r: np.concatenate(parts[r])[order] for r in READINGS}
    return [metas[i] for i in order], arrays, {"manifests": loaded, "unfinished": unfinished}


def fit_all(metas: Sequence[Dict[str, Any]], arrays: Dict[str, np.ndarray], max_turns: int, seed: int = SEED,
            device: str = "cpu", max_iter: int = LOGISTIC_MAX_ITER, mlp_epochs: int = MLP_EPOCHS) -> Dict[str, Any]:
    """The probes on every reading, both baselines and the reading of the result."""
    keep = [i for i, m in enumerate(metas) if m["critic_features"] is not None]
    kept = [metas[i] for i in keep]
    groups = np.array([str(m["scene_id"]) for m in kept])
    folds = assign_folds(groups, OUTER_FOLDS, seed)
    res: Dict[str, Any] = {
        "version": VERSION, "seed": int(seed), "device": device,
        "turns": len(metas), "turns_without_features": len(metas) - len(kept), "turns_used": len(kept),
        "episodes": len({m["key"] for m in kept}), "scenes": len(set(groups.tolist())),
        "folds": {"k": OUTER_FOLDS, "inner_k": INNER_FOLDS, "by": "scene_id",
                  "turns_per_fold": np.bincount(folds, minlength=OUTER_FOLDS).tolist(),
                  "scenes_per_fold": [len(set(groups[folds == f].tolist())) for f in range(OUTER_FOLDS)]},
        "grid": {"ridge_alpha": list(ALPHAS), "logistic_C": list(CS)},
        "G_t": {"mean": _num(np.mean([m["G_t"] for m in kept])), "var": _num(np.var([m["G_t"] for m in kept]))},
        "success_rate_turns": _num(np.mean([bool(m["success"]) for m in kept])),
        "readings": {}}
    oof_g: Dict[str, np.ndarray] = {}
    for name in READINGS:
        t0 = time.time()
        res["readings"][name] = probe_reading(arrays[name][keep], kept, folds, groups, seed, device, max_iter)
        oof_g[name] = res["readings"][name].pop("_oof_G_t")
        res["readings"][name]["seconds"] = round(time.time() - t0, 1)
    tf = ridge_cv(turn_family_design(kept, max_turns), np.array([m["G_t"] for m in kept]), folds, groups, seed=seed,
                  device=device)
    res["baselines"] = {
        "privileged_mlp": mlp_baseline(kept, folds, max_turns, seed, mlp_epochs),
        "turn_family_ridge": {"ev_G_t": _num(explained_variance([m["G_t"] for m in kept], tf["oof"][:, 0])),
                              "alpha_per_fold": [a[0] if a else None for a in tf["alphas"]]}}
    evs = {n: res["readings"][n]["G_t"]["ev"] for n in READINGS}
    best = max((n for n in READINGS if evs[n] is not None), key=lambda n: evs[n], default=None)
    res.update(best_reading=best, best_ev_G_t=evs.get(best) if best else None,
               verdict=reading_of(evs.get(best) if best else None),
               rule=("docs/PREREG_PPO2.md gate 0 (frozen): the best reading's out-of-fold EV of G_t >= 0.3 "
                     "'linearly readable', < 0.1 'not readable', otherwise 'weak'; the other targets numbers only"),
               grid_edges=grid_edges(res))
    res["reported_only"] = budget_report(metas, kept, oof_g.get(best) if best else None)
    return res


EVAL_ANSWER_CAP = 384  # evaluation.VllmPolicy's max_tokens per answer


def budget_report(metas: Sequence[Dict[str, Any]], kept: Sequence[Dict[str, Any]], oof_best: Optional[np.ndarray]
                  ) -> Dict[str, Any]:
    """Reported only (not part of the pre-registered reading): the turns past RL's response budget (their first answer
    token at response offset >= the concat budget: RL's sequence never holds them), the answers the eval's token cap
    cut (finish_reason length), over every extracted turn; and the best reading's out-of-fold EV of G_t on the kept
    turns within the budget (the main fit's predictions, restricted)."""
    from data_gen.interiorgs_v10.training_prep import CONCAT_RESPONSE_LENGTH as budget

    def past(m):
        return m.get("response_offset") is not None and int(m["response_offset"]) >= budget

    within = np.array([m.get("response_offset") is not None and not past(m) for m in kept], dtype=bool)
    g = np.array([m["G_t"] for m in kept], dtype=np.float64)
    ev = (explained_variance(g[within], oof_best[within]) if oof_best is not None and within.sum() >= 2 else None)
    return {"note": "reported only, not part of the pre-registered reading",
            "rl_response_budget_tokens": int(budget), "eval_answer_cap_tokens": EVAL_ANSWER_CAP,
            "turns": len(metas), "turns_past_budget": sum(past(m) for m in metas),
            "turns_offset_unknown": sum(m.get("response_offset") is None for m in metas),
            "answers_cut_at_eval_cap": sum(m.get("finish_reason") == "length" for m in metas),
            "answers_finish_unknown": sum(m.get("finish_reason") is None for m in metas),
            "kept_turns_within_budget": int(within.sum()),
            "best_reading_ev_G_t_within_budget": _num(ev)}


def grid_edges(res: Dict[str, Any]) -> List[str]:
    """The probes whose penalty was chosen at an end of its grid in some fold (the grid may be too narrow there)."""
    out = []
    edges = {"alpha_per_fold": (min(ALPHAS), max(ALPHAS), "alpha"), "C_per_fold": (min(CS), max(CS), "C")}
    probes = [(f"{reading} {target}", r) for reading, rs in res["readings"].items() for target, r in rs.items()
              if isinstance(r, dict)]
    probes.append(("baseline turn_family_ridge G_t", res["baselines"]["turn_family_ridge"]))
    for name, r in probes:
        for key, (lo, hi, what) in edges.items():
            values = r.get(key) or []
            hit = sorted({v for v in values if v is not None and (math.isclose(v, lo) or math.isclose(v, hi))})
            if hit:
                folds = [f for f, v in enumerate(values) if v in hit]
                out.append(f"{name}: {what} {', '.join(f'{v:g}' for v in hit)} at the grid's edge in folds {folds}")
    return out


def results_md(res: Dict[str, Any]) -> str:
    f = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    lines = [f"# Gate 0, representation probe ({res['version']})", "",
             f"**Reading: {res['verdict']}** (best reading {res['best_reading']}, out-of-fold EV of G_t "
             f"{f(res['best_ev_G_t'])}).  {res['rule']}.", "",
             f"Turns {res['turns_used']} of {res['turns']} (without the env's truth: {res['turns_without_features']}), "
             f"episodes {res['episodes']}, scenes {res['scenes']}; {res['folds']['k']} folds by scene, turns per fold "
             f"{res['folds']['turns_per_fold']}.", "",
             "| reading | EV G_t | EV d | EV turns left | success AUC | success acc (majority) | bearing acc "
             "(majority) | condition acc (majority) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in res["readings"].items():
        def acc(key):
            v = r.get(key) or {}
            return f"{f(v.get('accuracy'))} ({f(v.get('majority_accuracy'))})" if "accuracy" in v else "—"

        lines.append(f"| {name} | {f(r['G_t']['ev'])} | {f((r.get('d') or {}).get('ev'))} | "
                     f"{f(r['turns_left']['ev'])} | {f(r['success'].get('auc'))} | {acc('success')} | "
                     f"{acc('bearing')} | {acc('condition')} |")
    b = res["baselines"]
    lines += ["", f"Baselines (same turns, same folds), EV of G_t: privileged-feature MLP "
                  f"{f(b['privileged_mlp']['ev_G_t'])}; ridge on turn index + family {f(b['turn_family_ridge']['ev_G_t'])}."]
    edges = res.get("grid_edges") or []
    lines += ["", f"Penalty at the grid's edge (ridge alpha {min(ALPHAS):g}..{max(ALPHAS):g}, logistic C = 1 / alpha): "
                  + ("none." if not edges else "")] + [f"- {e}" for e in edges]
    ro = res.get("reported_only")
    if ro:
        lines += ["", "Reported only (not part of the pre-registered reading):",
                  f"- turns past RL's {ro['rl_response_budget_tokens']}-token response budget: {ro['turns_past_budget']} "
                  f"of {ro['turns']} (offset unknown: {ro['turns_offset_unknown']});",
                  f"- answers cut at the eval's {ro['eval_answer_cap_tokens']}-token cap: {ro['answers_cut_at_eval_cap']} "
                  f"(finish unknown: {ro['answers_finish_unknown']});",
                  f"- best reading ({res['best_reading']}) out-of-fold EV of G_t on the {ro['kept_turns_within_budget']} "
                  f"turns within the budget: {f(ro['best_reading_ev_G_t_within_budget'])}."]
    lines += ["", "Notes: the hidden states are read in the training concat layout (vagen/agent_loop/gym_agent_loop.py) "
                  "at the last token of each turn's generation prompt.  The episodes ran in V4's env (evaluation run "
                  "--critic-probe), so the result rows' progress fields (progress, failed_gates) are its grouped_gated "
                  "progress, not comparable with plain evaluation folders."]
    return "\n".join(lines) + "\n"


def cmd_fit(args: argparse.Namespace) -> None:  # pragma: no cover - the full run (fit_all is tested)
    import torch

    torch.set_num_threads(threads_of(args.threads, 4))
    torch.manual_seed(int(args.seed))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    metas, arrays, info = load_features(Path(args.features), args.allow_partial)
    max_turns = int(info["manifests"][0]["max_turns"])
    t0 = time.time()
    res = fit_all(metas, arrays, max_turns, int(args.seed), device)
    res.update(features=str(Path(args.features).resolve()), model=info["manifests"][0]["model"],
               unfinished_shards=info["unfinished"], seconds=round(time.time() - t0, 1),
               extraction=[{k: m.get(k) for k in ("shard", "counts", "diagnosis", "status")} for m in info["manifests"]])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    (out / "results.md").write_text(results_md(res), encoding="utf-8")
    print(results_md(res))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("extract", help="the hidden states at each turn's value position (one card)")
    x.add_argument("--results", nargs="+", required=True, help="result folders of evaluation run --critic-probe")
    x.add_argument("--model", default=None, help="the HF model (default and must be: the episodes' model.json model)")
    x.add_argument("--out", required=True)
    x.add_argument("--shard", default="0/1", help="i/n: episode k of the listing goes to shard k %% n")
    x.add_argument("--limit", type=int, default=None, help="at most N more episodes in this call (a smoke test)")
    x.add_argument("--on-mismatch", choices=("fail", "skip"), default="fail",
                   help="a rebuilt eval-layout prompt that differs from the run's, or a first prompt that differs "
                        "between the layouts: stop (default) or record the episode in skipped.jsonl, count it and go on")
    x.add_argument("--skip-run-check", action="store_true",
                   help="do not compare the rebuilt eval-layout prompts with the run's record (prompt_tokens, "
                        "prompt_ids_sha256); the first prompt's agreement between the layouts stays; recorded in the "
                        "manifest (default: compared)")
    x.add_argument("--ppo-config", default=None,
                   help="the RL run's ppo.yaml: the agent loop's chat-template arguments from its "
                        "data.apply_chat_template_kwargs (default: TEMPLATE_KWARGS, the PPO template's); recorded in "
                        "the manifest")
    x.add_argument("--chunk-episodes", type=int, default=CHUNK_EPISODES)
    x.add_argument("--attn", default="sdpa", help="attn_implementation of the forward")
    x.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads (the processor); default OMP_NUM_THREADS / MKL_NUM_THREADS, else 8")
    f = sub.add_parser("fit", help="the probes, the baselines and the reading")
    f.add_argument("--features", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--device", default=None, help="default: cuda when available, else cpu")
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--threads", type=int, default=None,
                   help="torch CPU threads; default OMP_NUM_THREADS / MKL_NUM_THREADS, else 4")
    f.add_argument("--allow-partial", action="store_true", help="fit shards that are not done (recorded)")
    args = ap.parse_args(argv)
    {"extract": cmd_extract, "fit": cmd_fit}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

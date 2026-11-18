"""q38 smoke S5: fault injection (docs/q38/DESIGN_ASYNC.md 5.2, S5).  Off by default; for smoke runs only.

Configuration (all under ``vagen.fault_injection``; absent, or ``enable: false`` with nothing else set, is off):

    vagen:
      fault_injection:
        enable: true                       # the master switch; the contract (vagen.q38.contract) refuses it unless
                                           # the rule "vagen.fault_injection.enable" is in vagen.contract.allow_override
                                           # and trainer.experiment_name contains "smoke" (rule fault_injection_config,
                                           # which cannot be overridden)
        env_step_exception:                # (1) env.step raises in the hit episodes, at env turn ``turn``
          {every_n_groups: 4, session: 0, turn: 2}
        generate_exception:                # (2) the generate call of turn ``turn`` raises in the hit episodes
          {every_n_groups: 4, session: 0, turn: 1}
        image_pad_extra:                   # (3) one <|image_pad|> more than the images need, in the hit episodes:
          {every_n_groups: 1, session: 0, turn: 1, where: prompt}
                                           #   where "prompt": the initial prompt's first image run gets one pad more
                                           #     (turn must be 1); verl's vLLM server collapses a run of pads to one
                                           #     (verl/workers/rollout/utils.py qwen2_5_vl_dedup_image_tokens, called
                                           #     at vllm_async_server.py:643), so the rollout itself goes on
                                           #   where "answer": one stray pad at the end of the answer of turn ``turn``
                                           #     (before its stop token), as if the model had sampled it (the 8B drift
                                           #     of the user's rule 91); the next turn's prompt carries it to vLLM
        stuck_warn: true                   # (4) lets vagen.stuck_warn go below the contract's floor (min_s 60 s,
                                           #     factor 1, repeat_s 60 s; rule stuck_warn_below_floor), e.g.
                                           #     vagen.stuck_warn {min_s: 1, factor: 0, repeat_s: 30}; the sampler
                                           #     (vagen/q38/sampler.py) is unchanged

Which episodes are hit: a group (prompt uid) is hit when crc32(uid) % every_n_groups == 0 (1: every group), and in a
hit group the session ``session`` (0 .. rollout.n - 1).  The uid and session_id are the ones AgentLoopWorkerTQ passes
to the agent loop (verl/verl/trainer/ppo/v1/agent_loop_tq.py:128-131, the row's uid); a run with injection on and a
row without them raises.  ``turn`` counts from 1 (the first generate call / the first env.step).  An episode that ends
before that turn is not hit.  Every hit is logged once by the agent loop as a WARNING that starts with MESSAGE_PREFIX.

What each injection should show in a run (DESIGN 5.2 S5; the S5 run notes have the checks):
  (1) the episode ends as done with reward 0 for that turn (termination "env_error"), vagen/env/step_exceptions and
      vagen/env/termination/env_error count it, training goes on (gym_agent_loop_v1._env_turn);
  (2) the episode raises, its group's status is "failure", the q38 sampler raises "q38 sampler: agent loop raised in N
      prompt group(s): <uid> (draw batch B)" (failure_policy raise);
  (3) the trainer stops; the error text is what S5 records for the training chain's rule 91;
  (4) filter/step/stuck_warnings > 0 and "q38 sampler: step S has waited E s (threshold T s)" lines, training goes on.
"""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

SECTION = "vagen.fault_injection"
ENABLE_KEY = SECTION + ".enable"
KINDS = ("env_step_exception", "generate_exception", "image_pad_extra")
KEYS = ("enable",) + KINDS + ("stuck_warn",)
TARGET_KEYS = ("every_n_groups", "session", "turn")
IMAGE_PAD_WHERE = ("prompt", "answer")
MESSAGE_PREFIX = "q38 fault injection"
# the run name must contain this for injection to be allowed (contract rule fault_injection_config)
SMOKE_MARK = "smoke"


class FaultInjected(RuntimeError):
    """The exception an injected fault raises (env.step: caught by the agent loop; generate: not)."""


@dataclass(frozen=True)
class Target:
    """Which episodes an injection hits, and at which turn."""

    every_n_groups: int = 1
    session: int = 0
    turn: int = 1
    where: Optional[str] = None  # image_pad_extra only

    def hits(self, uid: Any, session_id: Any) -> bool:
        return group_hit(uid, self.every_n_groups) and int(session_id) == self.session


@dataclass(frozen=True)
class Plan:
    env_step_exception: Optional[Target] = None
    generate_exception: Optional[Target] = None
    image_pad_extra: Optional[Target] = None
    stuck_warn: bool = False

    def for_episode(self, uid: Any, session_id: Any) -> dict[str, Target]:
        """The injections that hit the episode (uid, session_id): {kind: target}."""
        hit = {}
        for kind in KINDS:
            target = getattr(self, kind)
            if target is not None and target.hits(uid, session_id):
                hit[kind] = target
        return hit

    def describe(self) -> str:
        parts = [f"{k}={getattr(self, k)}" for k in KINDS if getattr(self, k) is not None]
        if self.stuck_warn:
            parts.append("stuck_warn=true")
        return ", ".join(parts) or "nothing"


def group_hit(uid: Any, every_n_groups: int) -> bool:
    """Whether a group is hit: crc32 of its uid (a stable hash, the same in every process) modulo every_n_groups."""
    return zlib.crc32(str(uid).encode("utf-8")) % int(every_n_groups) == 0


def _plain(value: Any) -> Any:
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(value, (DictConfig, ListConfig)):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is always there in the q38 env
        pass
    return value


def section_of(config: Any) -> Any:
    """The raw ``vagen.fault_injection`` value of a config (DictConfig or nested dict), as plain Python; None if
    absent."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(config, DictConfig):
            return _plain(OmegaConf.select(config, SECTION, default=None))
    except ImportError:  # pragma: no cover
        pass
    node = config
    for part in SECTION.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return _plain(node)


def _is_set(value: Any) -> bool:
    """A sub-key counts as set unless it is null, false or an empty mapping."""
    return not (value is None or value is False or (isinstance(value, Mapping) and not value))


def _int(value: Any, name: str, minimum: int, problems: list[str]) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        problems.append(f"{name} must be an integer >= {minimum}, got {value!r}")
        return minimum
    return value


def _target(kind: str, spec: Any, problems: list[str]) -> Optional[Target]:
    name = f"{SECTION}.{kind}"
    if not isinstance(spec, Mapping):
        problems.append(f"{name} must be a mapping {{every_n_groups, session, turn}}, got {spec!r}")
        return None
    allowed = TARGET_KEYS + (("where",) if kind == "image_pad_extra" else ())
    unknown = sorted(set(map(str, spec)) - set(allowed))
    if unknown:
        problems.append(f"{name}: unknown keys {unknown} (allowed: {list(allowed)})")
    every = _int(spec.get("every_n_groups", 1), f"{name}.every_n_groups", 1, problems)
    session = _int(spec.get("session", 0), f"{name}.session", 0, problems)
    turn = _int(spec.get("turn", 1), f"{name}.turn", 1, problems)
    where = None
    if kind == "image_pad_extra":
        where = spec.get("where", "prompt")
        if where not in IMAGE_PAD_WHERE:
            problems.append(f"{name}.where must be one of {list(IMAGE_PAD_WHERE)}, got {where!r}")
        elif where == "prompt" and turn != 1:
            problems.append(f"{name}: where 'prompt' puts the pad into the initial prompt, turn must be 1 (got {turn})")
    return Target(every_n_groups=every, session=session, turn=turn, where=where)


def parse(section: Any) -> tuple[Optional[Plan], list[str]]:
    """(the plan, or None when injection is off; the problems of the section).  Off = the section absent / null, or
    ``enable`` not true; a section with ``enable`` not true but some injection set is a problem (it would be
    silently ignored)."""
    problems: list[str] = []
    if section is None:
        return None, problems
    if not isinstance(section, Mapping):
        return None, [f"{SECTION} must be a mapping, got {section!r}"]
    unknown = sorted(set(map(str, section)) - set(KEYS))
    if unknown:
        problems.append(f"{SECTION}: unknown keys {unknown} (allowed: {list(KEYS)})")
    enable = section.get("enable", False)
    if not isinstance(enable, bool):
        problems.append(f"{ENABLE_KEY} must be true or false, got {enable!r}")
        enable = False
    stuck = section.get("stuck_warn", False)
    if stuck is not None and not isinstance(stuck, bool):
        problems.append(f"{SECTION}.stuck_warn must be true or false, got {stuck!r}")
        stuck = False
    if not enable:
        set_keys = [k for k in KINDS + ("stuck_warn",) if _is_set(section.get(k))]
        if set_keys:
            problems.append(f"{SECTION}: {set_keys} set while enable is not true")
        return None, problems
    targets = {k: _target(k, section[k], problems) if _is_set(section.get(k)) else None for k in KINDS}
    plan = Plan(stuck_warn=bool(stuck), **targets)
    if not any(targets.values()) and not plan.stuck_warn:
        problems.append(f"{ENABLE_KEY} is true but no injection is set")
    return plan, problems


def plan_of(config: Any) -> Optional[Plan]:
    """The run's plan (None when off); raises ValueError when the section has problems."""
    plan, problems = parse(section_of(config))
    if problems:
        raise ValueError("; ".join(problems))
    return plan


def insert_extra_pad_in_prompt(prompt_ids: list[int], image_token_id: int) -> tuple[list[int], int]:
    """The prompt with one <|image_pad|> more in its first image run (inserted right after the run's first pad); and
    the position of the inserted token.  Raises ValueError when the prompt has no image token."""
    try:
        first = prompt_ids.index(image_token_id)
    except ValueError:
        raise ValueError(f"{MESSAGE_PREFIX}: image_pad_extra needs an image in the initial prompt") from None
    return prompt_ids[: first + 1] + [image_token_id] + prompt_ids[first + 1:], first + 1


def insert_stray_pad_in_answer(
    token_ids: list[int], log_probs: list[float], image_token_id: int, stop_ids: set[int], max_tokens: int
) -> tuple[list[int], list[float]]:
    """The answer with one stray <|image_pad|> before its stop token (at the end when it has none).  An answer that
    already has max_tokens tokens loses the token before the stop token (or its last one) to make room.  With
    logprobs (one per token), the pad gets its left neighbour's (only the lengths matter: the episode is not meant
    to be trained on)."""
    ids, lps = list(token_ids), list(log_probs or [])
    with_lps = bool(ids) and len(lps) == len(ids)
    if ids and len(ids) >= max_tokens:
        drop = len(ids) - 2 if ids[-1] in stop_ids and len(ids) >= 2 else len(ids) - 1
        del ids[drop]
        if with_lps:
            del lps[drop]
    pos = len(ids) - 1 if ids and ids[-1] in stop_ids else len(ids)
    ids.insert(pos, int(image_token_id))
    if with_lps:
        lps.insert(pos, float(lps[pos - 1]) if pos > 0 else (float(lps[0]) if lps else -1.0))
    return ids, lps

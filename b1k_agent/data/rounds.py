"""Training-round registry (task spec 7.4): rounds.yaml records every round, so any round's config can
be restored and a round can be rolled back.

    python -m data.rounds show    REGISTRY
    python -m data.rounds new     REGISTRY --spec ROUND_SPEC.yaml [--root DIR]
    python -m data.rounds freeze  REGISTRY --round K [--root DIR]
    python -m data.rounds dev-runs REGISTRY --round K RUN_ID [RUN_ID ...]
    python -m data.rounds decide  REGISTRY --round K keep|rollback [--notes TEXT]
    python -m data.rounds resolve REGISTRY --round K [--own] [--root DIR]   # eval.launch arguments
    python -m data.rounds verify  REGISTRY [--root DIR]

REGISTRY is one YAML file; keep it in the repo (data/rounds.yaml) next to the configs it names, with
artifact paths on the NAS (absolute) or relative to --root (default: the repo).

Round k (7.4): collect under a fixed budget -> label -> train exactly ONE trainable component
(skill_executor | scorer | checker | memory) -> freeze -> Dev evaluation -> keep or rollback.
Round 0 trains nothing: it is the baseline whose Step 6 / 7 runs give the failure distribution and
decision logs. Each round records what was collected (budget, sources), how it was labelled, the
trained component, the agent config (file + dotted overrides; at freezing: the resolved-config hash,
as in run manifests, and the sha256 of its base chain, prompt files and skill catalog), artifact paths
+ sha256, the frozen flag, the Dev run ids and the decision.

Rules enforced here:
  * a new round starts only after the previous one is frozen (and decided, from round 1 on);
  * freezing hashes the config file and every artifact (a directory hashes its sorted file list and
    contents); after that the round's content is immutable and ``verify`` detects changed files;
  * Dev results are recorded only for a frozen round, and a decision needs at least one Dev run;
  * the registry's invariants (Report split, evaluator version, per-episode budget: frozen forever)
    are hashed at creation and checked on every load.

"The config of round k": ``resolve(reg, k)`` returns the config of the latest round j <= k whose
decision is keep (round 0, the baseline, always counts as kept once frozen), so a rolled-back round
resolves to the round before it; ``resolve(reg, k, effective=False)`` returns round k's own config.

Status: written and unit-tested (tests/test_data_diag_rounds.py); no round has been run yet.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import shlex
import sys
from typing import Any, Optional, Sequence

import yaml

from data.schema import LABEL_SOURCES, SCHEMA_VERSION, SchemaError, from_dict, to_dict

TRAINABLE_COMPONENTS = ("skill_executor", "scorer", "checker", "memory")
DECISIONS = ("keep", "rollback")
DEFAULT_INVARIANTS = {
    "report_split": "public_test indices 0-9 (instance ids 301-310); frozen configs only, one rollout each",
    "evaluator": "BEHAVIOR-1K v3.9.3-post2 (a8247a8c), python -m omnigibson.eval.eval, unmodified",
    "episode_budget": "official max_steps; 2026 rule: max_steps x 1 s wall clock, first query uncharged (<= 600 s)",
}


class RoundError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Artifact:
    name: str
    path: str  # absolute, or relative to the registry root
    kind: str = ""  # weights | library | dataset | labels | ...
    sha256: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class RoundConfig:
    path: str  # agent config YAML, relative to the registry root (the repo) or absolute
    overrides: tuple[str, ...] = ()  # dotted key=value overrides (agent/config.py apply_override)
    sha256: Optional[str] = None  # the config file itself
    resolved_sha256: Optional[str] = None  # agent.config.config_hash of the resolved config (as in run manifests)
    files: dict[str, str] = dataclasses.field(default_factory=dict)  # base chain, prompts, skill catalog -> sha256


@dataclasses.dataclass(frozen=True)
class Collection:
    budget: dict[str, Any] = dataclasses.field(default_factory=dict)  # e.g. episodes, env_steps, provider_usd
    sources: tuple[str, ...] = ()  # run ids / dataset ids
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class Labelling:
    label_sources: tuple[str, ...] = ()  # subset of data.schema.LABEL_SOURCES
    files: tuple[Artifact, ...] = ()
    counts: dict[str, Any] = dataclasses.field(default_factory=dict)
    notes: str = ""


@dataclasses.dataclass(frozen=True)
class RoundRecord:
    round: int
    trained_component: Optional[str]  # None only for round 0
    config: RoundConfig
    collection: Collection = dataclasses.field(default_factory=Collection)
    labelling: Labelling = dataclasses.field(default_factory=Labelling)
    artifacts: tuple[Artifact, ...] = ()
    frozen: bool = False
    frozen_at: Optional[str] = None
    dev_run_ids: tuple[str, ...] = ()
    decision: Optional[str] = None
    decision_notes: str = ""
    parent_round: Optional[int] = None  # the effective round this one started from
    created_at: str = ""
    notes: str = ""

    def errors(self) -> list[str]:
        out = []
        if self.round < 0:
            out.append("round must be >= 0")
        if self.round == 0 and self.trained_component is not None:
            out.append("round 0 trains nothing (trained_component must be null)")
        if self.round > 0 and self.trained_component not in TRAINABLE_COMPONENTS:
            out.append(f"round {self.round}: trained_component {self.trained_component!r} must be exactly one of "
                       f"{list(TRAINABLE_COMPONENTS)}")
        bad = [s for s in self.labelling.label_sources if s not in LABEL_SOURCES]
        if bad:
            out.append(f"labelling.label_sources {bad} not in {list(LABEL_SOURCES)}")
        if self.decision is not None and self.decision not in DECISIONS:
            out.append(f"decision {self.decision!r} not in {list(DECISIONS)}")
        if self.decision is not None and not (self.frozen and self.dev_run_ids):
            out.append("a decision needs a frozen round with at least one Dev run")
        if self.round == 0 and self.decision == "rollback":
            out.append("round 0 is the baseline; it cannot be rolled back")
        if self.dev_run_ids and not self.frozen:
            out.append("Dev runs are recorded only for a frozen round")
        if not self.config.path:
            out.append("config.path is empty")
        if self.frozen:
            missing = [a.name for a in self.all_artifacts() if not a.sha256]
            missing += [] if (self.config.sha256 and self.config.resolved_sha256) else ["config"]
            if missing:
                out.append(f"frozen round without sha256 for {missing}")
        names = [a.name for a in self.all_artifacts()]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            out.append(f"duplicate artifact names {dup}")
        return out

    def all_artifacts(self) -> tuple[Artifact, ...]:
        return tuple(self.artifacts) + tuple(self.labelling.files)


@dataclasses.dataclass
class Registry:
    invariants: dict[str, str]
    invariants_sha256: str
    rounds: list[RoundRecord] = dataclasses.field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def get(self, k: int) -> RoundRecord:
        for r in self.rounds:
            if r.round == k:
                return r
        raise RoundError(f"no round {k} (have {[r.round for r in self.rounds]})")

    def _put(self, rec: RoundRecord) -> None:
        self.rounds = [rec if r.round == rec.round else r for r in self.rounds]

    def errors(self) -> list[str]:
        out = []
        if _invariants_hash(self.invariants) != self.invariants_sha256:
            out.append("registry invariants changed since creation "
                       "(Report split, evaluator and budget are frozen forever)")
        for key in DEFAULT_INVARIANTS:
            if not self.invariants.get(key):
                out.append(f"invariant {key!r} missing")
        for i, r in enumerate(self.rounds):
            if r.round != i:
                out.append(f"rounds must be numbered 0, 1, 2, ... (position {i} holds round {r.round})")
            out += [f"round {r.round}: {e}" for e in r.errors()]
        return out


def _invariants_hash(inv: dict) -> str:
    return hashlib.sha256(json.dumps(inv, sort_keys=True).encode("utf-8")).hexdigest()


def new_registry(invariants: Optional[dict] = None) -> Registry:
    inv = dict(invariants or DEFAULT_INVARIANTS)
    return Registry(invariants=inv, invariants_sha256=_invariants_hash(inv))


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------ persistence


def load_registry(path: str) -> Registry:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    try:
        reg = from_dict(Registry, data, validate=False)
    except SchemaError as e:
        raise RoundError(f"{path}: {e}") from None
    problems = reg.errors()
    if problems:
        raise RoundError(f"{path}: " + "; ".join(problems))
    return reg


def save_registry(reg: Registry, path: str) -> None:
    problems = reg.errors()
    if problems:
        raise RoundError("; ".join(problems))
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("# Training-round registry (data/rounds.py). Edit only through `python -m data.rounds`.\n")
        yaml.safe_dump(to_dict(reg), f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------ hashing


def _abs(path: str, root: str) -> str:
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(root, os.path.expanduser(path)))


def sha256_path(path: str) -> str:
    """sha256 of a file, or of a directory as its sorted (relative path, file sha256) list."""
    if os.path.isfile(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if os.path.isdir(path):
        entries = []
        for base, dirs, files in os.walk(path):
            dirs.sort()
            for fn in sorted(files):
                p = os.path.join(base, fn)
                entries.append([os.path.relpath(p, path), sha256_path(p)])
        return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()
    raise RoundError(f"{path} does not exist")


def config_fingerprint(path: str, overrides: Sequence[str], root: str) -> tuple[str, dict[str, str]]:
    """(resolved config hash, {file: sha256}) for an agent config: every file of its ``base:`` chain,
    the prompt files of enabled modules (configs/prompts/<package>/<version>.txt), the skill catalog
    and the repo-relative task metadata snapshots (task prompts). A checkpoint directory is not hashed
    here; list trained weights as artifacts. Not covered: absolute paths into the BEHAVIOR-1K tree
    (pinned by the evaluator invariant) and model ids read from env vars (providers.*.model_env)."""
    from agent.config import REPO_ROOT, config_hash, get, load_config
    from agent.prompts import DEFAULT_PROMPT_ROOT

    try:
        cfg = load_config(path, list(overrides))
    except (OSError, ValueError) as e:
        raise RoundError(f"config {path}: {e}") from None
    paths = [os.path.normpath(os.path.join(REPO_ROOT, rel)) for rel in cfg["_meta"]["config_chain"]]
    for slot, m in (cfg.get("modules") or {}).items():
        if isinstance(m, dict) and m.get("enabled") and m.get("prompt_version"):
            impl = str(m.get("impl", "null"))
            package = impl.split(":", 1)[0] if ":" in impl else slot
            paths.append(os.path.join(DEFAULT_PROMPT_ROOT, package, f"{m['prompt_version']}.txt"))
    catalog = get(cfg, "executor.skill_catalog")
    if catalog:
        paths.append(catalog if os.path.isabs(catalog) else os.path.join(REPO_ROOT, catalog))
    for key in ("catalog_tsv", "tasks_jsonl"):  # repo snapshots that set task prompts / max_steps (agent/task_meta.py)
        rel = get(cfg, f"task_metadata.{key}")
        if rel and not os.path.isabs(rel):
            paths.append(os.path.join(REPO_ROOT, rel))
    files = {}
    for p in paths:
        files[os.path.relpath(p, root)] = sha256_path(p) if os.path.exists(p) else "missing"
    return config_hash(cfg), files


def _hashed(a: Artifact, root: str) -> Artifact:
    digest = sha256_path(_abs(a.path, root))
    if a.sha256 and a.sha256 != digest:
        raise RoundError(f"artifact {a.name} ({a.path}): recorded sha256 {a.sha256[:12]} != current {digest[:12]}")
    return dataclasses.replace(a, sha256=digest)


# ------------------------------------------------------------------------------------ round lifecycle


def round_from_spec(spec: dict) -> RoundRecord:
    """A RoundRecord from a spec dict (round number, frozen state, Dev runs and decision are not taken)."""
    spec = {k: v for k, v in dict(spec).items() if k not in ("round", "frozen", "frozen_at", "dev_run_ids", "decision",
                                                             "decision_notes", "parent_round", "created_at")}
    try:
        return from_dict(RoundRecord, {"round": 0, **spec}, validate=False)
    except SchemaError as e:
        raise RoundError(str(e)) from None


def new_round(reg: Registry, rec: RoundRecord, now: Optional[str] = None) -> RoundRecord:
    """Append ``rec`` as the next round; the previous round must be frozen (and decided, from round 1)."""
    k = len(reg.rounds)
    if k:
        prev = reg.rounds[-1]
        if not prev.frozen:
            raise RoundError(f"round {prev.round} is not frozen; freeze it (and evaluate it on Dev) before round {k}")
        if prev.round > 0 and prev.decision is None:
            raise RoundError(f"round {prev.round} has no keep/rollback decision yet")
    rec = dataclasses.replace(rec, round=k, frozen=False, frozen_at=None, dev_run_ids=(), decision=None,
                              decision_notes="", parent_round=effective_round(reg, k - 1) if k else None,
                              created_at=now or _now())
    problems = rec.errors()
    if problems:
        raise RoundError(f"round {k}: " + "; ".join(problems))
    reg.rounds.append(rec)
    return rec


def freeze_round(reg: Registry, k: int, root: str, now: Optional[str] = None) -> RoundRecord:
    """Hash the config file and every artifact (verifying any hash already recorded) and mark round k frozen."""
    rec = reg.get(k)
    if rec.frozen:
        raise RoundError(f"round {k} is already frozen")
    cfg_path = _abs(rec.config.path, root)
    if not os.path.isfile(cfg_path):
        raise RoundError(f"round {k}: config {cfg_path} does not exist")
    resolved, files = config_fingerprint(cfg_path, rec.config.overrides, root)
    config = dataclasses.replace(rec.config, sha256=sha256_path(cfg_path), resolved_sha256=resolved, files=files)
    if rec.config.sha256 and rec.config.sha256 != config.sha256:
        raise RoundError(f"round {k}: config {rec.config.path} changed since its sha256 was recorded")
    missing = [p for p, digest in files.items() if digest == "missing"]
    if missing:
        raise RoundError(f"round {k}: files the config needs do not exist: {missing}")
    labelling = dataclasses.replace(rec.labelling, files=tuple(_hashed(a, root) for a in rec.labelling.files))
    frozen = dataclasses.replace(rec, config=config, labelling=labelling,
                                 artifacts=tuple(_hashed(a, root) for a in rec.artifacts),
                                 frozen=True, frozen_at=now or _now())
    reg._put(frozen)
    return frozen


def amend_round(reg: Registry, k: int, **changes: Any) -> RoundRecord:
    """Change an unfrozen round's content (collection, labelling, artifacts, config, notes, component)."""
    rec = reg.get(k)
    if rec.frozen:
        raise RoundError(f"round {k} is frozen; its content can no longer change (start a new round instead)")
    allowed = {"trained_component", "config", "collection", "labelling", "artifacts", "notes"}
    bad = set(changes) - allowed
    if bad:
        raise RoundError(f"amend_round cannot change {sorted(bad)}")
    new = dataclasses.replace(rec, **changes)
    problems = new.errors()
    if problems:
        raise RoundError(f"round {k}: " + "; ".join(problems))
    reg._put(new)
    return new


def record_dev_runs(reg: Registry, k: int, run_ids: Sequence[str]) -> RoundRecord:
    rec = reg.get(k)
    if not rec.frozen:
        raise RoundError(f"round {k} is not frozen; Dev evaluation must run the frozen round")
    if rec.decision is not None:
        raise RoundError(f"round {k} is already decided ({rec.decision})")
    new = dataclasses.replace(rec, dev_run_ids=tuple(dict.fromkeys([*rec.dev_run_ids, *map(str, run_ids)])))
    reg._put(new)
    return new


def decide(reg: Registry, k: int, decision: str, notes: str = "") -> RoundRecord:
    rec = reg.get(k)
    if decision not in DECISIONS:
        raise RoundError(f"decision must be one of {list(DECISIONS)}")
    if rec.decision is not None:
        raise RoundError(f"round {k} is already decided ({rec.decision})")
    new = dataclasses.replace(rec, decision=decision, decision_notes=notes)
    problems = new.errors()
    if problems:
        raise RoundError(f"round {k}: " + "; ".join(problems))
    reg._put(new)
    return new


def effective_round(reg: Registry, k: int) -> int:
    """The latest round j <= k whose decision is keep; round 0 counts once frozen."""
    for r in sorted((r for r in reg.rounds if r.round <= k), key=lambda r: -r.round):
        if r.decision == "keep" or (r.round == 0 and r.frozen):
            return r.round
    raise RoundError(f"no kept round <= {k} (round 0 must be frozen first)")


def resolve(reg: Registry, k: int, *, effective: bool = True) -> RoundRecord:
    """The round whose config is "the config of round k" (see the module docstring)."""
    reg.get(k)
    return reg.get(effective_round(reg, k) if effective else k)


def verify_round(reg: Registry, k: int, root: str) -> list[str]:
    """Problems with a frozen round's files: missing, or contents differ from the recorded sha256."""
    rec = reg.get(k)
    if not rec.frozen:
        return [f"round {k} is not frozen"]
    out = []
    try:
        resolved, files = config_fingerprint(_abs(rec.config.path, root), rec.config.overrides, root)
        if resolved != rec.config.resolved_sha256:
            out.append(f"round {k} config: the resolved config of {rec.config.path} changed since freezing")
        for p in sorted(set(files) | set(rec.config.files)):
            if files.get(p) != rec.config.files.get(p):
                out.append(f"round {k} config file {p} changed since freezing")
    except RoundError as e:
        out.append(f"round {k} config: {e}")
    items = [("config", rec.config.path, rec.config.sha256)] + [(a.name, a.path, a.sha256) for a in rec.all_artifacts()]
    for name, path, digest in items:
        try:
            current = sha256_path(_abs(path, root))
        except RoundError as e:
            out.append(f"round {k} {name}: {e}")
            continue
        if current != digest:
            out.append(f"round {k} {name}: {path} changed since freezing ({digest[:12]} -> {current[:12]})")
    return out


def launch_args(rec: RoundRecord, root: str) -> list[str]:
    """``python -m eval.launch`` arguments that select the round's agent config."""
    args = ["--config", _abs(rec.config.path, root)]
    for ov in rec.config.overrides:
        args += ["-o", ov]
    return args


def load_round_config(reg: Registry, k: int, root: str, *, effective: bool = True) -> tuple[dict, RoundRecord]:
    """The resolved agent config dict of round k (after verifying the frozen files) and its round."""
    from agent.config import load_config

    rec = resolve(reg, k, effective=effective)
    problems = verify_round(reg, rec.round, root)
    if problems:
        raise RoundError("; ".join(problems))
    return load_config(_abs(rec.config.path, root), list(rec.config.overrides)), rec


# ------------------------------------------------------------------------------------ CLI


def _show(reg: Registry) -> str:
    lines = ["invariants: " + "; ".join(f"{k}: {v}" for k, v in reg.invariants.items())]
    for r in reg.rounds:
        state = "frozen" if r.frozen else "open"
        lines.append(f"round {r.round}: {r.trained_component or 'no training'} | {state} | "
                     f"dev runs {len(r.dev_run_ids)} | "
                     f"decision {r.decision or '-'} | config {r.config.path} {' '.join(r.config.overrides)} | "
                     f"artifacts {len(r.all_artifacts())}")
    if reg.rounds:
        try:
            lines.append(f"effective round now: {effective_round(reg, reg.rounds[-1].round)}")
        except RoundError as e:
            lines.append(str(e))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    root_default = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def add(name, **kw):
        p = sub.add_parser(name, **kw)
        p.add_argument("registry")
        p.add_argument("--root", default=root_default, help="base for relative config / artifact paths (default: repo)")
        return p

    add("show")
    p = add("new")
    p.add_argument("--spec", required=True,
                   help="YAML with trained_component, config, collection, labelling, artifacts")
    p = add("freeze")
    p.add_argument("--round", type=int, required=True)
    p = add("dev-runs")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("run_ids", nargs="+")
    p = add("decide")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("decision", choices=DECISIONS)
    p.add_argument("--notes", default="")
    p = add("resolve")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--own", action="store_true", help="round k's own config, even if it was rolled back")
    add("verify")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "new" and not os.path.exists(args.registry):
            reg = new_registry()
        else:
            reg = load_registry(args.registry)
        if args.cmd == "show":
            print(_show(reg))
            return 0
        if args.cmd == "verify":
            problems = [p for r in reg.rounds if r.frozen for p in verify_round(reg, r.round, args.root)]
            print("\n".join(problems) or "all frozen rounds match their recorded sha256")
            return 1 if problems else 0
        if args.cmd == "resolve":
            rec = resolve(reg, args.round, effective=not args.own)
            problems = verify_round(reg, rec.round, args.root)
            if problems:
                print("\n".join(problems), file=sys.stderr)
                return 1
            print(f"# round {args.round} -> config of round {rec.round}", file=sys.stderr)
            print(" ".join(shlex.quote(a) for a in launch_args(rec, args.root)))
            return 0
        if args.cmd == "new":
            with open(args.spec, encoding="utf-8") as f:
                rec = new_round(reg, round_from_spec(yaml.safe_load(f) or {}))
            print(f"opened round {rec.round} ({rec.trained_component or 'no training'})", file=sys.stderr)
        elif args.cmd == "freeze":
            freeze_round(reg, args.round, args.root)
        elif args.cmd == "dev-runs":
            record_dev_runs(reg, args.round, args.run_ids)
        elif args.cmd == "decide":
            decide(reg, args.round, args.decision, args.notes)
        save_registry(reg, args.registry)
        return 0
    except (RoundError, SchemaError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

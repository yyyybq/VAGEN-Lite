"""The privileged critic of the per-turn PPO (trainer.critic_source "features"; the user's supplement of 2026-11-13,
gate 1 of docs/PREREG_PPO2.md).  Default "model": the value network as before, nothing here runs.

With "features" no critic worker starts (critic.enable false).  A turn's value comes from a three-layer MLP in the
trainer's driver process, on the env's ground truth at the pose the turn's observation was rendered from
(vagen/envs/active_spatial/critic_features.py; env critic_features on), recorded by the agent loop with the turn
(add_to_record):
  - the remaining minimum number of primitive actions to success (the task's distance table), min(d, D_CAP) / D_CAP,
    and a flag for a pose with no distance (1.0 and flag 1);
  - the bearing sector of the task's object A (front / right / back / left / none, one-hot);
  - the distance condition (absolute_positioning: too_close / ok / too_far; none for the other families, one-hot);
  - the turn's index in the episode (one-hot over max_turns);
  - the task family (one-hot over FAMILIES).
The MLP has two outputs: the logit of the episode's success (binary cross-entropy against the episode's outcome) and
the turns left in the episode, this one included, over max_turns (squared error, weight steps_weight).  The value is
a linear read-out of the two, V = a * sigmoid(logit) + b * turns_left + c (turns_left in turns), with a, b, c refit
after every update by ridge least squares (ridge on a and b) on the buffer's Monte-Carlo returns: the undiscounted
sum of the turn rewards from the turn to the episode's end (the return turn_gae's gamma 1 targets).  Before the first
update the read-out is 0, so every value is 0.

Every update (vagen/ray_trainer.py): values() at every candidate's turn anchors (the anchors of vagen/turn_gae.py,
the first response position of each kept turn) before the update, with the MLP and read-out of the updates before;
then update() on the every-candidate batch (before the group filter): its rows join a buffer of the last
buffer_updates updates, the MLP trains epochs passes over the buffer in shuffled mini-batches (Adam, lr), the
read-out is refit.  lambda, gamma, the estimator, the whitening, the filter and the actor are the package's.
The MLP, Adam, the read-out, the buffer and the shuffling RNG go into the checkpoint (feature_critic.pt); a run
resumes only with that file.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from vagen import turn_gae as TG

SOURCE_KEY = "critic_source"
SOURCES = ("model", "features")
CONFIG_KEY = "feature_critic"
CHECKPOINT_FILE = "feature_critic.pt"
FEATURE_VERSION = 1  # the env's critic_features.VERSION this encoding reads
FAMILIES = ("absolute_positioning", "equidistance", "fov_inclusion", "occlusion_alignment", "projective_relations",
            "screen_occupancy")
BEARINGS = ("front", "right", "back", "left")
CONDITIONS = ("too_close", "ok", "too_far")
D_CAP = 12
DEFAULTS = {"hidden": 128, "lr": 1e-3, "epochs": 4, "mini_batch": 256, "buffer_updates": 8, "steps_weight": 1.0,
            "ridge": 1e-4, "threads": 4}
INT_KEYS = ("hidden", "epochs", "mini_batch", "buffer_updates", "threads", "max_turns")
FLOAT_KEYS = ("lr", "steps_weight", "ridge")


def _get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    getter = getattr(config, "get", None)
    return getter(key, default) if callable(getter) else getattr(config, key, default)


def source(config: Any) -> str:
    """trainer.critic_source of a full trainer config: "model" (default) or "features"."""
    value = _get(_get(config, "trainer"), SOURCE_KEY, None) or "model"
    if value not in SOURCES:
        raise ValueError(f"trainer.{SOURCE_KEY} must be one of {SOURCES}, not {value!r}")
    return str(value)


def settings(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """trainer.feature_critic with the defaults; max_turns (the env's turn limit) has none."""
    raw = dict(raw or {})
    unknown = sorted(set(raw) - set(DEFAULTS) - {"max_turns", "seed"})
    if unknown:
        raise ValueError(f"trainer.{CONFIG_KEY}: unknown settings {unknown}")
    out = dict(DEFAULTS, **{k: v for k, v in raw.items() if k != "seed"})
    if "max_turns" not in out:
        raise ValueError(f"trainer.{CONFIG_KEY}.max_turns (the env's turn limit) is required")
    for key in INT_KEYS:
        v = out[key]
        if isinstance(v, bool) or int(v) != v or int(v) < 1:
            raise ValueError(f"trainer.{CONFIG_KEY}.{key} must be a whole number >= 1, not {v!r}")
        out[key] = int(v)
    for key in FLOAT_KEYS:
        v = out[key]
        if isinstance(v, bool) or not float(v) >= 0 or (key == "lr" and not float(v) > 0):
            raise ValueError(f"trainer.{CONFIG_KEY}.{key} must be a non-negative number (lr positive), not {v!r}")
        out[key] = float(v)
    return out


def check_config(config: Any) -> None:
    """Refuse a features run whose other settings it cannot honour: the per-turn estimator, no critic worker, every
    candidate kept for the critic (the group filter on), none of the value network's diagnostics."""
    algo, trainer = _get(config, "algorithm"), _get(config, "trainer")
    if not TG.enabled(algo):
        raise ValueError(f"trainer.{SOURCE_KEY} features needs algorithm.adv_estimator turn_gae")
    if _get(_get(config, "critic"), "enable", None) is not False:
        raise ValueError(f"trainer.{SOURCE_KEY} features runs without the critic worker: critic.enable must be false")
    if not bool(_get(trainer, "critic_all_candidates", False)):
        raise ValueError(f"trainer.{SOURCE_KEY} features trains on every candidate: trainer.critic_all_candidates")
    if not bool(_get(_get(config, "filter"), "enable", False)):
        raise ValueError(f"trainer.{SOURCE_KEY} features needs the group filter (filter.enable): the every-candidate "
                         "batch is the batch before it")
    for key in ("critic_overfit_steps", "critic_replay_updates"):
        if _get(trainer, key, 0):
            raise ValueError(f"trainer.{SOURCE_KEY} features cannot be combined with trainer.{key}")


def from_config(config: Any) -> Optional["FeatureCritic"]:
    """The trainer's privileged critic, or None (trainer.critic_source model)."""
    if source(config) == "model":
        if _get(_get(config, "trainer"), CONFIG_KEY, None):
            raise ValueError(f"trainer.{CONFIG_KEY} is set but trainer.{SOURCE_KEY} is not features")
        return None
    check_config(config)
    raw = _get(_get(config, "trainer"), CONFIG_KEY, None)
    raw = dict(raw.items()) if raw is not None else {}
    seed = raw.get("seed", _get(_get(config, "data"), "seed", 0))
    return FeatureCritic(raw, int(seed if seed is not None else 0))


# ---------------------------------------------------------------------------
# the agent loop's record and the encoding
# ---------------------------------------------------------------------------
def add_to_record(record: Dict[str, Any], obs_features: Sequence[Optional[Dict[str, Any]]], success: bool) -> None:
    """The agent loop's turn record (vagen/turn_gae.turn_record) with each generated turn's observation truth
    (obs_features: reset's, then each step's) and the episode's outcome."""
    n = len(record["spans"])
    if len(obs_features) < n:
        raise ValueError(f"feature critic: {n} generated turns but {len(obs_features)} observations with features")
    record["features"] = [dict(f) if f is not None else None for f in obs_features[:n]]
    record["success"] = bool(success)


def dim(max_turns: int) -> int:
    return 2 + len(BEARINGS) + 1 + len(CONDITIONS) + 1 + int(max_turns) + len(FAMILIES)


def _one_hot(value: Optional[str], names: Sequence[str], what: str) -> List[float]:
    """names one-hot plus a last "none" slot (value None)."""
    if value is None:
        return [0.0] * len(names) + [1.0]
    if value not in names:
        raise ValueError(f"feature critic: unknown {what} {value!r}")
    return [1.0 if value == n else 0.0 for n in names] + [0.0]


def encode(feat: Optional[Mapping[str, Any]], turn: int, max_turns: int) -> List[float]:
    """One turn's input row (dim(max_turns) numbers)."""
    if feat is None:
        raise ValueError("feature critic: a generated turn has no observation features (the env's critic_features)")
    if int(feat.get("v", -1)) != FEATURE_VERSION:
        raise ValueError(f"feature critic: features version {feat.get('v')!r}, this encoding reads {FEATURE_VERSION}")
    d = feat.get("d")
    head = [min(int(d), D_CAP) / float(D_CAP), 0.0] if d is not None else [1.0, 1.0]
    turns = [0.0] * int(max_turns)
    turns[min(max(int(turn), 0), int(max_turns) - 1)] = 1.0
    family = feat.get("family")
    if family not in FAMILIES:
        raise ValueError(f"feature critic: unknown family {family!r}")
    return (head + _one_hot(feat.get("bearing"), BEARINGS, "bearing")
            + _one_hot(feat.get("condition"), CONDITIONS, "condition")
            + turns + [1.0 if family == f else 0.0 for f in FAMILIES])


def kept_turns(record: Optional[Dict[str, Any]], response_mask_row: torch.Tensor
               ) -> List[Tuple[int, int, float]]:
    """(anchor, the turn's index in the episode, reward) of the turns vagen/turn_gae.py keeps, with the same folding
    of a turn with no response position (checked against turn_gae.turn_anchors)."""
    anchors, _ = TG.turn_anchors(record, response_mask_row)
    mask = [bool(x) for x in response_mask_row.tolist()]
    length = len(mask)
    index = []
    for i, (start, end) in enumerate(record["spans"]):
        s, e = max(0, int(start)), min(length, int(end))
        if e > s and any(mask[p] for p in range(s, e)):
            index.append(i)
    if len(index) != len(anchors):
        raise ValueError("feature critic: the kept turns differ from turn_gae's anchors")
    return [(a, i, r) for (a, _, r), i in zip(anchors, index)]


def episode_rows(record: Optional[Dict[str, Any]], response_mask_row: torch.Tensor, max_turns: int
                 ) -> Dict[str, Any]:
    """One episode's kept turns: anchors, input rows, turns left (this one included), the Monte-Carlo returns (the
    turn rewards summed from the turn on) and the outcome."""
    kept = kept_turns(record, response_mask_row)
    feats = record.get("features")
    if feats is None or len(feats) != len(record["spans"]):
        raise KeyError("feature critic: a turn record without the observation features (env critic_features off?)")
    rewards = [r for _, _, r in kept]
    returns, acc = [0.0] * len(kept), 0.0
    for j in reversed(range(len(kept))):
        acc += rewards[j]
        returns[j] = acc
    return {"anchors": [a for a, _, _ in kept],
            "x": [encode(feats[i], i, max_turns) for _, i, _ in kept],
            "turns_left": [float(len(kept) - j) for j in range(len(kept))],
            "returns": returns, "success": float(bool(record.get("success", False))),
            "d": [feats[i].get("d") for _, i, _ in kept], "near": [bool(feats[i].get("near")) for _, i, _ in kept],
            "table": [bool(feats[i].get("table")) for _, i, _ in kept]}


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    """1 - var(target - pred) / var(target), as vagen/ray_trainer.compute_valid_return_metrics."""
    if target.numel() < 2:
        return float("nan")
    return float(1.0 - torch.var((pred - target).float()) / (torch.var(target.float()) + 1e-5))


# ---------------------------------------------------------------------------
# the critic
# ---------------------------------------------------------------------------
class _MLP(torch.nn.Module):
    def __init__(self, n_in: int, hidden: int) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [:, 0] success logit, [:, 1] turns left / max_turns
        return self.body(x)


class _Threads:
    """torch's intra-op threads for the critic's work, the driver's own setting restored after."""

    def __init__(self, n: int) -> None:
        self.n, self.prev = int(n), None

    def __enter__(self):
        self.prev = torch.get_num_threads()
        torch.set_num_threads(self.n)

    def __exit__(self, *exc):
        torch.set_num_threads(self.prev)


class FeatureCritic:
    def __init__(self, raw: Optional[Mapping[str, Any]], seed: int) -> None:
        self.cfg = settings(raw)
        self.seed = int(seed)
        self.max_turns = self.cfg["max_turns"]
        self.n_in = dim(self.max_turns)
        with torch.random.fork_rng(devices=[]):  # the driver's global RNG is not touched
            torch.manual_seed(self.seed)
            self.net = _MLP(self.n_in, self.cfg["hidden"])
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"])
        self.gen = torch.Generator().manual_seed(self.seed + 1)
        self.readout = [0.0, 0.0, 0.0]  # a, b, c: V = a * p + b * turns_left + c
        self.buffer: List[Dict[str, torch.Tensor]] = []  # per update: x, success, turns_left (over max_turns), returns
        self.updates = 0

    # -- the batch ---------------------------------------------------------
    def batch_rows(self, data: Any) -> List[Dict[str, Any]]:
        records = data.non_tensor_batch.get(TG.RECORD_KEY) if hasattr(data, "non_tensor_batch") else None
        if records is None:
            raise KeyError(f"feature critic needs the agent loop's {TG.RECORD_KEY!r} records (non_tensor_batch)")
        mask = data.batch["response_mask"].detach().cpu()
        if len(records) != mask.shape[0]:
            raise ValueError(f"feature critic: {len(records)} turn records for {mask.shape[0]} samples")
        return [episode_rows(records[i], mask[i], self.max_turns) for i in range(mask.shape[0])]

    def _predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(success probability, turns left in turns, value) of input rows."""
        with torch.no_grad():
            out = self.net(x)
        p = torch.sigmoid(out[:, 0])
        left = out[:, 1] * float(self.max_turns)
        a, b, c = self.readout
        return p, left, a * p + b * left + c

    def values(self, data: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """The value tensor (samples x response positions: each kept turn's value at its anchor, 0 elsewhere) and
        readings of the features."""
        rows = self.batch_rows(data)
        mask = data.batch["response_mask"]
        values = torch.zeros(mask.shape, dtype=torch.float32)
        xs = [x for r in rows for x in r["x"]]
        if xs:
            with _Threads(self.cfg["threads"]):
                _, _, v = self._predict(torch.tensor(xs, dtype=torch.float32))
            k = 0
            for i, r in enumerate(rows):
                for a in r["anchors"]:
                    values[i, a] = float(v[k])
                    k += 1
        ds = [d for r in rows for d in r["d"]]
        n = max(1, len(ds))
        first = [r["d"][0] for r in rows if r["d"] and r["d"][0] is not None]
        return values.to(device=mask.device), {
            "critic/feature/anchors": float(len(ds)),
            "critic/feature/d_missing_rate": sum(d is None for d in ds) / n,
            "critic/feature/d_near_rate": sum(x for r in rows for x in r["near"]) / n,
            "critic/feature/no_table_rate": sum(not x for r in rows for x in r["table"]) / n,
            "critic/feature/first_turn_d_mean": (sum(first) / len(first)) if first else -1.0}

    # -- the update --------------------------------------------------------
    def _losses(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(out[:, 0], s)
        mse = torch.mean((out[:, 1] - t) ** 2)
        return bce, mse

    def _fit_readout(self, x: torch.Tensor, g: torch.Tensor) -> None:
        p, left, _ = self._predict(x)
        phi = torch.stack([p, left, torch.ones_like(p)], dim=1).double()
        reg = torch.diag(torch.tensor([self.cfg["ridge"], self.cfg["ridge"], 0.0], dtype=torch.float64)) * len(g)
        w = torch.linalg.solve(phi.T @ phi + reg + 1e-9 * torch.eye(3, dtype=torch.float64), phi.T @ g.double())
        self.readout = [float(w[0]), float(w[1]), float(w[2])]

    def update(self, data: Any) -> Dict[str, float]:
        """Train on an update's every-candidate batch (before the group filter); readings with the critic before the
        update are on this batch only (out of sample), the others on the buffer after."""
        t0 = time.time()
        rows = [r for r in self.batch_rows(data) if r["x"]]
        if not rows:
            raise RuntimeError("feature critic: no turn to train on (no candidate has a kept turn)")
        x = torch.tensor([x for r in rows for x in r["x"]], dtype=torch.float32)
        s = torch.tensor([r["success"] for r in rows for _ in r["x"]], dtype=torch.float32)
        t = torch.tensor([v for r in rows for v in r["turns_left"]], dtype=torch.float32) / float(self.max_turns)
        g = torch.tensor([v for r in rows for v in r["returns"]], dtype=torch.float32)
        m: Dict[str, float] = {}
        with _Threads(self.cfg["threads"]):
            with torch.no_grad():
                bce0, mse0 = self._losses(x, s, t)
            p0, _, v0 = self._predict(x)
            m.update({"critic/feature/bce_before": float(bce0), "critic/feature/steps_mse_before": float(mse0),
                      "critic/feature/success_acc_before": float(((p0 > 0.5).float() == s).float().mean()),
                      "critic/feature/ev_mc_before": explained_variance(v0, g)})
            self.buffer.append({"x": x, "s": s, "t": t, "g": g})
            del self.buffer[:-self.cfg["buffer_updates"]]
            bx = torch.cat([b["x"] for b in self.buffer])
            bs = torch.cat([b["s"] for b in self.buffer])
            bt = torch.cat([b["t"] for b in self.buffer])
            bg = torch.cat([b["g"] for b in self.buffer])
            n, mb = len(bx), self.cfg["mini_batch"]
            self.net.train()
            for _ in range(self.cfg["epochs"]):
                order = torch.randperm(n, generator=self.gen)
                for k in range(0, n, mb):
                    idx = order[k:k + mb]
                    bce, mse = self._losses(bx[idx], bs[idx], bt[idx])
                    loss = bce + self.cfg["steps_weight"] * mse
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    self.opt.step()
            self.net.eval()
            self._fit_readout(bx, bg)
            with torch.no_grad():
                bce1, mse1 = self._losses(bx, bs, bt)
            _, _, v1 = self._predict(bx)
        self.updates += 1
        a, b, c = self.readout
        m.update({"critic/feature/bce_after": float(bce1), "critic/feature/steps_mse_after": float(mse1),
                  "critic/feature/ev_mc_buffer_after": explained_variance(v1, bg),
                  "critic/feature/readout_a": a, "critic/feature/readout_b": b, "critic/feature/readout_c": c,
                  "critic/feature/rows_new": float(len(x)), "critic/feature/buffer_rows": float(n),
                  "critic/feature/buffer_updates": float(len(self.buffer)),
                  "critic/feature/success_rate_new": float(s.mean()),
                  "critic/feature/updates": float(self.updates),
                  "critic/feature/seconds": time.time() - t0})
        return m

    # -- the checkpoint ----------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"feature_version": FEATURE_VERSION, "cfg": dict(self.cfg), "seed": self.seed, "n_in": self.n_in,
                "net": self.net.state_dict(), "opt": self.opt.state_dict(), "readout": list(self.readout),
                "buffer": [dict(b) for b in self.buffer], "updates": self.updates, "gen": self.gen.get_state()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for key, want in (("feature_version", FEATURE_VERSION), ("n_in", self.n_in), ("cfg", dict(self.cfg))):
            if state.get(key) != want:
                raise ValueError(f"feature critic checkpoint: {key} {state.get(key)!r}, this run has {want!r}")
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.readout = [float(v) for v in state["readout"]]
        self.buffer = [dict(b) for b in state["buffer"]]
        self.updates = int(state["updates"])
        self.gen.set_state(state["gen"])
        self.net.eval()

    def save(self, path: str) -> None:
        import os

        tmp = path + ".tmp"
        torch.save(self.state_dict(), tmp)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))

"""RoboCasa task lists and horizon lookup.

Atomic / composite "seen" names match the RoboCasa-365 seen split used
for VLA evaluation. Horizons are conservative episode-length fallbacks
when a dataset or env does not advertise a horizon.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

ATOMIC_SEEN: List[str] = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
]

COMPOSITE_SEEN: List[str] = [
    "DeliverStraw",
    "GetToastedBread",
    "KettleBoiling",
    "LoadDishwasher",
    "PackIdenticalLunches",
    "PreSoakPan",
    "PrepareCoffee",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
]

TASK_SETS = {
    "atomic_seen": ATOMIC_SEEN,
    "composite_seen": COMPOSITE_SEEN,
    "seen": ATOMIC_SEEN + COMPOSITE_SEEN,
    "all_seen": ATOMIC_SEEN + COMPOSITE_SEEN,
}

DEFAULT_HORIZON = 400

HORIZON_FALLBACKS = {
    "CloseBlenderLid": 200,
    "CloseFridge": 250,
    "CloseToasterOvenDoor": 200,
    "CoffeeSetupMug": 300,
    "NavigateKitchen": 200,
    "OpenCabinet": 250,
    "OpenDrawer": 250,
    "OpenStandMixerHead": 200,
    "PickPlaceCounterToCabinet": 400,
    "PickPlaceCounterToStove": 400,
    "PickPlaceDrawerToCounter": 400,
    "PickPlaceSinkToCounter": 400,
    "PickPlaceToasterToCounter": 400,
    "SlideDishwasherRack": 250,
    "TurnOffStove": 150,
    "TurnOnElectricKettle": 150,
    "TurnOnMicrowave": 150,
    "TurnOnSinkFaucet": 150,
    "DeliverStraw": 600,
    "GetToastedBread": 700,
    "KettleBoiling": 600,
    "LoadDishwasher": 800,
    "PackIdenticalLunches": 800,
    "PreSoakPan": 700,
    "PrepareCoffee": 700,
    "RinseSinkBasin": 600,
    "ScrubCuttingBoard": 600,
    "SearingMeat": 700,
}


def resolve_tasks(
    task: Optional[Union[str, Sequence[str]]] = None,
    task_set: Optional[str] = None,
) -> List[str]:
    """Resolve a task name, comma-list, or named set into a task list.

    Args:
        task: Single task, comma-separated tasks, or a sequence of names.
        task_set: Named set: atomic_seen, composite_seen, seen, all_seen.

    Returns:
        Deduplicated task names in request order.

    Raises:
        ValueError: if neither argument is given or a name is unknown.
    """
    names: List[str] = []
    if task_set:
        key = str(task_set).strip().lower()
        if key not in TASK_SETS:
            raise ValueError(
                f"Unknown task_set {task_set!r}. Valid: {sorted(TASK_SETS)}"
            )
        names.extend(TASK_SETS[key])
    if task:
        if isinstance(task, str):
            parts = [p.strip() for p in task.split(",") if p.strip()]
        else:
            parts = [str(p).strip() for p in task if str(p).strip()]
        names.extend(parts)
    # Deduplicate while preserving order.
    seen = set()
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    if not out:
        raise ValueError("Provide --task and/or --task-set to select RoboCasa tasks.")
    return out


def get_horizon(task: str, override: Optional[int] = None) -> int:
    """Return the episode horizon for ``task``."""
    if override is not None and int(override) > 0:
        return int(override)
    return int(HORIZON_FALLBACKS.get(task, DEFAULT_HORIZON))

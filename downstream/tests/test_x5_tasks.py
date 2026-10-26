"""X5 trial order (downstream/x/x5_tasks.py): registry minus target50, sorted, shuffled with seed 0, deterministic."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
import x5_tasks as X  # noqa: E402


def test_order_leaves_out_the_training_tasks_and_is_deterministic():
    registry = ["OpenDrawer", "CloseDrawer", "ArrangeTea", "TurnOnToaster", "WashFruitColander", "MakeTea"]
    target = ["OpenDrawer", "ArrangeTea", "WashFruitColander"]
    a, b = X.order(registry, target), X.order(list(reversed(registry)), target)
    assert a == b and set(a) == {"CloseDrawer", "TurnOnToaster", "MakeTea"}
    expected = sorted(["CloseDrawer", "TurnOnToaster", "MakeTea"])
    random.Random(0).shuffle(expected)
    assert a == expected

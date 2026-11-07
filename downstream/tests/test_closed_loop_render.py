"""closed_loop's render fix (2026-11-07, RENDER_CAUSE_20261107; the model line agreed): refresh_render calls
n_env.fresh_render_context on the live sim and takes the first observation again from the simulator; the switch
CLOSED_LOOP_RENDER_FIX=0 turns it off."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT.parent))
import closed_loop  # noqa: E402
from downstream.n import n_env  # noqa: E402


def fake_env(calls):
    sim = SimpleNamespace(name="sim")

    class Robosuite:
        def __init__(self):
            self.sim = sim

        def _get_observations(self, force_update=False):
            calls.append(("observations", force_update))
            return {"raw": 1}

    class Wrapper:
        def __init__(self):
            self.env = Robosuite()

        def get_observation(self, raw):
            calls.append(("map", raw))
            return {"mapped": raw}

    return SimpleNamespace(unwrapped=Wrapper()), sim


def test_refresh_render_fixes_then_observes_again(monkeypatch):
    calls = []
    env, sim = fake_env(calls)
    monkeypatch.setattr(n_env, "fresh_render_context", lambda s: calls.append(("fix", s is sim)))
    obs = closed_loop.refresh_render(env)
    assert obs == {"mapped": {"raw": 1}}
    assert calls == [("fix", True), ("observations", True), ("map", {"raw": 1})]  # the fix before the new frame


def test_switch_reads_the_environment(monkeypatch):
    monkeypatch.delenv("CLOSED_LOOP_RENDER_FIX", raising=False)
    assert importlib.reload(closed_loop).RENDER_FIX is True
    monkeypatch.setenv("CLOSED_LOOP_RENDER_FIX", "0")
    assert importlib.reload(closed_loop).RENDER_FIX is False
    monkeypatch.delenv("CLOSED_LOOP_RENDER_FIX")
    importlib.reload(closed_loop)

"""Round 12 freeze (the user's 3.1.2 of 2026-10-22): the env's lenient reading (config lenient_parse, default off) of
an answer the strict reading finds no action in: spaces instead of underscores, the first complete action block, a
missing opening tag.  Every answer the strict reading executes is read exactly as before."""

from types import SimpleNamespace

import pytest

from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.active_spatial.utils import ACTION_SPACE_PRESETS, lenient_actions, parse_actions

STRAFE = set(ACTION_SPACE_PRESETS["strafe"]) - {"done"}


def parse(response, lenient):
    fake = SimpleNamespace(_allowed_actions=STRAFE, config=SimpleNamespace(lenient_parse=lenient))
    return ActiveSpatialEnv._default_parse_func(fake, response, action_sep="|", max_actions=5)


READ_STRICTLY = [
    "<action>move_forward|turn_left|</action>",
    "<action>move_left|</action></details>",  # the strict reading takes the first block and ignores the rest
    "<action>turn_right|</action><action></action>",
    "action>move_backward|</action>",
    "<think>look</think><action>Move_Forward|</action>",
    "<think>the chair is left</think>turn_left|</action>",  # bare actions after </think>
]
RESCUED = [
    ("<action>turn left|move forward|</action>", "spaces", ["turn_left", "move_forward"]),
    ("<action>move_forward|<action>turn_right|</action>", "first_block", ["turn_right"]),
    ("move_forward|turn_left|</action>", "missing_opening_tag", ["move_forward", "turn_left"]),
    ("move right|</action>", "missing_opening_tag", ["move_right"]),
]
NOT_READ = [
    "<action></action>",
    "</action>",
    "<action>action1|action2|</action>",
    "<action>move_back|</action>",
    "<action>turn left|look_up|</action>",  # look_up is not in the strafe action space
    "<action>move_forward|",  # cut off before the closing tag
    "I will move forward.",
]


@pytest.mark.parametrize("response", READ_STRICTLY)
def test_strict_readings_unchanged(response):
    strict = ActiveSpatialEnv._strict_parse_func(SimpleNamespace(_allowed_actions=STRAFE), response,
                                                 action_sep="|", max_actions=5)
    assert strict["actions"]
    assert parse(response, False) == strict
    assert parse(response, True) == strict  # the lenient reading only adds answers the strict one cannot read


@pytest.mark.parametrize("response,rule,actions", RESCUED)
def test_lenient_reads(response, rule, actions):
    assert parse(response, False)["actions"] == []
    out = parse(response, True)
    assert (out["actions"], out["lenient_read"], out["format_correct"], out["parse_error"]) == (actions, rule, True, "")
    assert lenient_actions(response, "|", STRAFE) == (rule, actions)


@pytest.mark.parametrize("response", NOT_READ)
def test_still_unreadable(response):
    assert lenient_actions(response, "|", STRAFE) is None
    assert parse(response, True) == parse(response, False)


def test_parse_actions_spaces_only_when_lenient():
    assert parse_actions("turn left|move_forward|") == (False, [])
    ok, acts = parse_actions("turn  left|move_forward|", lenient=True)
    assert ok and [a.name for a in acts] == ["turn_left", "move_forward"]


def test_env_switch():
    assert ActiveSpatialEnvConfig().lenient_parse is False
    for lenient in (False, True):
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(render_backend=None, enable_collision_detection=False,
                                                      action_space="strafe", enable_explicit_done=False,
                                                      lenient_parse=lenient))
        out = env._default_parse_func("<action>turn left|</action>", action_sep="|", max_actions=5)
        assert out["actions"] == (["turn_left"] if lenient else [])

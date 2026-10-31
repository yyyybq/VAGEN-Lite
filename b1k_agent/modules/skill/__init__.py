"""M-skill: skill catalogs and skill-segment data for the executor. Not a loop slot. See README.md.

This package will hold code that turns skill-level data into something the executor can use: skill catalogs
(configs/skills/*.yaml, read by executor.load_skill_catalog), skill-segment extraction into data/schema.py
SkillSegment records, and the evaluation of a skill-trained executor. Nothing here is called by agent/loop.py;
skills reach the planner only through EpisodeContext.skills, and only those marked available.

Precondition (task spec, Step 5): the skill-instruction diagnosis (scripts/skill_probe.py) shows that the executor
responds to sub-task instructions, or a minimal skill fine-tune is decided. Neither has happened, so there is no
implementation yet.

``IMPLS`` is empty on purpose: agent/registry.py builds only the nine slots of agent.interfaces.SLOT_PROTOCOLS, and a
"skill:<impl>" reference would fail with a clear KeyError until an implementation for some slot exists here.
"""

IMPLS: dict = {}

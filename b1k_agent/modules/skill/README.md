# M-skill: skill catalogs, skill-segment data, skill executors

Not a loop slot: there is no `skill` entry in `agent.interfaces.SLOT_PROTOCOLS`, and `modules/skill/__init__.py` has
`IMPLS = {}`.

## Purpose

Make the executor's skills explicit and, if the executor can be made to follow skill-level instructions, provide the
data and evaluation for it:

- skill catalogs (`configs/skills/<executor>.yaml`): which skills the CURRENT executor can perform, with instruction
  templates, inputs, end conditions, known failures, and the `available` / `verified` flags;
- skill-segment data (`data/schema.py` `SkillSegment`): source trajectory and data version, task / skill instruction,
  raw start and end indices, synchronised observation and action references, completion label and its source,
  episode outcome, split and round; segments are split by source trajectory;
- later, the separate evaluation of a skill-trained executor.

## Status

Five states: planned / code written / interface-verified (mock tests) / simulator-run / full-eval results.

| part | state |
|---|---|
| skill-trained executor, segment extraction from demos | planned |
| `configs/skills/pi05_turning_on_radio.yaml` | code written: only `__full_task__` is available; sub-skills are listed as unavailable and unverified |
| `executor.load_skill_catalog` (catalog -> `EpisodeContext.skills`) | interface-verified (mock tests) |
| `data/schema.py` `SkillSegment` | code written and unit-tested; no real data converted |
| `scripts/skill_probe.py` (Step-5 diagnosis) | code written, tested against a fake executor server; not run on 105 |

## Source paper

ManiUnit, arXiv 2610.12089 (skill dataset and benchmark: skill-segment training and separate skill evaluation), and
MobiAgent, arXiv 2610.03476 (skill data curation). **Not verified by us**: arXiv was unreachable from the environment
that wrote this README; `docs/PAPERS.md` records what search snippets and code repositories showed.

## Interface

No protocol of its own. Skills reach the loop as `EpisodeContext.skills: tuple[SkillSpec, ...]`
(`agent/interfaces.py`): `name, instruction_template, description, inputs, ends_when, known_failures, available,
verified, evidence`. Only `available` skills reach M-plan's prompt; instruction-type recovery needs `verified`.

## When the loop calls it

Never directly. `agent/server.py::build_policy` loads the catalog named by `executor.skill_catalog` once per run, and
each episode gets the skills whose `tasks` patterns match its task.

## Implementations

None (`IMPLS = {}`).

## Config

```yaml
executor:
  skill_catalog: configs/skills/pi05_turning_on_radio.yaml
# catalog entry
skills:
  - name: press_button
    tasks: ["*"]
    instruction_template: "Press the power button of the {object}."
    inputs: [object]
    ends_when: the object visibly turns on
    available: false        # true only after the Step-5 diagnosis
    verified: false
    evidence: "path to the skill_probe report and the Dev episodes"
```

## Known failure modes

- No released executor is trained on skill-level language (`docs/PROTOCOL.md` section 12): a policy trained only on
  full-task prompts usually ignores a changed instruction, so a catalog skill must not be marked available on hope.
- Instruction wording that differs from the training annotations (the 2026 demos annotate 31 skill types).
- Skill end conditions that are not visible need M-check to be detected.

## Enable condition

Only after the Step-5 diagnosis (`scripts/skill_probe.py`, then full Dev episodes) shows that the executor responds
to skill instructions, or after a minimal skill fine-tune is decided (recorded as an extra training contribution).

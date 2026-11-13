# R6: Tasks, instances and dataset for the BEHAVIOR-1K 2026 Challenge

**Scope.** This report covers the 100 challenge tasks and their per-task metadata, which tasks carry over from 2025 and which are new, the instance-ID conventions, the `2026-challenge-demos` dataset (format, state and action layout, cameras, depth, annotations, size), and a proposed set of 5 DEV tasks.

**Sources.** Everything here was read only, and nothing in either repo was modified:
- `B1K` = `/home/user/stanfordvl/behavior-1k` at tag `v3.9.3-post2` (`a8247a8`). The other refs were compared with `git diff`.
- `OPENPI` = `/home/user/wensi-ai/openpi` (`0cc8e35`).
- Two GitHub fetches:
  - `StanfordVL/BEHAVIOR-1K` tag `v3.7.2`, file `OmniGibson/omnigibson/learning/utils/eval_utils.py` (the 2025 code).
  - `StanfordVL/b1k-baselines` `main` @ `59cee35`, file `tutorials/dataset.ipynb` (the 2025 dataset tutorial, which includes a real annotation JSON).
- WebSearch snippets.

**What could not be accessed:**
- huggingface.co is blocked, so `behavior-1k/2026-challenge-demos`, `.../zipped-datasets` and `.../2026-challenge-task-instances` were unreachable.
- The GitHub repo `wensi-ai/2026-challenge-task-instances` returned 403, and `add_repo` reported "you don't have access".
- As a result, the canonical `B100_task_misc.csv`, `available_tasks.yaml`, `task.jsonl` and `meta/tasks.jsonl` were not read. Every place that depends on them says NOT FOUND or "inferred".

**Status legend:**
- **[code]**: verified in code.
- **[docs]**: stated in the docs. `B1K/docs/challenge/*.md` are the official pages.
- **[derived]**: computed by me from repo files using the official `bddl` grounding code. The scripts are listed in the appendix.
- **[snippet]**: a web-search snippet.
- **[inferred]**: my inference.

---

## 0. Summary of the key findings

1. **The task list and order.**
   - `B1K/docs/challenge/task_data.json` holds exactly 100 tasks. Entries 0–49 are the 2025 tasks, copied verbatim. Entries 50–99 are the new 2026 tasks, sorted by their index in `B100_task_misc.csv`.
     - Sources: `docs/gen_2026_task_data.py:3-6, 88-110` [code]; `task_data.json` [derived].
   - The 2025 order 0–49 is identical to the 2025 `TASK_NAMES_TO_INDICES` (`v3.7.2:OmniGibson/omnigibson/learning/utils/eval_utils.py:185-241`) [code, fetched].
2. **The runtime task index.**
   - At eval time, the index comes from `B100_task_misc.csv` (columns `Task ID`, `Task`, `Rooms to inlcude` [sic]) under `$OMNIGIBSON_DATA_PATH/2026-challenge-task-instances/metadata/` (`OmniGibson/omnigibson/eval/utils/eval_utils.py:60-77`) [code].
   - The evaluator puts `task_id` into every observation sent to the policy (`evaluator.py:591`) [code].
   - The CSV itself was NOT FOUND locally. I infer that the CSV index equals the `task_data.json` position: verified for 0–49, inferred for 50–99.
3. **Instances.**
   - 300 training instances are sampled per task, plus seed instance 0 (`docs/behavior_components/task_sampling.md:145-150, 179-183`) [docs].
   - The test instance IDs are `301..340`. Public is 301–320 and hidden is 321–340 (`eval_utils.py:18-21`, `evaluator.py:73-87`) [code].
   - Files live at `2026-challenge-task-instances/{scenes | scene_test/public | scene_test/private}/<scene>/json/<scene>_task_<task>_instances/<scene>_task_<task>_0_<id>_template-tro_state.json` (`asset_utils.py:125-146`, `evaluator.py:464-476`, `behavior_task.py:181`) [code].
   - Reported results use public indices 0–9, i.e. IDs 301–310 (`docs/challenge/evaluation.md:282`) [docs].
4. **The default timeout.**
   - It is `int(1.5 × mean human demo length in frames)`, read from `task.jsonl` (`evaluator.py:281-286`, `score_utils.py:25-46, 132-180`) [code].
   - At 30 Hz this is about **45 × (mean demo seconds)** [inferred].
   - The official per-task demo-duration plot (`docs/assets/challenge_2026/per_task_demo_duration.png`) gives the means. I measured them pixel-accurately [derived].
     - For the 50 carry-over tasks they match `task_data.json` `duration` within about 1 s.
     - **For the 50 new tasks, `task_data.json` durations differ from the plot by up to 166 s.**
   - Anchor: the longest task, assembling_gift_baskets, has a plot mean of 868.4 s. That gives 868.4 × 45 ≈ 39,078 steps ≈ 10.85 h at 1 s/step, which matches "**10.85 hours** of episode timeout for the longest task" (`origin/main:docs/challenge/updates.md:15`) [docs].
5. **Scoring partial credit (q_score).**
   - q_score = 1.0 on full success. Otherwise it is the max over grounded goal options of (literals **not satisfied at start but satisfied at end**) / (option length) (`OmniGibson/omnigibson/metrics/task_metric.py:6-28`) [code].
   - Literals already true at reset are never credited, so they cap partial credit below 1.0. Examples [derived, heuristic]:
     - can_meat: 0.44
     - sorting_bottles_cans_and_paper: 0.38
     - canning_food: 0.40
     - put_together_a_basic_pruning_kit: 0.50
     - many "keep the fridge/cabinet closed" tasks: 0.67–0.92
6. **The demo dataset.**
   - LeRobot **v3**, 20,000 demos (200 per task), 3.27 TB [docs]. Each task is one chunk: `chunk-%03d` = task id (`docs/challenge/baselines.md:55-70`, `score_utils.py:60-79`) [docs + code].
   - State is **61-D** (table in §4.3) and action is **23-D** (`eval_utils.py:105-141`) [code].
   - Cameras: head `zed_link_camera_0` at 720×720 and two wrist `*_realsense_link_camera_0` at 480×480.
   - Each camera has RGB (HEVC yuv420p, crf 30) and `depth_linear` (HEVC gray12le, log-quantized over 0.01–10 m with shift 3.5).
   - There is a per-depth-camera `observation.robot2cam_pose.<cam>` (7-D) (`envs/lerobot_data_wrapper.py:86-111, 159-196`; `eval/utils/obs_utils.py:24-95`) [code].
   - Annotations are per-episode JSON. The **2025** schema is `skill_annotation` / `primitive_annotation` with `frame_duration`, `skill_description`, `skill_type`, `object_id`, … [fetched tutorial]. The 2026 schema is undocumented in the repo; snippets suggest it is similar.
7. **No pure-navigation task exists.**
   - None of the 100 BDDL goals mentions the agent, `inroom` or `grasped` [derived].
   - The proposed DEV set therefore uses a navigation-dominant task (turning_out_all_lights_before_sleep) as the "navigation" probe (§5).

---

## 1. The 100 tasks

### 1.1 Where the list comes from and in what order

`docs/gen_2026_task_data.py` (verbatim docstring, lines 3-6):
```python
The 2026 BEHAVIOR Challenge covers 100 tasks: the 50 tasks from the 2025 challenge
(carried over verbatim, with their existing Vimeo media) plus 50 brand-new tasks whose
instances live in ``datasets/2026-challenge-task-instances/``.
```

The sources are declared at lines 29-35: `available_tasks.yaml`, `task_custom_lists.json` and `B100_task_misc.csv` (`# Canonical B100 task ordering: column "Task ID" is a numeric index, column "Task" is the id.`).

The new tasks are ordered as follows (lines 92-98):
```python
    # 50 new tasks: authoritative list = keys of available_tasks.yaml, ordered by their index
    # in the canonical B100 task list. ...
    b100_order = load_b100_order()
    new_ids = list(available_tasks.keys())
    new_ids.sort(key=lambda task_id: b100_order.get(task_id, len(b100_order) + 1))
```

The runtime mapping (`OmniGibson/omnigibson/eval/utils/eval_utils.py:60-71`) [code]:
```python
def _load_challenge_task_misc() -> tuple[dict[str, int], dict[str, list[str]]]:
    ...
    task_misc_path = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances", "metadata", "B100_task_misc.csv")
    with open(task_misc_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_ids[row["Task"]] = int(row["Task ID"])
            task_rooms[row["Task"]] = [room.strip() for room in row["Rooms to inlcude"].splitlines() if room.strip()]
    return task_ids, task_rooms

TASK_NAMES_TO_INDICES, TASK_NAMES_TO_ROOMS = _load_challenge_task_misc()
```

How the mapping is used:
- `TASK_NAMES_TO_INDICES` sets `obs["task_id"]` (`evaluator.py:591`).
- `TASK_NAMES_TO_ROOMS` sets the partial-scene room instances (`evaluator.py:253-254`; `eval.py` always sets `"partial_scene_load": True`, `eval.py:160`).
- `replay_obs.py:39-40` maps demo IDs to tasks with `task_id = demo_id // 10000`, and the raw HDF5 path is `2026-challenge-rawdata/task-{task_id:04d}/episode_{demo_id:08d}.hdf5` (`replay_obs.py:122`) [code].

**Is the `task_data.json` position the same as the B100 `Task ID`?**
- For indices 0–49: **yes** [code, fetched]. The 2025 `TASK_NAMES_TO_INDICES` (`v3.7.2 .../learning/utils/eval_utils.py:185-241`) maps `turning_on_radio:0 … make_pizza:49` in exactly the `task_data.json` order. The 2025 tutorial shows `task-0000/episode_00000010` for turning_on_radio.
- For indices 50–99: **inferred**. They are sorted by B100 index, so if the new Task IDs are the contiguous 50–99 they coincide. The CSV was NOT FOUND (checked locally, on GitHub, and via HF, which is blocked).
- `baselines.md:55-58` also says "Tasks are stored as numbered chunks (`chunk-000` is the first task …) `TASK_ID=0 # 0-indexed task id`".

**Other files that are not the 2026 list:**
- `OmniGibson/scripts/sampling/task_custom_lists.json` (77 tasks, an older sampling list; 40 of the 2026 tasks are missing from it).
- `docs/behavior_100/*` (the legacy iGibson BEHAVIOR-100, `docs/behavior_100/overview.md:20`).
- `docs/assets/b100/captioned/*.jpg` (legacy B100 activity images).

**Fields in `task_data.json`:**
- All 100 entries have `id, name, rooms, duration, thumbnail, video, scene_model`.
- **Only the 50 carry-over tasks have `instruction`.** The new 50 have none [derived].
- The natural-language task descriptions for all 100 are in the demo dataset's `meta/tasks.jsonl` (`docs/challenge/updates.md:175`) [docs]. That file was not readable here.

### 1.2 Column definitions for the full table (§1.4)

- **idx**: position in `docs/challenge/task_data.json`. It equals the B100 `Task ID` for 0–49 [code] and is assumed equal for 50–99 [inferred].
- **set**: `2025` (carry-over, idx 0–49) or `new` (idx 50–99) [code: `gen_2026_task_data.py:82-110`].
- **scene**: `scene_model` from `task_data.json`. Abbreviations: HDFL = house_double_floor_lower, HDFU = house_double_floor_upper, HSF = house_single_floor, RD = restaurant_diner, HSL = hotel_suite_large, Rs = Rs_int, OCR = office_cubicles_right.
- **rooms**: room *types* from `task_data.json`. For new tasks this comes from `task_custom_lists.json` `room_types`. The evaluator actually loads room *instances* from the CSV column `Rooms to inlcude`, which can include connecting rooms (task_sampling.md:94-103).
- **demo mean s (plot)**: the per-task mean human demo duration, measured from the official plot `docs/assets/challenge_2026/per_task_demo_duration.png`. I calibrated with the gridlines (0 min at x=613.5 px, 17.5 min at x=3134.5 px, 144.06 px/min) and added +0.5 s to correct a systematic offset seen on the 2025 tasks [derived].
  - Validation: the mean of the 100 plotted means is 351.1 s, against `dataset.md:112` "Avg. Trajectory Duration 351.54 seconds". For 2025 tasks the values agree with `task_data.json` within about 1 s.
- **task_data s**: the `duration` field (the docs gallery renders it as "Duration: … avg", `docs/gen_task_pages.py:670`).
- **~default max_steps**: estimated as `int(45 × plot mean s)` = 1.5 (`EVAL_TIMEOUT_MULTIPLIER`, `eval_utils.py:17`) × 30 frames/s × seconds [inferred].
  - The exact value is `int(task.jsonl["length"] × 1.5)` (`evaluator.py:281-286`). `task.jsonl` was NOT FOUND.
  - The 30 Hz rate comes from `eval_utils.py:204` `action_frequency: 30` and the 2025 parquet timestamps step of 0.0333 s (tutorial).
- **goal lits**: the number of grounded goal literals in a goal option. This equals the q_score denominator (all options of a task have the same length for these 100 tasks) [derived, computed with `bddl.condition_evaluation.get_ground_state_options` exactly as `BehaviorTask` uses it].
  - `N+` marks tasks whose goal quantifies over a **wildcard** object (`cabinet.n.01_*`, `electric_refrigerator.n.01_*`). Their literal count grows with the number of matching scene objects in the loaded rooms (`bddl3/bddl/wildcard.py:1-11`), so it is a lower bound.
- **goal predicates**: the BDDL predicate tokens that appear in the goal.
- **movable (goal-rel)**: the number of non-`inroom`, non-agent, non-substance object instances, with the number referenced by the goal (by instance or by quantified category) in parentheses [derived].
- **credit cap**: the max partial q_score attainable without full success. It equals the max over options of (literals not already true at init) / (option length). This is [derived, heuristic]: it uses closed-world reasoning on BDDL `:init` (objects default closed, off, real unless `future`, and `ontop` implies `touching`). The true value depends on the sampled instance.
- **flags** [derived heuristics]:

  | Flag | Meaning |
  |---|---|
  | MR | more than one room type |
  | ART | goal has `open`, or a goal `inside` an openable, or an init object inside an openable (openable = `properties_to_synsets.json["openable"]`) |
  | TOG | `toggled_on` in goal |
  | PRT | substance/particle systems or `covered`/`filled`/`contains`/`saturated` |
  | COOK | `cooked`/`hot`/`frozen`/`on_fire` |
  | TRN | `future`/`real` (objects created by transition rules) |
  | SLC | slicer plus sliceable/diceable objects present |
  | ATT | `attached` |
  | MO | 3 or more goal-relevant movable objects |
  | LH | plot mean demo ≥ 400 s |

  `ART*` on make_microwave_popcorn means the microwave must be opened and closed even though the goal does not mention it [inferred].

### 1.3 Aggregate statistics [derived]

- **Scenes (7)** — `index.md:104` says "7 scenes, including 4 new scenes":

  | Scene | All tasks | 2025 | new |
  |---|---:|---:|---:|
  | house_single_floor | 34 | 23 | 11 |
  | house_double_floor_lower | 32 | 22 | 10 |
  | house_double_floor_upper | 10 | 5 | 5 |
  | restaurant_diner | 8 | 0 | 8 |
  | Rs_int | 6 | 0 | 6 |
  | hotel_suite_large | 5 | 0 | 5 |
  | office_cubicles_right | 5 | 0 | 5 |

- **Number of room types per task:** 1 room for 71 tasks, 2 for 19, 3 for 3, 4 for 5, 6 for 1, 8 for 1.
- **Goal literals per task:** mean 5.5, max 26 (cook_brussels_sprouts). 11 tasks have exactly 1 literal: turning_on_radio, hanging_pictures, attach_a_camera_to_a_tripod, clean_a_patio, clean_a_trumpet, scrubbing_bathroom_floor, installing_smoke_detectors, vacuuming_floors, make_cabinet_doors, store_honey, clean_a_keyboard.
- **Flag counts (100 tasks):**

  | Flag | Count |
  |---|---:|
  | MO | 69 |
  | ART | 38 (+1 implicit) |
  | LH (task_data durations) | 34 |
  | MR | 29 |
  | PRT | 20 |
  | COOK | 9 |
  | TRN | 9 |
  | SLC | 9 |
  | TOG | 7 |
  | ATT | 4 |

- **Number of tasks whose goal uses each predicate:**

  | Predicate | Tasks |
  |---|---:|
  | inside | 54 |
  | ontop | 40 |
  | open (always as `not open` = "closed") | 22 |
  | nextto | 16 |
  | covered | 13 |
  | real | 9 |
  | toggled_on | 7 |
  | cooked | 6 |
  | under | 4 |
  | attached | 4 |
  | contains | 4 |
  | touching | 3 |
  | frozen | 2 |
  | on_fire | 1 |
  | filled | 1 |

- **Demo durations (plot):** 71 s (turning_on_radio) to 868 s (assembling_gift_baskets). The approximate default max_steps therefore ranges from about 3.2k to about 39.1k.

### 1.4 Full table of 100 tasks

| idx | task | set | scene | rooms | demo mean s (plot) | task_data s | ~default max_steps | goal lits | goal predicates | movable (goal-rel) | credit cap | flags |
|---:|---|---|---|---|---:|---:|---:|---:|---|---|---:|---|
| 0 | turning_on_radio | 2025 | HDFL | living_room | 72 | 72 | 3226 | 1 | toggled_on | 1 (1) | 1 | TOG |
| 1 | picking_up_trash | 2025 | HDFL | kitchen, living_room | 176 | 176 | 7911 | 3 | inside | 4 (4) | 1 | MR MO |
| 2 | putting_away_Halloween_decorations | 2025 | HDFL | living_room | 460 | 460 | 20695 | 8+ | inside, nextto, open | 6 (6) | 0.75 | ART MO LH |
| 3 | cleaning_up_plates_and_food | 2025 | HDFL | kitchen | 457 | 457 | 20547 | 8+ | inside, ontop, open | 6 (6) | 0.75 | ART MO LH |
| 4 | can_meat | 2025 | HSF | kitchen | 395 | 395 | 17770 | 9 | inside, open | 7 (6) | 0.44 | ART MO |
| 5 | setting_mousetraps | 2025 | HDFU | bathroom | 340 | 339 | 15295 | 6 | nextto, ontop, under | 4 (4) | 1 | MO |
| 6 | hiding_Easter_eggs | 2025 | HDFL | garden | 254 | 254 | 11434 | 9 | inside, nextto, ontop | 4 (4) | 1 | MO |
| 7 | picking_up_toys | 2025 | HSF | childs_room | 630 | 630 | 28340 | 6 | inside | 7 (7) | 1 | MO LH |
| 8 | rearranging_kitchen_furniture | 2025 | HDFL | kitchen | 298 | 298 | 13423 | 4 | inside, open | 3 (3) | 0.75 | ART MO |
| 9 | putting_up_Christmas_decorations_inside | 2025 | HSF | kitchen, living_room | 457 | 457 | 20583 | 9 | nextto, ontop, touching, under | 11 (10) | 1 | MR MO LH |
| 10 | set_up_a_coffee_station_in_your_kitchen | 2025 | HSF | kitchen | 209 | 209 | 9414 | 6 | nextto, ontop | 6 (6) | 0.83 | MO |
| 11 | putting_dishes_away_after_cleaning | 2025 | HSF | kitchen | 365 | 365 | 16443 | 10+ | inside, open | 8 (8) | 0.80 | ART MO |
| 12 | preparing_lunch_box | 2025 | HSF | kitchen | 275 | 275 | 12375 | 6 | inside, open | 7 (6) | 0.83 | ART MO |
| 13 | loading_the_car | 2025 | HDFL | living_room, garage | 641 | 641 | 28849 | 3 | inside | 3 (3) | 1 | MR ART MO LH |
| 14 | carrying_in_groceries | 2025 | HDFL | kitchen, garage | 476 | 476 | 21424 | 4 | inside, open | 3 (2) | 0.75 | MR ART LH |
| 15 | bringing_in_wood | 2025 | HDFL | garden, corridor | 451 | 451 | 20304 | 3 | ontop | 3 (3) | 1 | MR MO LH |
| 16 | moving_boxes_to_storage | 2025 | HDFL | living_room, garage | 487 | 487 | 21897 | 2 | ontop | 2 (2) | 1 | MR LH |
| 17 | bringing_water | 2025 | HSF | living_room, kitchen | 314 | 315 | 14152 | 3 | ontop, open | 2 (2) | 0.67 | MR ART |
| 18 | tidying_bedroom | 2025 | HSF | bedroom | 368 | 368 | 16555 | 3 | nextto, ontop | 3 (3) | 1 | MO |
| 19 | outfit_a_basic_toolbox | 2025 | HSF | utility_room | 354 | 355 | 15952 | 7 | inside, ontop, open | 6 (6) | 0.71 | ART MO |
| 20 | sorting_vegetables | 2025 | HSF | kitchen | 397 | 397 | 17865 | 13 | inside | 18 (16) | 1 | MO |
| 21 | collecting_childrens_toys | 2025 | HSF | childs_room | 640 | 640 | 28790 | 7 | inside | 7 (7) | 1 | MO LH |
| 22 | putting_shoes_on_rack | 2025 | HDFL | corridor | 258 | 258 | 11587 | 10 | nextto, touching | 4 (4) | 1 | MO |
| 23 | boxing_books_up_for_storage | 2025 | HDFU | living_room | 808 | 808 | 36346 | 6 | inside | 7 (7) | 1 | MO LH |
| 24 | storing_food | 2025 | HSF | kitchen | 662 | 662 | 29803 | 8 | inside | 8 (8) | 1 | ART MO LH |
| 25 | clearing_food_from_table_into_fridge | 2025 | HDFL | kitchen | 436 | 436 | 19606 | 5 | inside, open | 6 (4) | 0.80 | ART MO LH |
| 26 | assembling_gift_baskets | 2025 | HDFL | living_room | 869 | 869 | 39100 | 16 | inside | 20 (20) | 1 | MO LH |
| 27 | sorting_household_items | 2025 | HSF | bedroom, bathroom | 527 | 527 | 23715 | 8 | inside, nextto, ontop, under | 9 (7) | 0.88 | MR MO LH |
| 28 | getting_organized_for_work | 2025 | HDFU | bedroom | 522 | 522 | 23508 | 10 | nextto, ontop, under | 7 (7) | 0.70 | MO LH |
| 29 | clean_up_your_desk | 2025 | HSF | childs_room | 714 | 714 | 32130 | 11 | inside, ontop, open | 10 (10) | 0.91 | ART MO LH |
| 30 | setting_the_fire | 2025 | HDFL | living_room | 304 | 304 | 13684 | 8 | inside, on_fire, ontop, toggled_on | 4 (4) | 0.88 | TOG COOK MO |
| 31 | clean_boxing_gloves | 2025 | HSF | utility_room | 275 | 275 | 12357 | 2 | covered | 2 (2) | 1 | PRT |
| 32 | wash_a_baseball_cap | 2025 | HSF | utility_room | 278 | 278 | 12523 | 2 | covered | 2 (2) | 1 | PRT |
| 33 | wash_dog_toys | 2025 | HSF | utility_room | 374 | 374 | 16834 | 6 | covered | 4 (4) | 0.67 | ART PRT MO |
| 34 | hanging_pictures | 2025 | HDFL | kitchen | 80 | 80 | 3581 | 1 | attached | 1 (1) | 1 | ATT |
| 35 | attach_a_camera_to_a_tripod | 2025 | HDFU | bedroom | 130 | 130 | 5868 | 1 | attached | 2 (2) | 1 | ATT |
| 36 | clean_a_patio | 2025 | HDFL | garden | 402 | 402 | 18108 | 1 | covered | 1 (0) | 1 | PRT LH |
| 37 | clean_a_trumpet | 2025 | HDFU | bedroom | 177 | 177 | 7969 | 1 | covered | 2 (1) | 1 | PRT |
| 38 | spraying_for_bugs | 2025 | HDFL | garden | 216 | 216 | 9729 | 2 | covered | 3 (2) | 1 | PRT |
| 39 | spraying_fruit_trees | 2025 | HDFL | garden | 278 | 278 | 12523 | 2 | covered | 1 (0) | 1 | PRT |
| 40 | make_microwave_popcorn | 2025 | HDFL | kitchen | 108 | 108 | 4860 | 2 | contains, real | 1 (1) | 1 | ART* PRT TRN |
| 41 | cook_cabbage | 2025 | HSF | kitchen | 472 | 471 | 21222 | 4 | contains, real | 6 (1) | 1 | ART PRT TRN SLC LH |
| 42 | chop_an_onion | 2025 | HDFL | kitchen | 213 | 213 | 9598 | 4 | contains, inside, real | 4 (3) | 1 | PRT TRN SLC MO |
| 43 | slicing_vegetables | 2025 | HSF | kitchen | 495 | 495 | 22270 | 9 | open, real | 8 (5) | 0.89 | ART PRT TRN SLC MO LH |
| 44 | chopping_wood | 2025 | HDFL | garden | 359 | 358 | 16141 | 8 | real | 14 (8) | 1 | TRN SLC MO |
| 45 | cook_hot_dogs | 2025 | HSF | kitchen | 305 | 305 | 13725 | 2 | cooked | 2 (2) | 1 | ART COOK |
| 46 | cook_bacon | 2025 | HSF | kitchen | 256 | 256 | 11529 | 7 | cooked, open | 8 (6) | 0.86 | ART COOK MO |
| 47 | freeze_pies | 2025 | HSF | kitchen | 415 | 415 | 18688 | 7 | frozen, inside, open | 6 (4) | 0.86 | ART COOK MO LH |
| 48 | canning_food | 2025 | HSF | kitchen | 766 | 766 | 34470 | 10 | contains, filled, inside, open, real | 6 (2) | 0.40 | ART PRT TRN SLC LH |
| 49 | make_pizza | 2025 | HDFL | kitchen | 640 | 640 | 28790 | 2 | ontop, real | 16 (2) | 1 | ART PRT TRN SLC LH |
| 50 | freeze_fruit | new | HSF | kitchen | 421 | 339 | 18931 | 7 | inside, open | 8 (6) | 0.86 | ART MO LH |
| 51 | cook_a_brisket | new | HDFL | kitchen | 245 | 251 | 11007 | 3 | cooked, ontop | 3 (3) | 0.67 | ART COOK MO |
| 52 | sorting_bottles_cans_and_paper | new | HDFL | kitchen | 342 | 312 | 15390 | 16 | inside | 9 (9) | 0.38 | MO |
| 53 | tidying_living_room | new | HDFU | living_room | 419 | 316 | 18859 | 4 | inside, ontop | 4 (4) | 1 | MO LH |
| 54 | putting_away_toys | new | HSF | dining_room, living_room | 377 | 438 | 16947 | 8 | inside | 10 (10) | 1 | MR MO |
| 55 | re_shelving_library_books | new | HDFU | living_room | 318 | 281 | 14305 | 3 | inside | 3 (3) | 1 | MO |
| 56 | make_rose_centerpieces | new | HDFL | living_room | 151 | 111 | 6808 | 4 | inside, ontop | 4 (4) | 1 | MO |
| 57 | sweeping_garage | new | HDFL | garage | 149 | 126 | 6713 | 2 | covered | 2 (0) | 1 | PRT |
| 58 | stacking_wood | new | HSF | garden | 532 | 447 | 23917 | 6 | touching | 6 (6) | 1 | MO LH |
| 59 | organizing_art_supplies | new | HDFU | bedroom | 207 | 228 | 9301 | 5 | inside, ontop | 5 (5) | 1 | MO |
| 60 | scrubbing_bathroom_floor | new | HSF | bathroom | 105 | 123 | 4725 | 1 | covered | 1 (0) | 1 | PRT |
| 61 | bringing_paper_to_recycling | new | HDFL | kitchen, garden | 374 | 318 | 16834 | 3 | inside, open | 3 (3) | 0.67 | MR ART MO |
| 62 | halve_an_egg | new | HSF | kitchen | 212 | 212 | 9562 | 5 | inside, ontop, real | 6 (4) | 1 | ART TRN SLC MO |
| 63 | installing_smoke_detectors | new | HDFL | living_room, kitchen, garden | 86 | 68 | 3865 | 1 | attached | 1 (1) | 1 | MR ATT |
| 64 | setting_the_table | new | HSF | dining_room, kitchen | 594 | 727 | 26712 | 8 | nextto, ontop | 8 (8) | 1 | MR ART SLC MO LH |
| 65 | unloading_the_car | new | HDFL | garage, living_room | 377 | 378 | 16983 | 2 | nextto | 2 (2) | 1 | MR ART |
| 66 | turning_out_all_lights_before_sleep | new | HSF | corridor, utility_room, childs_room, dining_room, bedroom, kitchen, living_room, bathroom | 334 | 337 | 15016 | 5 | toggled_on | 2 (2) | 1 | MR TOG |
| 67 | boxing_food_after_dinner | new | HSF | dining_room, kitchen | 223 | 240 | 10048 | 6 | inside, ontop, open | 5 (5) | 0.83 | MR ART MO |
| 68 | cleaning_up_branches_and_twigs | new | HDFL | garden | 418 | 434 | 18801 | 5 | inside, ontop | 5 (5) | 0.80 | ART MO LH |
| 69 | vacuuming_floors | new | HDFU | bedroom | 80 | 65 | 3622 | 1 | covered | 1 (0) | 1 | PRT |
| 70 | thawing_frozen_food | new | HDFL | kitchen | 275 | 348 | 12357 | 9 | frozen, inside, ontop, open, toggled_on | 4 (4) | 0.78 | ART TOG COOK MO |
| 71 | clean_your_rusty_garden_tools | new | HSF | garden | 505 | 484 | 22738 | 5 | covered, inside, open | 4 (3) | 0.80 | ART PRT MO LH |
| 72 | cook_a_frozen_pie | new | RD | kitchen | 289 | 323 | 13009 | 2 | cooked, ontop | 2 (2) | 1 | ART COOK |
| 73 | organizing_school_stuff | new | HSF | bedroom | 487 | 426 | 21897 | 6 | inside, ontop | 6 (6) | 1 | MO LH |
| 74 | carrying_out_garden_furniture | new | HSF | corridor, dining_room, entryway, garden, kitchen, living_room | 369 | 358 | 16591 | 2 | ontop | 2 (2) | 1 | MR |
| 75 | put_together_a_basic_pruning_kit | new | HDFL | garage | 376 | 448 | 16911 | 4 | inside, ontop, open | 3 (3) | 0.50 | ART MO |
| 76 | dispose_of_glass | new | HSL | bathroom, bedroom | 307 | 240 | 13797 | 4 | inside | 5 (5) | 1 | MR MO |
| 77 | installing_a_modem | new | Rs | bedroom, entryway, kitchen, living_room | 80 | 63 | 3622 | 4 | nextto, ontop, toggled_on | 1 (1) | 1 | MR TOG |
| 78 | make_cabinet_doors | new | Rs | bedroom, entryway, kitchen, living_room | 115 | 120 | 5175 | 1 | attached | 2 (2) | 1 | MR ATT |
| 79 | polishing_shoes | new | HSL | bedroom | 358 | 363 | 16105 | 6 | covered, nextto, ontop | 3 (3) | 0.83 | PRT MO |
| 80 | clean_up_broken_glass | new | RD | corridor, dining_room | 290 | 286 | 13050 | 3 | inside | 4 (4) | 1 | MR MO |
| 81 | packing_meal_for_delivery | new | RD | kitchen | 295 | 300 | 13292 | 6 | inside, ontop | 8 (7) | 1 | MO |
| 82 | store_batteries | new | Rs | bedroom, entryway, kitchen, living_room | 257 | 328 | 11569 | 3 | inside | 3 (3) | 1 | MR ART MO |
| 83 | store_honey | new | Rs | bedroom, entryway, kitchen, living_room | 226 | 193 | 10161 | 1 | inside | 1 (1) | 1 | MR ART |
| 84 | tidying_bathroom | new | HSL | bathroom | 434 | 541 | 19516 | 4 | inside, ontop | 6 (6) | 1 | MO LH |
| 85 | putting_dirty_dishes_in_sink | new | RD | kitchen, dining_room | 405 | 337 | 18220 | 4 | inside | 4 (4) | 1 | MR PRT MO LH |
| 86 | make_gift_bags_for_baby_showers | new | Rs | living_room | 322 | 242 | 14472 | 6 | inside, ontop | 6 (6) | 1 | ART MO |
| 87 | collecting_aluminum_cans | new | HSL | bedroom | 340 | 325 | 15317 | 6 | inside | 7 (7) | 1 | MO |
| 88 | rearrange_your_room | new | HSL | bedroom, bathroom | 429 | 515 | 19291 | 3 | ontop | 3 (3) | 1 | MR MO LH |
| 89 | installing_a_fax_machine | new | OCR | corridor, shared_office | 135 | 148 | 6093 | 2 | ontop, toggled_on | 1 (1) | 1 | MR TOG |
| 90 | composting_waste | new | Rs | entryway, kitchen, living_room | 121 | 115 | 5458 | 2 | inside | 3 (3) | 1 | MR MO |
| 91 | store_produce | new | RD | kitchen | 318 | 316 | 14287 | 4 | inside | 5 (4) | 1 | ART MO |
| 92 | installing_a_scanner | new | OCR | private_office | 142 | 122 | 6394 | 2 | nextto, toggled_on | 2 (2) | 1 | TOG |
| 93 | clean_a_keyboard | new | OCR | private_office | 129 | 142 | 5813 | 1 | covered | 2 (1) | 1 | PRT |
| 94 | dispose_of_batteries | new | OCR | copy_room, corridor, private_office, shared_office | 481 | 441 | 21649 | 4 | inside, ontop | 4 (4) | 0.75 | MR MO LH |
| 95 | cook_brussels_sprouts | new | RD | bar, corridor, kitchen | 522 | 687 | 23490 | 26 | cooked, inside, open, real | 16 (8) | 0.92 | MR ART COOK TRN SLC MO LH |
| 96 | cook_broccolini | new | RD | kitchen | 154 | 137 | 6938 | 11 | cooked, ontop | 7 (6) | 1 | COOK MO |
| 97 | setup_a_bar_for_a_cocktail_party | new | RD | bar | 447 | 350 | 20133 | 20 | inside, nextto, ontop | 15 (14) | 1 | MO LH |
| 98 | laying_tile_floors | new | OCR | bathroom, corridor | 479 | 520 | 21537 | 8 | nextto, ontop | 4 (4) | 1 | MR MO LH |
| 99 | sorting_books_on_shelf | new | HDFU | living_room | 258 | 200 | 11605 | 11 | inside, ontop | 7 (7) | 0.64 | MO |

The machine-readable per-task data, including the literal list of the shortest grounded option, object counts, openables, substances, `future` objects and the instruction, is in `r6_task_metadata.json`. The initially-satisfied literals are in `r6_init_satisfied.json`. Both are in the same directory as this report.

### 1.5 How goal literals are counted, and caveats (important for q_score)

How the q_score is computed (`OmniGibson/omnigibson/metrics/task_metric.py:6-28`) [code]:
```python
def compute_q_score(success, now_satisfied_options, initial_satisfied_options) -> float:
    if success:
        return 1.0
    ...
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        ...
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0
```

- The `initial_satisfied_options` are captured by `TaskMetric.reset` (`task_metric.py:43-52`). This runs after the instance is loaded and settled (`evaluator.py:567-568`).
- The options are `BehaviorTask.ground_goal_state_options`, evaluated per env (`OmniGibson/omnigibson/tasks/behavior_task.py:206-235`).
- The options come from `bddl3/bddl/condition_evaluation.py:741-794`. That function takes the Cartesian product of each top-level goal condition's ground options, removes self-contradictory options, and sorts shortest-first.

Caveats:
1. **Literals true at reset are uncredited but still in the denominator.** Typical cases are "fridge/cabinet closed" goals (`(not (open X))` with X closed at start) and objects already in place (`(ontop coffee_maker countertop)`). The partial-credit caps in the table follow from this. Full success still returns 1.0.
2. **Quantifier expansion.** `forall`/`forn`/`forpairs` expand over the declared instances of exactly that category (`condition_evaluation.py:172-221, 267-320, 322-407`).
   - The option counts can be huge. assembling_gift_baskets has 4 `forpairs` over 4×4, which gives 24⁴ = 331,776 raw options of 16 literals. I could not enumerate it within a 60 s budget and computed its length analytically.
   - stacking_wood has 46,656 options and putting_up_Christmas_decorations_inside has 3,888.
3. **Self-referential options.** sorting_books_on_shelf's goal allows grounded options containing `(ontop comic_book.n.01_1 comic_book.n.01_1)` (`bddl3/bddl/activity_definitions/sorting_books_on_shelf/problem0.bddl`, the goal's `forn`/`exists` over `or`). Such options are unsatisfiable, but q_score takes the max over options, so other options apply.
4. **Wildcards.** 16 tasks declare wildcard scene objects (`*_*`). They are expanded from the loaded room layout (`bddl3/bddl/wildcard.py:1-11`, `behavior_task.py:334-600`).
   - The literal count is affected only when the goal quantifies with `forall` over the wildcard category: idx 2, 3, 11 ("every cabinet/fridge closed"). For `exists` the length is unchanged (e.g. idx 53, 55, 88, 89).

---

## 2. The 2025 carry-over tasks and the new 2026 tasks

- **2025 carry-over (50) = idx 0–49:** turning_on_radio, picking_up_trash, putting_away_Halloween_decorations, cleaning_up_plates_and_food, can_meat, setting_mousetraps, hiding_Easter_eggs, picking_up_toys, rearranging_kitchen_furniture, putting_up_Christmas_decorations_inside, set_up_a_coffee_station_in_your_kitchen, putting_dishes_away_after_cleaning, preparing_lunch_box, loading_the_car, carrying_in_groceries, bringing_in_wood, moving_boxes_to_storage, bringing_water, tidying_bedroom, outfit_a_basic_toolbox, sorting_vegetables, collecting_childrens_toys, putting_shoes_on_rack, boxing_books_up_for_storage, storing_food, clearing_food_from_table_into_fridge, assembling_gift_baskets, sorting_household_items, getting_organized_for_work, clean_up_your_desk, setting_the_fire, clean_boxing_gloves, wash_a_baseball_cap, wash_dog_toys, hanging_pictures, attach_a_camera_to_a_tripod, clean_a_patio, clean_a_trumpet, spraying_for_bugs, spraying_fruit_trees, make_microwave_popcorn, cook_cabbage, chop_an_onion, slicing_vegetables, chopping_wood, cook_hot_dogs, cook_bacon, freeze_pies, canning_food, make_pizza.
  - Evidence: `docs/challenge/archive/2025/task_data.json` lists the same 50 ids in the same order, and every field except the newly added `scene_model` is identical [derived diff].
  - All 50 use the 3 houses (HSF 23, HDFL 22, HDFU 5).
- **New in 2026 (50) = idx 50–99:** freeze_fruit … sorting_books_on_shelf (see the table).
  - 24 of them use the 4 new scenes: restaurant_diner, Rs_int, hotel_suite_large, office_cubicles_right.
  - The new tasks are shorter on average: task_data mean 303 s vs 397 s for 2025. They also have more multi-room tasks (21 vs 8 flagged MR).
- **Robot start poses differ between the two sets.** The carry-over tasks store presampled robot poses only per robot model (`fetch, r1, r1pro, stretch, tiago`). A custom robot must therefore set `eval.use_presampled_pose_key` (`docs/challenge/evaluation.md:218`; `behavior_task.py:35-60`) [docs + code].

---

## 3. Instance conventions

**Constants** (`OmniGibson/omnigibson/eval/utils/eval_utils.py:16-21`) [code; identical on origin/main, 2026-eval and group-time-budget]:
```python
DEFAULT_EVAL_SEED = 0
EVAL_TIMEOUT_MULTIPLIER = 1.5
NUM_TEST_INSTANCES = 40
NUM_PUBLIC_TEST_INSTANCES = 20
NUM_HIDDEN_TEST_INSTANCES = NUM_TEST_INSTANCES - NUM_PUBLIC_TEST_INSTANCES
TEST_INSTANCE_IDS = list(range(301, 341))
```

**Resolving indices to IDs** (`OmniGibson/omnigibson/eval/evaluator.py:73-87`) [code]:
```python
def resolve_instance_ids(task_name, instance_indices, mode="public_test"):
    if mode == "train":
        return [int(instance_id) for instance_id in instance_indices]
    test_instances = (TEST_INSTANCE_IDS[:NUM_PUBLIC_TEST_INSTANCES] if mode == "public_test"
                      else TEST_INSTANCE_IDS[NUM_PUBLIC_TEST_INSTANCES:])
    ...  # indices must be in range(20) for each test split
    return [int(test_instances[i]) for i in instance_indices]
```

**Where instance files live** (`OmniGibson/omnigibson/utils/asset_utils.py:125-146`) [code]:
```python
    mode_to_dir = {"train": "scenes",
                   "public_test": os.path.join("scene_test", "public"),
                   "hidden_test": os.path.join("scene_test", "private")}
    task_instances_path_2026 = os.path.join(gm.DATA_PATH, "2026-challenge-task-instances",
                                            mode_to_dir[mode], scene_name, "json", f"{instance_name}.json")
```

- The instance name is `f"{scene}_task_{task}_instances/{scene}_task_{task}_0_{instance_id}_template-tro_state"` (`evaluator.py:464-476`, `behavior_task.py:181`).
- Only the robot base pose and the task-relevant object (TRO) states are restored from the TRO file. The robot joints are reset to default (`evaluator.py:458-507`).

**Split summary:**

| Split | IDs | CLI | Use |
|---|---|---|---|
| seed/template | 0 | — | `..._0_0_template.json` scene file; teleop default (task_sampling.md:68, og_robot.py:54-61) |
| train | 1–300 | `--mode train --instance-indices <id>` | sampled training configs (task_sampling.md:145-150); 200 demos/task collected on these (og_robot.py:56-58, 1073-1087 increments instance id per reset) [inferred that demos ⊂ 1–300] |
| public_test | 301–320 | `--mode public_test --instance-indices 0..19` (default mode) | **0–9 → IDs 301–310 for reported results**; 10–19 local test set (evaluation.md:282) |
| hidden_test | 321–340 | `--mode hidden_test --instance-indices 0..19` | organizers' final evaluation; `scene_test/private` (probably not shipped) [inferred] |

**Docs statements** (evaluation.md:66, 282, 290, 294):
- "Use indices 0 1 2 3 4 5 6 7 8 9 for reported evaluation results. Indices 0-19 are public test instances; indices 20-39 are hidden instances reserved for final evaluation."
- "We will hold out 10 more instances for final evaluation … evaluate the top-5 solutions."
- "Each instance differs in terms of initial object states and initial robot poses."
- `submission.md:10`: "Full evaluation size: 100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs."
- `origin/main:docs/challenge/updates.md:23, 61`: "hidden-test scores will replace public-instance scores for final ranking" and "Hidden-test evaluation will follow the tasks and instances covered by your submission."

**How instances are distributed:**
- `setup.sh --dataset` calls `download_2026_challenge_task_instances()` (`setup.sh:505`). That downloads `2026-challenge-task-instances.zip` from HF dataset `behavior-1k/zipped-datasets` (`asset_utils.py:426-441, 589-592`) [code].
- The development source is the git repo `wensi-ai/2026-challenge-task-instances` (`docs/behavior_components/task_sampling.md:7-11`). It was not accessible from here.
- The metadata files in it are `metadata/{B100_task_misc.csv, available_tasks.yaml, task_custom_lists.json, task.jsonl}` (`gen_2026_task_data.py:29-35`, `og_teleop_utils.py:1333`, `score_utils.py:25-26`).

**Room loading.** The rooms loaded at eval are the CSV room instances (partial scene) (`evaluator.py:253-254`; `docs/challenge/updates.md:150`). The guidance for choosing them is "minimal set of rooms … also those that connects them … as well as any room that could be in sight" (`task_sampling.md:94-103`). Navigation outside these rooms is impossible or empty [inferred].

**One task is special-cased.** turning_out_all_lights_before_sleep is in `LIGHT_EVAL_TASKS` (`evaluator.py:52, 230-239, 509-510`). It gets light-toggle synchronization, and `v3.9.3-post1` fixed light-toggle state (`origin/main:updates.md:51, 69`).

---

## 4. The demonstration dataset (`behavior-1k/2026-challenge-demos`)

### 4.1 Format and size

- **Size:** 20,000 demos across 100 tasks, 3.27 TB, LeRobot V3 "with customizations" (`docs/challenge/dataset.md:9`) [docs].
  - "1,950 hours in total" (`docs/challenge/index.md:112`).
  - 200 per task (`evaluation.md:241, 278`).
  - A snippet says 3.0 TB with 210,916,774 frames at 30 fps. Note that 3.27 TB ≈ 2.97 TiB, and 210.9 M frames / 30 / 3600 = 1,953 h, so the figures are consistent [snippet: huggingface.co/datasets/mahgoobi/b1k_2026_extras].
- **Raw data:** `2026-challenge-rawdata` is 1.44 TB of HDF5 for exact replay through `OmniGibson/scripts/learning/replay_obs.py` (`dataset.md:7`).
- **Folders:** `annotations/`, `data/`, `meta/`, `videos/` (`dataset.md:11-38`).
- **Layout** [docs + code]:
  - `data/chunk-XXX/file-YYY.parquet`
  - `meta/{info.json, stats.json, tasks.parquet, tasks.jsonl, episodes/chunk-XXX/file-YYY.parquet}`
  - `videos/<video_key>/chunk-XXX/file-YYY.mp4`
  - **chunk index = task id** (`baselines.md:55-70`; `score_utils.py:58-79`; `aggregate_lerobot_task_shards.py:117, 145, 164`).
- **Download one task** (`baselines.md:58-70`):
  ```bash
  CHUNK=$(printf "chunk-%03d" $TASK_ID)
  huggingface-cli download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir "$DATA_ROOT" \
      --include "data/$CHUNK/**" --include "meta/episodes/$CHUNK/**" --include "videos/*/$CHUNK/**" \
      --include "meta/info.json" --include "meta/stats.json" --include "meta/tasks.parquet"
  ```
- **Episode metadata columns** that the official tools read include `episode_index, task_index, raw_episode_id, length, data/chunk_index, data/file_index` (`OmniGibson/scripts/learning/update_lerobot_base_qvel.py:329-347`). The raw-file pattern is `task-{task_index:04d}/episode_{raw_episode_id:08d}.hdf5` (`update_lerobot_base_qvel.py:578`) [code]. Raw demo IDs encode the task as `demo_id // 10000` (`replay_obs.py:39-40`).
- **`meta/tasks.jsonl`** was added with natural-language task descriptions for all 100 tasks. "The first 50 tasks follow the 2025 challenge descriptions … the remaining 50 were derived from the 2026 annotations and task definitions" (`docs/challenge/updates.md:175`) [docs]. Contents NOT FOUND.
- **`info.json`** also stores `cam_intrinsics` per camera and `omnigibson_git_hash` (`lerobot_data_wrapper.py:319-335`) [code].
- **fps** comes from the collection config's `rendering_frequency` (`envs/data_wrapper.py:164-165, 606-607`). It is 30 Hz [inferred from `eval_utils.py:204-205` and the 2025 tutorial's timestamps 0.0333 s; snippet: "30 fps"].

### 4.2 Feature keys (writer code: `OmniGibson/omnigibson/envs/lerobot_data_wrapper.py:142-196, 272-294, 337-363`) [code]

| key | dtype/shape | notes |
|---|---|---|
| `action` | float32 (23,) | robot action (layout §4.4) |
| `observation.state` | float32 (61,) | proprio (all `proprio` modality obs map to this key, l.192); layout §4.3 |
| `observation.rgb.zed_link_camera_0` | video (720,720,3) | head; alpha dropped (l.221-222) |
| `observation.rgb.left_realsense_link_camera_0` / `right_…` | video (480,480,3) | wrists |
| `observation.depth_linear.<same 3 cams>` | video (H,W,1), `info.is_depth_map=True`, `video.output_unit="m"` | |
| `observation.robot2cam_pose.<cam>` | float32 (7,) | camera pose in robot base frame, pos(3)+quat(4), added per depth camera (l.173-182, 206-214) |
| `next.reward`, `next.terminated`, `next.truncated` | (1,) | RL fields |
| `task` | string | task name passed by replay (`replay_obs.py:158`) |
| standard LeRobot: `episode_index, frame_index, timestamp, index, task_index` | | (`update_lerobot_base_qvel.py:330`) |

- The camera key names come from the sensor names "robot_r1:zed_link:Camera:0" → "zed_link_camera_0" (`lerobot_data_wrapper.py:150-158`). They are confirmed by `OmniGibson/scripts/learning/visualize_lerobot_pointcloud.py:23-27` and by OPENPI `src/openpi/configs/robots/b1k.py:10-27` [code].
- Pairing convention: frame t stores obs[t] together with action[t] and reward/terminations from the step that action produced (`lerobot_data_wrapper.py:339-358`).
- **Encoders** (`lerobot_data_wrapper.py:92-110`) [code]:
  ```python
  "camera_encoder": VideoEncoderConfig(vcodec="hevc", pix_fmt="yuv420p", g=8, crf=30, ...),
  "depth_encoder": DepthEncoderConfig(vcodec="hevc", pix_fmt="gray12le", g=8, crf=0,
                                      depth_min=MIN_DEPTH, depth_max=MAX_DEPTH, shift=DEPTH_SHIFT,
                                      use_log=True, output_unit="m", ...),
  ```
- **Depth quantization** with `MIN_DEPTH = 0.01`, `MAX_DEPTH = 10.0`, `DEPTH_SHIFT = 3.5` (`OmniGibson/omnigibson/eval/utils/obs_utils.py:24-26`). It is a 12-bit log quantization: `norm = (log(d+3.5) - log(0.01+3.5)) / (log(10+3.5) - log(0.01+3.5))`, `q = clip(round(norm*4095), 0, 4095)`. Dequantization is `exp(q/4095*(log_max-log_min)+log_min) - 3.5`, clipped to [0.01, 10] (`obs_utils.py:29-95`) [code].
  - The docs say: "Depth videos are log-quantized during replay and dequantized back to metric depth values. Range [0, 10] meters", RGB at 720×720 head / 480×480 wrists, and `depth_linear` float32 (H,W) (`dataset.md:40-77`) [docs].
  - The depth videos were re-released after a bug fix (`updates.md:171`).
- **Replay sensor settings** (`replay_obs.py:100-114`): `horizontal_aperture` 40.0 on the head camera and modalities `["proprio","rgb","depth_linear"]` (`replay_obs.py:133-134`). This matches the eval-time `EVAL_HEAD_HORIZONTAL_APERTURE = 40.0` (`evaluator.py:54`) [code].
- **Intrinsics** (`eval_utils.py:80-90`): head K = [[306,0,360],[0,306,360],[0,0,1]] (720×720) and wrist K = [[388.6639,0,240],[0,388.6639,240],[0,0,1]] (480×480) [code]. Note that `CAMERA_EXTRINSICS` (l.92-102) is a copy-paste of the intrinsics and is not real extrinsics.

### 4.3 `observation.state` layout (61-D, R1Pro): verbatim from `OmniGibson/omnigibson/eval/utils/eval_utils.py:121-141` [code]

| slice | field | dims | meaning |
|---|---|---:|---|
| 0:3 | `base_qvel` | 3 | base velocity **in the robot-local frame** [vx, vy, wz] (fixed 07/27: `updates.md:170`; conversion `eval_utils.py:144-162`) |
| 3:10 | `arm_left_qpos` | 7 | left arm joint positions |
| 10:17 | `arm_left_qvel` | 7 | left arm joint velocities (raw sim; fixed 08/24, `updates.md:140-148`) |
| 17:20 | `eef_left_pos` | 3 | left EEF position (robot frame) |
| 20:24 | `eef_left_quat` | 4 | left EEF orientation quaternion |
| 24:26 | `gripper_left_qpos` | 2 | left gripper finger positions |
| 26:28 | `gripper_left_qvel` | 2 | left gripper finger velocities |
| 28:35 | `arm_right_qpos` | 7 | right arm joint positions |
| 35:42 | `arm_right_qvel` | 7 | right arm joint velocities |
| 42:45 | `eef_right_pos` | 3 | right EEF position |
| 45:49 | `eef_right_quat` | 4 | right EEF quaternion |
| 49:51 | `gripper_right_qpos` | 2 | right gripper positions |
| 51:53 | `gripper_right_qvel` | 2 | right gripper velocities |
| 53:57 | `trunk_qpos` | 4 | torso/trunk joint positions |
| 57:61 | `trunk_qvel` | 4 | torso/trunk joint velocities |

- The eval robot config `OmniGibson/omnigibson/eval/r1pro.yaml:13-28` lists `proprio_obs` in exactly this order, so the dataset state and the eval proprio are aligned [code].
- Unlike 2025 (a 256-D state including global pose, `v3.7.2 eval_utils.py:76-121`), the 2026 state has **no global pose**. `score_utils.py:186-196` still supports the 256-D 2025 layout.
- The OpenPI baseline uses only base_qvel 0:3, trunk_qpos 53:57, arm qpos 3:10 / 28:35, and gripper qpos 24:26 / 49:51 (`OPENPI/src/openpi/configs/robots/b1k.py:39-46`) [code].

### 4.4 `action` layout (23-D): `eval_utils.py:105-117` [code]

| slice | group | notes |
|---|---|---|
| 0:3 | base | HolonomicBaseJointController velocity cmd; input limits [-1,1]³ → output [-0.75,-0.75,-1]..[0.75,0.75,1] (`r1pro.yaml:62-72`) |
| 3:7 | torso (trunk) | absolute joint positions (`r1pro.yaml:73-80`) |
| 7:14 | left_arm | absolute joint positions (`use_delta_commands: false`, `r1pro.yaml:36-43`) |
| 14:15 | left_gripper | MultiFingerGripperController smooth (`r1pro.yaml:52-56`) |
| 15:22 | right_arm | absolute joint positions |
| 22:23 | right_gripper | |

The OpenPI config marks torso and arms `needs_delta_comp=True` and grippers `is_eef=True` (`OPENPI/src/openpi/configs/robots/b1k.py:31-38`).

### 4.5 Annotations

**2026, from the docs:**
- "`annotations` Language annotations for each episode." (`dataset.md:22-23`)
- "Uploaded per-episode language annotations for all 20,000 demonstrations." (`updates.md:176`)
- "skill/subtask annotations" (`index.md:129`)
- Skill statistics table (`dataset.md:80-152`): **31 unique skills**:
  > attach, chop, close door, close drawer, close lid, hand over, hang, hold, ignite, insert, move to, open door, open drawer, open lid, pick up from, place in, place in next to, place on, place on next to, place under, pour, press, push to, release, spray, sweep surface, tip over, turn off switch, turn on switch, turn to, wipe hard.

**The schema is not in the B1K repo** (NOT FOUND: grep of OmniGibson/joylo/docs for `skill_annotation` found nothing).

**2025 schema** [fetched: `StanfordVL/b1k-baselines@59cee35:tutorials/dataset.ipynb`, cell 15 output, file `annotations/task-0000/episode_00000010.json`]:
```python
{'data_folder': '',
 'meta_data': {'task_duration': 1776, 'valid_duration': [0, 1776]},
 'primitive_annotation': [{'frame_duration': [0, 1162], 'manipulating_object_id': ['radio_89'],
                           'memory_prefix': [], 'object_id': [['radio_89', 'coffee_table_koagbh_0']],
                           'primitive_description': ['pick up from'], 'primitive_id': [2],
                           'primitive_idx': 0, 'skill_idxes': [0, 1], 'spatial_prefix': []}, ...],
 'skill_annotation': [{'frame_duration': [0, 265], 'manipulating_object_id': [], 'memory_prefix': [],
                       'mp_ef': [], 'object_id': [['radio_89']], 'skill_description': ['move to'],
                       'skill_id': [1], 'skill_idx': 0, 'skill_type': ['navigation'], 'spatial_prefix': []},
                      {'frame_duration': [265, 1162], ..., 'skill_description': ['pick up from'],
                       'skill_id': [2], 'skill_idx': 1, 'skill_type': ['uncoordinated'], ...},
                      {'frame_duration': [1162, 1434], ..., 'skill_description': ['press'], 'skill_id': [67],
                       'skill_idx': 2, 'skill_type': ['coordinated'], ...},
                      {'frame_duration': [1434, 1776], ..., 'memory_prefix': ['back'],
                       'skill_description': ['place on'], 'skill_id': [3], 'skill_idx': 3,
                       'skill_type': ['uncoordinated'], ...}],
 'task_name': 'turning on radio'}
```
`skill_type` ∈ {navigation, uncoordinated, coordinated} in this example. `object_id` holds scene object names, not BDDL instance names.

**2026, third-party evidence** [snippet]:
- A derived dataset says its `b1k_skill_segments.parquet` has "one row per skill span, all 20,000 episodes: 407,054 rows".
- It lists columns such as `skill_idx, skill_type, skill_description, start, stop`, and says "35 skill verbs label 98.86% of … 210,916,774 frames".
- It says "only 3,325 of the 20,000 episodes are labelled end to end".
- It says "the primitive layer exists only on the 50 tasks carried over from the 2025 challenge".
- Source: [mahgoobi/b1k_2026_extras](https://huggingface.co/datasets/mahgoobi/b1k_2026_extras).

A paper snippet describes per-episode JSON `skill_annotation` entries with `skill_idx, frame_duration (half-open), skill_description, skill_type, object_id` ([arXiv 2608.04425](https://arxiv.org/pdf/2608.04425)). These are unverified.

### 4.6 Statistics

- 20,000 trajectories, 100 tasks, average trajectory duration 351.54 s (`dataset.md:91-113`).
- The plot mean is 5.86 min and the median 5.58 min (`docs/assets/challenge_2026/overall_demo_duration.png`, read visually). The histogram tail extends to about 19 min.
- "Total Skills 270,600 / Avg. Skills per Trajectory 27.06" (`dataset.md:99-109`) is **identical to the 2025 page** (`archive/2025/dataset.md:82-84`, which had 10,000 trajectories). For 20,000 trajectories it is inconsistent: 270,600 / 20,000 = 13.53. This is probably stale, and the snippet count is 407,054 spans.

---

## 5. Proposed DEV set (5 tasks)

**Selection criteria:**
- Cover the 5 required capabilities.
- Keep the default timeout short for throughput.
- Prefer granular partial credit (a credit cap of 1, several literals).
- Prefer tasks with official assets: turning_on_radio has baseline checkpoints for both π0.5 and GR00T (`docs/challenge/baselines.md:14-18`), and the OpenPI fork registers only that prompt (`OPENPI/src/openpi/configs/tasks/b1k.py:4-6`).

**Pure navigation is NOT FOUND.** No goal in the 100 BDDLs references the agent or a location predicate [derived]. The navigation probe is therefore the task where navigation dominates and manipulation is trivial.

| Role | Task (idx) | Scene / rooms | Demo mean (plot) | ~default max_steps | Goal literals (grounded) | Why |
|---|---|---|---:|---:|---|---|
| **Navigation-dominant** | `turning_out_all_lights_before_sleep` (66, new) | HSF; agent starts in corridor; switches in kitchen, utility_room, dining_room; lamps in living_room & utility_room (BDDL `:init`) | 334 s | ~15.0k | 5: `(not (toggled_on switch.n.01_{1,2,3}))`, `(not (toggled_on table_lamp.n.01_{1,2}))`; all on at init → cap 1, each room visit = +0.2 | widest room coverage of any task (8 room types), manipulation = press; special `LIGHT_EVAL_TASKS` sync (evaluator.py:52) |
| **Single-object manipulation** | `turning_on_radio` (0, 2025) | HDFL living_room; radio on table | 72 s (task_data 72) | ~3.2k (72×45 = 3,240) | 1: `(toggled_on radio_receiver.n.01_1)` | shortest-but-one demo; baseline checkpoints; instruction "Turn on the radio receiver that's on the table in the living room." (task_data.json:4-14); 2025 annotation example: move to → pick up from → press → place on |
| **Multi-object** | `picking_up_trash` (1, 2025) | HDFL; cans on living_room floor, trash can in kitchen, agent starts in kitchen | 176 s | ~7.9k | 3: `(inside can__of__soda.n.01_{1,2,3} ashcan.n.01_1)` → +1/3 per can | repeated pick-carry-place across 2 rooms; instruction available |
| **Articulated object** | `store_honey` (83, new) | Rs_int; jar on kitchen countertop, target cabinet in entryway, agent starts in living_room | 226 s (task_data 193) | ~10.2k | 1: `(inside jar__of__honey.n.01_1 cabinet.n.01_1)` | cheapest pure "open cabinet → place inside" task; small new scene |
| **Long-horizon** | `putting_dishes_away_after_cleaning` (11, 2025) | HSF kitchen | 365 s | ~16.4k | 10+: 8 × `(inside plate.n.04_i cabinet.n.01_1)` + `(not (open cabinet.n.01_1))` + wildcard cabinets closed → cap ≈0.8, +0.1 per plate | 8 sequential pick-place cycles with articulated cabinet; granular credit |

Approximate budget for one public instance of all 5: about 52.8k sim steps. At the documented 13.52 FPS for full-res RGB-D with random actions (`evaluation.md:366-369`), that is about 65 min of stepping plus 150–300 s of scene load per task (`evaluation.md:335`) [inferred].

**Alternates:**
- *Navigation*: `installing_a_fax_machine` (89, about 6.1k steps, 2 rooms, 2 literals). `carrying_out_garden_furniture` (74, 6 room types, about 16.6k).
- *Single-object*: `hanging_pictures` (34, about 3.6k, `attached`). `installing_smoke_detectors` (63, about 3.9k, 3 rooms, `attached`).
- *Multi-object*: `make_rose_centerpieces` (56, about 6.8k, 4 literals, 1 room).
- *Articulated*: `make_microwave_popcorn` (40, about 4.9k; microwave open/close plus toggle plus cooking transition, 2 literals). `rearranging_kitchen_furniture` (8, about 13.4k, cap 0.75). `bringing_water` (17, about 14.2k, fridge plus 2 rooms, cap 0.67).
- *Long-horizon / sequential dependencies*: `can_meat` (4, about 17.8k, 9 literals). It is a strong test of planning order (open cabinet → jars → open jars → bratwursts → close → return → close). **Caution:** its credit cap is about 0.44 without full success, because 5 of the 9 literals are already true at init.
- *Cooking*: `cook_broccolini` (96, about 6.9k, 11 literals).

**turning_on_radio full metadata** [code/derived]:
- idx 0, B100 Task ID 0 (verified via the 2025 mapping), scene `house_double_floor_lower`, room type `living_room`.
- `task_data.json` duration 72 s; plot mean 71.7 s; default max_steps ≈ 3.2k.
- BDDL objects: `radio_receiver.n.01_1`, `table.n.02_1`, `floor.n.01_1`, `agent.n.01_1`.
- Init: radio not toggled on, `ontop` table; table and floor `inroom living_room`; agent on floor.
- Goal: `(toggled_on ?radio_receiver.n.01_1)`. One grounded option of 1 literal, so q_score ∈ {0, 1}.
- Movable objects: 1. Openables: none. Toggleables: radio_receiver.
- Task-sampling note: "one failed grasp is allowed for turning_on_radio because the gripper needs to close to poke the button" (`task_sampling.md:140`).

---

## 6. Discrepancies between the docs pages and the code

1. **Instance-index semantics.** `evaluation.md:66` says "Indices 0-19 are public test instances; indices 20-39 are hidden". In the code, `--instance-indices` must lie in `range(20)` for each `--mode`; hidden instances are `--mode hidden_test` indices 0–19, i.e. IDs 321–340 (`eval.py:50-65`, `evaluator.py:73-87`).
2. **Hidden-instance count.** The docs say "hold out 10 more instances for final evaluation" (`evaluation.md:290`). The code has `NUM_HIDDEN_TEST_INSTANCES = 20` (`eval_utils.py:18-20`).
3. **Scorer denominator.** `score_utils.compute_final_q_score` divides each task's summed scores by `NUM_PUBLIC_TEST_INSTANCES` = 20 (`score_utils.py:300, 339-344`). The docs ask for 10 instances per task (`evaluation.md:282`, `submission.md:10`). A 10-instance submission scored with this script would be halved, unless the leaderboard uses a different scorer.
   - Also, `sanity_check_filename` still requires track ∈ {standard, privileged} (`score_utils.py:250`), which is the 2025 two-track naming. 2026 has a single track (`evaluation.md:5`).
4. **Skill statistics copied from 2025.** `dataset.md:99-109` gives 270,600 skills and 27.06 per trajectory, identical to `archive/2025/dataset.md:82-84`. That is inconsistent with 20,000 trajectories.
5. **New-task durations.** `task_data.json` `duration` for the 50 new tasks disagrees with the official per-task demo-duration plot: setting_the_table 727 vs about 594 s, cook_brussels_sprouts 687 vs about 522 s, tidying_living_room 316 vs about 419 s, setup_a_bar 350 vs about 447 s. The 2025 tasks agree within about 1 s. The gallery is therefore misleading for timeout estimates on new tasks.
6. **Missing instructions.** `task_data.json` has no `instruction` for the 50 new tasks, yet `baselines.md:39` says "See the dataset page for the full task list". dataset.md has no task list; the list is in the gallery (`docs/challenge/tasks/`, generated by `gen_task_pages.py`).
7. **RGB channels.** `dataset.md:49` says RGB shape (H,W,4). The LeRobot writer drops alpha and stores 3 channels (`lerobot_data_wrapper.py:161, 221-222`).
8. **Release tags.** `evaluation.md:34-35` says use `v3.9.3`. `updates.md:136, 166` (HEAD) say `v3.9.2`. `origin/main:updates.md:51` says `v3.9.3-post1` for final hidden evaluations, and `v3.9.3-post2` for custom robots (`origin/main:updates.md:32-36`).
9. **Annotation wording.** `index.md:129` promises "skill/subtask annotations", while `dataset.md` documents only "language annotations". No schema is given for 2026.
10. **Fake extrinsics.** `CAMERA_EXTRINSICS` in `eval_utils.py:92-102` duplicates the intrinsics matrices. Use `observation.robot2cam_pose.*` / `cam_rel_poses` instead.

## 7. Open questions

- The exact `B100_task_misc.csv`: the Task IDs for the new 50, and the room-instance lists per task. NOT FOUND (GitHub `wensi-ai/2026-challenge-task-instances` returned 403 or no access; HF blocked).
- The exact `task.jsonl` human `length` per task, which gives the exact default max_steps. Plot-based estimates are within about ±1 s × 45 steps.
- The 2026 annotation JSON schema and coverage. Is it identical to 2025, and is it full-episode for all 20,000 episodes? A snippet claims only 3,325 are labelled end to end.
- The contents of `meta/tasks.jsonl`, i.e. the official NL instructions for the 50 new tasks.
- Whether the zipped instance dataset ships `scene_test/private` (hidden).
- Which train instance each demo used. The episode metadata may record it; unverified.
- The wildcard expansion counts for idx 2, 3, 11, which give the exact q_score denominators.
- Whether the leaderboard scorer divides by 10 or by 20.
- Whether the credit caps from the init-satisfied heuristic hold on the actual sampled test instances.

## 8. Implications for agent design

- **Use the BDDL goal as the planner spec.**
  - BDDL is allowed at eval and identical there (`evaluation.md:28`).
  - The obs carries `task_id` (`evaluator.py:591`), so the agent can map it to the task name and then to `bddl3/bddl/activity_definitions/<task>/problem0.bddl`.
  - Ship our own copy of the index→name mapping (`task_data.json` order). Verify it against `B100_task_misc.csv` when available.
- **Ground goals with the official bddl code and pursue one option.** Use `compile_state` + `get_ground_state_options`. For forpairs-heavy tasks (assembling_gift_baskets, about 331k options) choose greedily. Track per-literal progress as an internal reward.
- **Optimize for the partial-credit rule.**
  - Only literals that become true count, and literals already true at start are never credited. So the planner should first secure the "newly needed" literals (place objects).
  - Full success (1.0) still requires every literal, including "keep closed" ones. Always finish with a "close all opened containers / turn off" sweep, because the jump from the cap (0.67–0.92) to 1.0 is large.
  - Avoid undoing literals that were achieved.
- **Instruction text.** Do not depend on `task_data.json` instructions; 50 tasks have none. Use `meta/tasks.jsonl` from the dataset, or generate NL from BDDL for the executor's language prompt.
- **Time budget.**
  - max_steps ≈ 45 × mean demo seconds, between about 3.2k and 39.1k steps.
  - The wall-clock rule is max_steps × 1 s, averaging ≥1 FPS including inference (`origin/main:updates.md:15`; per-group in `origin/group-time-budget:updates.md:15`), plus a 600 s first-query allowance (`origin/2026-eval:docs/challenge/evaluation.md`).
  - With π0.5 action chunks (horizon 32 in training, 16 executed by default, `OPENPI/docs/b1k.md:119, 277`), the high-level planner can spend several seconds per subgoal without violating 1 FPS on average.
- **Navigation.** Only the CSV room instances are loaded (partial scene). The navigation map and search space are those rooms. 29 tasks span more than one room type, and 24 new tasks are in 4 scenes that are new for 2026.
- **Skill vocabulary for the planner→executor interface.** The 31 annotated skills give a natural primitive set (move to, pick up from, place in/on/under/next to, open/close door/drawer/lid, turn on/off switch, press, pour, spray, wipe hard, sweep surface, chop, attach, hang, insert, …).
  - Skill segments (`frame_duration` plus `skill_description` plus `object_id`) can be used to fine-tune a language-conditioned executor on sub-skill prompts instead of whole-task prompts.
  - The 2025 schema is known; verify the 2026 one.
- **Executor I/O is fixed by the dataset.**
  - 61-D state with **no global pose**. A planner needing localization must derive it from RGB-D plus odometry; `base_qvel` is in the local frame.
  - 23-D absolute-joint action with a base velocity command.
  - 3 cameras. Depth must be dequantized from the 12-bit log encoding if training on the videos. Camera poses come from `observation.robot2cam_pose.*` (dataset) or `robot_r1::cam_rel_poses` (eval obs, 21-D, `evaluator.py:579-590`).
- **DEV set.** turning_on_radio (0), picking_up_trash (1), store_honey (83), putting_dishes_away_after_cleaning (11), turning_out_all_lights_before_sleep (66). The alternates in §5 swap in for throughput or coverage.

---

## Appendix: scripts and artifacts (all under `/tmp/claude-0/-home-user-VAGEN-Lite/9977f130-b514-5e2e-ab6e-7d7795066892/scratchpad/research/`)

- `scripts/analyze_bddl.py`: the first-pass grounding of all 100 goals.
- `scripts/task_metadata.py`: per-task metadata, written to `r6_task_metadata.json`. It imports the official `bddl3/bddl/condition_evaluation.py` (read-only) with a `future.utils.with_metaclass` shim in `shim/`.
- `scripts/init_satisfied.py`: the credit-cap heuristic, written to `r6_init_satisfied.json`.
- `scripts/plot_durations.py`: pixel measurement of the official per-task duration plot, written to `r6_plot_durations.json`.
- `r6_full_table.md`: the table in §1.4.
- `r6_dl/eval_utils_v372.py`: the 2025 code. `r6_dl/b1kbaselines_dataset.ipynb` and `r6_dl/nb_dump.txt`: the 2025 dataset tutorial with the annotation example.

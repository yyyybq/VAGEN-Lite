# R1: Official 2026 BEHAVIOR Challenge rules, as stated in the docs

Scope: the official challenge pages are generated from `B1K/docs/challenge/*.md` (mkdocs nav, `B1K/mkdocs.yml:105-113`).
I read `index.md`, `evaluation.md`, `baselines.md`, `dataset.md`, `submission.md` and `updates.md` at HEAD (tag `v3.9.3-post2`, a8247a8c), `origin/main` (7d6718cb, 2026-10-14), `origin/2026-eval` (38d2c9fb, 2026-10-19) and `origin/group-time-budget` (ad934aa6, 2026-10-09). I also fetched the same files at tags `v3.9.2` and `v3.9.3-post1` from raw.githubusercontent.com into `scratchpad/research/rawdocs/`. I checked the most important claims against the evaluator code (`OmniGibson/omnigibson/eval/`).

Labels:
- **[DOC]**: stated in the docs.
- **[CODE]**: verified in code.
- **[INF]**: my inference.
- **NOT FOUND**: looked for and absent.

`B1K` = `/home/user/stanfordvl/behavior-1k`.

Important date context: today is 2026-10-21. The documented submission deadline, **2026-10-16 11:59 PM AoE**, has already passed (`index.md:40`; `origin/main:docs/challenge/updates.md:13`). No ref mentions an extension (NOT FOUND in any `updates.md`). Winners are announced 11/04/2026 (`index.md:41`), and the event is at CoRL 2026 in Austin, Nov 9-12 (`origin/main:docs/challenge/index.md:52-54`).

---

## 0. How the docs differ between refs

| File | HEAD (v3.9.3-post2) vs origin/main | origin/main vs origin/2026-eval | origin/main vs group-time-budget |
|---|---|---|---|
| index.md | main adds the event details (CoRL 2026, Nov 9-12 2026, Austin TX) in place of "To be announced" (HEAD `index.md:52-54`) | identical (same blob 431c65d) | identical |
| evaluation.md | identical (blob cad1acd) | **2026-eval adds `--policy-endpoints`, an initialization allowance, a first-query allowance, per-rollout time accounting, and variable-length action chunks** (see §8 and §13) | identical |
| baselines.md | identical (blob 789dfc3) | identical | identical |
| dataset.md | identical (blob 12bce3d) | identical | identical |
| submission.md | identical (blob d7cfd24) | identical | identical |
| updates.md | **main adds the 09/28, 10/07 and 10/09 entries** (HEAD's latest entry is 08/24) | 2026-eval rewrites 10/09 items 2-4 (time budget per rollout) | branch rewrites 10/09 items 2-4 as a **group** budget `N × max_steps` |

Older tags:
- `v3.9.2` evaluation.md says "Use the `v3.9.2` tag". It has no `--num-envs` row and no action-chunk section, and its instance-indices row lacks "Select exactly one instance per logical environment" (diff of `rawdocs/v3.9.2/evaluation.md` against HEAD).
- `v3.9.2` baselines.md clones `-b v3.9.2`.
- `v3.9.3-post1` docs equal HEAD's, except that HEAD adds the `eval.use_presampled_pose_key` paragraph (`evaluation.md:218`). HEAD and `v3.9.3-post1` have the same `updates.md`, whose latest entry is 08/24.

**Note:** HEAD (v3.9.3-post2) docs say "use `v3.9.2`" in updates.md (`updates.md:11`, `updates.md:41`) but "use `v3.9.3`" in evaluation.md and baselines.md (`evaluation.md:34-35`, `baselines.md:43-46`). The 09/28 and 10/07 entries that resolve this exist only on `origin/main` and `origin/2026-eval`.

---

## 1. Number of tasks and the task list

- **2026: 100 tasks.**
  - "solve **100 full-length household tasks**" (`index.md:3`)
  - "Tasks | 100 full-length household tasks" (`index.md:99-100`)
  - "Task success score: Averaged across 100 tasks." (`evaluation.md:230`)
  - "Total Tasks | 100" (`dataset.md:95-96`)
  - "Full evaluation size: 100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs." (`submission.md:10`) [DOC]
- **2025: 50 tasks.**
  - "Task success score: Averaged across 50 tasks." (`archive/2025/evaluation.md:174`)
  - "50 full-length household tasks" (`archive/2025/index.md:3,91`) [DOC]
- The 2026 set is the 50 tasks from 2025 plus 50 new ones. "Browse through all 100 household tasks in our 2026 challenge (the 50 tasks from 2025 plus 50 new ones)" (`B1K/docs/gen_task_pages.py:38-40`; also `docs/gen_2026_task_data.py:3-5`). [CODE/DOC-generator]
- **Task list source (docs):**
  - The "Demo Gallery" at `challenge/tasks/index.md` (`index.md:125`; `evaluation.md:58`).
  - It is generated at build time from `B1K/docs/challenge/task_data.json` (`docs/gen_task_pages.py:33-47,605-615`).
  - `task_data.json` has exactly 100 entries (verified by script). Index 0 is `turning_on_radio` and index 99 is `sorting_books_on_shelf`.
  - 7 distinct `scene_model`s: house_single_floor 34, house_double_floor_lower 32, house_double_floor_upper 10, restaurant_diner 8, Rs_int 6, hotel_suite_large 5, office_cubicles_right 5. This matches "7 scenes, including 4 new scenes" (`index.md:103-104`). [CODE/INF]
  - Only the first 50 entries have an `instruction` field. Entries 50-99 lack it (verified by script).
- **Task list source (evaluator):**
  - `TASK_NAMES_TO_INDICES` is loaded from `gm.DATA_PATH/2026-challenge-task-instances/metadata/B100_task_misc.csv`, using the columns "Task" and "Task ID" and the room column "Rooms to inlcude" [sic] (`OmniGibson/omnigibson/eval/utils/eval_utils.py:60-77`). [CODE]
  - That CSV is **NOT present locally**: `find / -name B100_task_misc.csv` returned nothing. It ships in the zipped dataset `2026-challenge-task-instances` (`OmniGibson/omnigibson/utils/asset_utils.py:590-592`, from HF `behavior-1k/zipped-datasets`, `asset_utils.py:435-437`). [CODE]
  - The demos are stored as one chunk per task: "chunk-000 is the first task" (`baselines.md:55-58`). [DOC]
- Natural-language task descriptions:
  - "Added `meta/tasks.jsonl` with natural-language task descriptions for all 100 challenge tasks. The first 50 tasks follow the 2025 challenge descriptions ...; the remaining 50 were derived from the 2026 annotations and task definitions." (`updates.md:50`)
  - "Uploaded per-episode language annotations for all 20,000 demonstrations." (`updates.md:51`) [DOC]

## 2. Tracks and allowed observations

- **2026: one track.** "For the 2026 BEHAVIOR Challenge, there is a single evaluation track" (`evaluation.md:5`); "Evaluation track | One track using RGB + depth + proprioception" (`index.md:107-108`). [DOC]
  - "Participants are restricted to robot onboard observations for policy inputs." (`evaluation.md:11`)
  - "Allowed policy inputs | RGB + depth + proprioception" (`evaluation.md:14-15`)
  - "Not allowed during evaluation | No ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only privileged information during evaluation." (`evaluation.md:22-23`)
  - "Submitted evaluation wrappers must expose only RGB, depth, and proprioception to the policy. Include the wrapper code in your submission; the organizers will manually inspect it ... and that the environment is not manipulated directly, e.g. by teleporting the robot or changing object states." (`evaluation.md:144`; repeated at `submission.md:42`) [DOC]
- **2025 had two tracks:**
  - Standard: "RGB + depth + segmentation (semantic, instance, instance id) + proprioception ... robot global poses are NOT allowed; No object state" (`archive/2025/evaluation.md:9-11`).
  - Privileged information: "allowed to query the simulator for any privileged information, such as target object poses, full-scene point cloud ..., robot global poses" (`archive/2025/evaluation.md:13`). [DOC]
  - **Change in 2026:** there is no privileged track, and segmentation is no longer allowed. [DOC/INF]
- **Training-time privileged info is allowed:** "You are allowed to use privileged information during training (e.g. other observation modalities, task info, etc.), so long as you are not using it during challenge-track evaluation." (`evaluation.md:28`) [DOC]
- **BDDL task definition at test time is allowed:** "BDDL task definitions can be used and are identical during evaluation." (`evaluation.md:28`) [DOC]
- **Extra data is allowed:** "You may also collect additional data yourself via teleoperation, RL, scripted policies, or other approaches." (`evaluation.md:28`) [DOC]
  - The 2025 rule "you may **not** collect data on evaluation instances" (`archive/2025/evaluation.md:15`) is **NOT FOUND** in the 2026 evaluation.md. [DOC/INF]
- **Task instruction text at test time:** the 2026 docs do not explicitly allow or forbid it (NOT FOUND in evaluation.md or submission.md).
  - The evaluator itself puts an integer `task_id` into every observation: `obs["task_id"] = th.tensor([TASK_NAMES_TO_INDICES[self.cfg.task.name]], dtype=th.int64)` (`OmniGibson/omnigibson/eval/evaluator.py:591`). [CODE]
  - The official π0.5 baseline server takes `--task b1k/$TASK_NAME` and loads a text prompt from `TASK_REGISTRY` (`baselines.md:174-182`; `OPENPI/scripts/b1k/serve_b1k.py:47,60,71`). [DOC/CODE]
  - [INF] Knowing the task identity and its NL description at test time is therefore sanctioned in practice. Ask the organizers if you plan to use extra text beyond the BDDL or `tasks.jsonl`.
- **Other fields the evaluator adds to the observation:** `"{robot.name}::cam_rel_poses"`, the camera poses relative to the robot base (`evaluator.py:578-590`). [CODE] The policy also receives the flattened robot observations (proprio plus the wrapper's camera modalities). `cfg["task"]["include_obs"] = False` (`evaluator.py:290`). [CODE]
- **Wrappers:**
  - `DefaultWrapper` gives RGB only, at 224×224 for all cameras (`evaluation.md:134-137`; `OmniGibson/omnigibson/eval/wrappers/default_wrapper.py:4,23-28`).
  - `RGBDFullResWrapper` is "Official RGB + depth challenge observations, with a 720 x 720 head camera and 480 x 480 wrist cameras" (`evaluation.md:138-141`). In code it returns modalities `["rgb", "depth_linear"]` (`wrappers/rgbd_full_res_wrapper.py:19-22`; `eval_utils.py:10-11`). [DOC/CODE]
  - The **default** `--env-wrapper` is `DefaultWrapper` (`evaluation.md:82`; `eval.py:88-92`). The docs say "use `omnigibson.eval.wrappers.RGBDFullResWrapper` for official RGB + depth challenge-track evaluation" (`evaluation.md:82`). [DOC/CODE]

## 3. External API and LLM calls

- "There are no restrictions on the type of policy used. Methods such as IL, RL, or TAMP are all allowed. Additional components like SLAM or LLM-based querying are also permitted, provided the policy follows the challenge-track observation restrictions during evaluation. If a submission depends on external model-query APIs, participants must provide the credentials, quota, and serving configuration needed for evaluation; the organizers will not cover external API usage costs." (`evaluation.md:30`) [DOC]
- `index.md:142` lists "LLM-assisted policies" among the intended approaches. [DOC]
- 10/07 update: the "Inference Speed Form" (https://forms.gle/5GNqc2eNhFTx4UhN8) asks for "expected average inference speed, including time spent on external API calls if applicable" (`origin/main:docs/challenge/updates.md:41`). [DOC]
- Hardware: "The submitted model should run on a single 24GB VRAM GPU. Final evaluation will use GPUs such as RTX 3090, A5000, and TitanRTX." (`submission.md:86`, Docker option). [DOC]

## 4. Robot embodiment and action space

- "Robot embodiment | Not fixed. Participants may use the default R1Pro robot or provide their own OmniGibson-supported robot through a custom robot configuration file." (`evaluation.md:18-19`; `--robot-config` described at `evaluation.md:148-156`). [DOC]
- `controller_config` may use "any robot controllers supported by OmniGibson, such as joint, IK, base, or gripper controllers, as long as the resulting action array returned by your policy matches `robot.action_dim`" (`evaluation.md:171-172`). [DOC]
- "The evaluator overwrites the robot `position` and `orientation` at runtime with the task instance's prescribed start pose." (`evaluation.md:179-180`) [DOC]
- Default config: `OmniGibson/omnigibson/eval/r1pro.yaml`. It sets `grasping_mode: assisted` (`r1pro.yaml:29`), uses joint-position arms and trunk, `HolonomicBaseJointController` velocity for the base, and `MultiFingerGripperController` smooth grippers (`r1pro.yaml:35-80`). Its proprio list is at `r1pro.yaml:13-28`. [CODE]
- Code nuance: the evaluator **overwrites** `obs_modalities` with `["proprio", *camera_spec["modalities"]]` taken from the wrapper (`evaluator.py:269`). Setting `obs_modalities` in the robot config is therefore ignored for the cameras, even though `evaluation.md:176` says it "may be customized". [CODE]
- The robot config must be included in the submission (`evaluation.md:183-184`; `submission.md:14,39-41,114-116`). [DOC]

## 5. Evaluation instances: public and hidden test

- **Docs:**
  - "Indices into the task's test instance list. Use indices `0 1 2 3 4 5 6 7 8 9` for reported evaluation results. Indices `0-19` are public test instances; indices `20-39` are hidden instances reserved for final evaluation. Select exactly one instance per logical environment in the evaluation batch." (`evaluation.md:66`)
  - Protocol table: "we provide 20 extra configuration instances for each task. Use the **first 10 public instances**, corresponding to instance indices `0-9` ... The **remaining 10 public instances**, indices `10-19`, are not used for leaderboard reporting and may serve as a test set" (`evaluation.md:282`)
  - "Final evaluation | We will hold out 10 more instances for final evaluation. After we freeze the leaderboard upon submission deadline, we will evaluate the top-5 solutions on the leaderboard using these instances." (`evaluation.md:290`)
  - "Each instance differs in terms of initial object states and initial robot poses." (`evaluation.md:294`) [DOC]
  - 09/28 update: "we will evaluate the top few submissions on hidden test instances before the November announcement. These hidden-test results will determine the final leaderboard ranking." (`origin/main:docs/challenge/updates.md:59`)
  - 10/09 update: "hidden-test scores will replace public-instance scores for final ranking." (`origin/main:docs/challenge/updates.md:23`) [DOC]
- **Code (HEAD):**
  - `NUM_TEST_INSTANCES = 40`, `NUM_PUBLIC_TEST_INSTANCES = 20`, `NUM_HIDDEN_TEST_INSTANCES = 20`, `TEST_INSTANCE_IDS = list(range(301, 341))` (`eval_utils.py:18-21`).
  - `--mode {train,public_test,hidden_test}`, default `public_test` (`eval.py:60-65`).
  - `--instance-indices` "for public_test / hidden_test these index into that 20-instance split" (`eval.py:50-59`).
  - `resolve_instance_ids` asserts `indices ⊆ range(20)` for the chosen split and maps public indices to IDs **301-320** and hidden indices to **321-340** (`evaluator.py:73-87`). The same code is in v3.9.2 (`rawcode/v3.9.2/evaluator.py:73-87`). [CODE]
  - Hidden instances load from `2026-challenge-task-instances/scene_test/private/...` and public ones from `scene_test/public/...` (`OmniGibson/omnigibson/utils/asset_utils.py:137-146`). [CODE]
  - [INF] The hidden JSONs may not be in the public download. Not verified, because the dataset is absent locally.
- Train instances: with `--mode train`, the indices are raw training instance IDs (`evaluator.py:75-76`). [CODE] Demos: "200 per task" (`evaluation.md:278`). [DOC]

## 6. Rollouts per instance and cherry-picking

- "You should evaluate your policy 1 time on each instance, using the default `1.5x` mean-human-length timeouts provided by our evaluation script." (`evaluation.md:282`) [DOC]
- Full submission: "100 tasks x 10 instances x 1 rollout = 1,000 rollout outputs." (`submission.md:10`) [DOC]
- `--num-rollouts` defaults to 1 (`eval.py:66`). Each rollout re-runs the same batch (`eval.py:181-190`). [CODE]
- **No cherry-picking:**
  - "Participants should not cherry-pick rollout results for individual instances or assemble the best outcomes across runs, instances, or tasks to improve the reported success rate." (`evaluation.md:286`)
  - "No cherry-picking rollout results ... Submissions should report the prescribed rollout for each instance rather than selecting the best outcomes across repeated runs, instances, or tasks." (`submission.md:13`)
  - "Evaluating the same task instances repeatedly and assembling a submission from the best result of each rollout is **not allowed**. Submit results from a single evaluation run of your final policy, with one rollout per submitted task instance." (`origin/main:docs/challenge/updates.md:25`) [DOC]
- "Multiple checkpoints from the same team and model family are considered one entry." (`submission.md:12`) [DOC]
- "Do not modify the output JSON files or rollout videos in any way." (`submission.md:74-76`) [DOC]
- Leaderboard farming: "we will hide leaderboard scores during the final week of submissions and restore their visibility after the deadline." (`origin/main:docs/challenge/updates.md:23`) "We will use each team's **latest valid submission** received before the deadline" (`origin/main:docs/challenge/updates.md:13`). [DOC]

## 7. Max steps (episode length)

- "`--max-steps` | Optional episode timeout in simulator steps. If omitted, the evaluator uses a task-specific default timeout equal to `1.5x` the mean human demonstration length." (`evaluation.md:78`) [DOC]
- Code: `EVAL_TIMEOUT_MULTIPLIER = 1.5` (`eval_utils.py:17`) and `max_steps = int(self.human_stats["length"] * EVAL_TIMEOUT_MULTIPLIER)` (`evaluator.py:281-286`). `human_stats["length"]` is the per-task mean episode length from `2026-challenge-task-instances/metadata/task.jsonl` (`score_utils.py:25-46,215-233`). [CODE]
- 2025 used 2× (`archive/2025/evaluation.md:55,192`). [DOC]
- Simulator rate: the sample JSON shows `"simulator_steps": 500` mapping to `"simulator_time": 16.6666666667` (`submission.md:51,66-69`), i.e. 30 Hz. [DOC/INF]
- [INF] "approximately 10.85 hours of episode timeout for the longest task" (`origin/main:docs/challenge/updates.md:15`) at 1 s per step implies max_steps ≈ 39,060 for that task. That fits the gallery's longest `duration` of 869 s (`assembling_gift_baskets`, `task_data.json`): 869 × 30 × 1.5 ≈ 39,105 steps. The actual per-task `task.jsonl` lengths are NOT FOUND locally.

## 8. Timeouts: episode budget, per-step limit, reconnection, FPS

### 8.1 As announced on origin/main (10/09 update)

`origin/main:docs/challenge/updates.md:15-19`:

> 2. **Average evaluation speed:** Each rollout must average at least **1 FPS**, measured from one action query to the next, including inference, communication, and simulator stepping. We enforce this through an episode timeout of `max_steps × 1 second`, excluding scene startup. Individual steps may take longer than one second. This corresponds to approximately **10.85 hours** of episode timeout for the longest task.
> 3. **Per-step timeout:** Each action-query round trip must complete within **600 seconds** and within the remaining episode time.
> 4. **Reconnections:** If the policy-server connection is lost, we will attempt to reconnect up to **three times** before stopping the rollout and counting it as a failure. Reconnection time counts toward the episode timeout.

Item 5 (`:21`): IP-based teams can DM the organizers for a connection and speed test. [DOC]

The 09/28 update (`origin/main:docs/challenge/updates.md:57,63`) adds:
- "For IP-based submissions, we will attempt to reconnect up to three times after a network disconnection. If reconnection fails, the rollout will count as a failure."
- "We will introduce a global episode timer and a per-action response timer". [DOC]

### 8.2 origin/2026-eval rewrite (latest docs, 10/19)

`origin/2026-eval:docs/challenge/updates.md:15-19`:

> 2. **Evaluation time budget:** Each task has a budget of `max_steps × 1 second` per rollout, excluding scene startup. When rollouts share one GPU, each rollout is charged for its own policy queries, including reconnections, plus an equal share of each shared simulator step among rollouts active for that step. Waiting for another port's policy response is not charged to it. The simulator share changes as rollouts finish or fail. For the longest task, the per-rollout budget is approximately **10.85 hours** of accounted time.
> 3. ... within **600 seconds** and within that rollout's remaining time budget.
> 4. ... up to **three times** before stopping the affected rollout ... Reconnection time counts toward that rollout's time budget.

`origin/2026-eval:docs/challenge/evaluation.md:111-115` adds:

> After scene initialization and instance loading, initial policy connection and sending the rollout reset have a separate 600-second allowance per endpoint, excluded from the episode budget. The current reset message has no acknowledgment. The evaluator does not restart participant servers.
>
> Each rollout also receives a **first-query allowance**: the first action query must finish within 600 seconds, but its duration is excluded from the episode budget. This allowance includes any checkpoint loading, GPU transfer, reset processing, warm-up, and inference performed during that query. ... A failed first query fails the affected rollout; it does not grant a replacement free query.
>
> Episode accounting starts after the first query completes and before its first action is executed. In multi-port evaluation, each rollout has an accounted time budget of max_steps seconds. The evaluator charges each active rollout for subsequent policy query time, including reconnections, plus the shared simulator step time divided by the total number of environments in the batch. This divisor stays fixed because finished or failed environments remain in the shared simulation ... If the websocket connection drops, the evaluator attempts to reconnect up to three times and resends the current observation without advancing the simulation step; reconnection does not restart the episode budget or grant another allowance. ... A failed or over-budget rollout receives zero score; other active rollouts continue. ... Recorded evaluation time and evaluation FPS exclude initialization and the first-query allowance.

### 8.3 group-time-budget branch (10/09, superseded)

The budget is per group: "When we evaluate `N` rollouts of the same task together on one GPU, the group has a wall-clock budget of `N × max_steps` seconds ... a group of 10 approximately **108.6 hours**." (`origin/group-time-budget:docs/challenge/updates.md:15`) [DOC] This branch changes only docs; there is no eval code diff vs main (`git diff --stat origin/main origin/group-time-budget -- OmniGibson/omnigibson/eval/` is empty). [CODE]

### 8.4 Code reality

- **HEAD (v3.9.3-post2) and origin/main do NOT implement** the 600 s per-step limit, the 3-reconnect limit or the wall-clock episode budget.
  - `WebsocketClientPolicy.act` has `max_retries = 2`, and it reconnects only if `allow_reconnect` (`network_utils.py:127-170`).
  - The health check loops forever (`network_utils.py:82-92`).
  - `eval.py` does not pass `allow_reconnect`, so it defaults to False (`eval.py:144-150`; `policies.py:63`).
  - The only cap is `max_steps` in simulator steps (`evaluator.py:281-289`). [CODE]
  - The eval code on origin/main is identical to HEAD (`git diff --stat HEAD origin/main -- OmniGibson/omnigibson/eval/` is empty). [CODE]
- **origin/2026-eval implements them:**
  - `MAX_RECONNECTS_PER_ROLLOUT = 3`, `POLICY_RESPONSE_TIMEOUT = 600` (`origin/2026-eval:OmniGibson/omnigibson/eval/utils/network_utils.py:31-32`)
  - `POLICY_INITIALIZATION_TIMEOUT = 600.0` (`origin/2026-eval:.../evaluator.py:58`)
  - the first query is uncharged (`:783-796`)
  - the budget is set to `self.max_steps` seconds (`:796`)
  - single-endpoint mode uses one group wall-clock deadline `time_limit_seconds=self.max_steps` (`:884`)
  - the sim-time share uses `share = seconds / len(self.clients)`, a fixed divisor (`origin/2026-eval:.../policies.py:218-220`)
  - a failed rollout gets `q_score.final = 0.0` and a `failure_reason` (`origin/2026-eval:.../evaluator.py:836-852`)
  - `eval.py` sets `allow_reconnect: True` (`origin/2026-eval:.../eval.py:178,187`). [CODE]
- [INF] Final hidden-test runs will probably use the 2026-eval logic, or the v3.9.3-post1 tag plus organizer tooling. The docs say "will use `v3.9.3-post1` for final hidden-test evaluations" (`origin/main:docs/challenge/updates.md:51`), and the time-budget code is not in any tag we have.

### 8.5 FPS figures in the docs

- Requirement: average ≥ 1 FPS (`origin/main:updates.md:15`). [DOC]
- Benchmarks with random actions on an RTX 4090: RGB 224/224 → 24.55 FPS; RGB 720/480 → 20.62; RGB+depth 224/224 → 16.55; RGB+depth 720/480 → 13.52 (`evaluation.md:351-368`). "Scene Load Time: Approximately 150-300 seconds" (`evaluation.md:335`). [DOC]

## 9. Metrics and aggregation

- **Primary metric:**
  - "**Task success score:** Averaged across 100 tasks. **Calculation:** Partial successes = (Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates)." (`evaluation.md:229-231`)
  - "Ranking metric | Average task success score with BDDL partial credit" (`index.md:119-120`)
  - "The success score (**Q**) is the metric used for ranking submissions. If two submissions achieve the same score, secondary metrics will be used to break ties." (`evaluation.md:243`) [DOC]
- **Secondary metrics:** "Simulated time", "Distance navigated" (base) and "Displacement of end effectors/hands" (`evaluation.md:234-237`). "Secondary metrics will be normalized using human averages from 200 demonstrations per task." (`evaluation.md:241`) [DOC]
- **Q-score in code:** `compute_q_score` returns 1.0 on success. Otherwise it is the fraction of goal predicates **not satisfied at episode start but satisfied now**, maximized over alternative goal options (`OmniGibson/omnigibson/metrics/task_metric.py:6-28`). [CODE] This differs from the docs' wording (see Discrepancies).
- `normalized_time = human_steps / timesteps` (`task_metric.py:76-81`). [CODE]
- **Aggregation in code** (`score_utils.compute_final_q_score`, `score_utils.py:271-372`):
  - Per task: Q is the sum over instances divided by `n_instances_per_task`, which is `NUM_PUBLIC_TEST_INSTANCES` (=20) for "public" and 20 for "hidden".
  - Overall: the mean over all tasks in `TASK_NAMES_TO_INDICES`.
  - `task_sr` counts rollouts with `q == 1`.
  - `time_score = 1.5/0.5 − 1/(0.5·normalized_time)`.
  - Missing files contribute 0. [CODE]
- **Missing instances:**
  - "Partial submissions are allowed. Missing rollout instances count as zero in the final score." (`submission.md:11`)
  - "Partial submissions are allowed, including results for a subset of tasks or instances. Hidden-test evaluation will follow the tasks and instances covered by your submission; unsubmitted tasks and instances will count as zero." (`origin/main:docs/challenge/updates.md:61`) [DOC]
- Per-rollout JSON:
  - "Each successful rollout produces a JSON result containing `q_score`, `time`, `agent_distance`, and normalized efficiency metrics." (`evaluation.md:129`)
  - The sample has keys task, instance_id, rollout_id, steps, success, agent_distance{base,left,right}, normalized_agent_distance, q_score{final}, time{simulator_steps, simulator_time, normalized_time} (`submission.md:46-71`). [DOC]
  - On 2026-eval, `time.policy_time`, `sim_time`, `evaluation_time` and `evaluation_fps` are added for multi-port runs, plus `failure_reason` (`origin/2026-eval:.../evaluator.py:836-866`). [CODE]

## 10. Version tags

| Source | Instruction |
|---|---|
| updates 07/27 (`updates.md:41`) | use `v3.9.2` rather than `v3.9.0` |
| updates 08/24 (`updates.md:11`) | use `v3.9.2` for challenge evaluation |
| evaluation.md HEAD/main/2026-eval (`:34-35`) | "Use the `v3.9.3` tag ... not the older `v3.9.0` tag" |
| baselines.md HEAD (`:43-48`) | `git clone -b v3.9.3 https://github.com/StanfordVL/BEHAVIOR-1K.git` then `./setup.sh --new-env --omnigibson --bddl --joylo --dataset --eval` |
| updates 09/28 (`origin/main:updates.md:51`) | "Please use `v3.9.3-post1` ... We accept final results evaluated on either `v3.9.2` or the `v3.9.3` series, and will use `v3.9.3-post1` for final hidden-test evaluations." |
| updates 10/07 (`origin/main:updates.md:33,37`) | "If your submission uses a custom robot, please update to `v3.9.3-post2`. Otherwise, no update is required"; post2 = "a single bug fix for custom robot evaluation" |

[CODE/INF] `git ls-remote --tags origin` (run 2026-10-21) lists `v3.9.0`, `v3.9.1`, `v3.9.2`, `v3.9.3-post1` (bd049de3) and `v3.9.3-post2` (a8247a8c). **There is no plain `v3.9.3` tag**, so `git clone -b v3.9.3` as written in `baselines.md:46` would fail. Use `v3.9.3-post2` (HEAD).

## 11. Exact eval command lines in the docs

1. `evaluation.md:39-48` (HEAD, main and 2026-eval):
```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --host 127.0.0.1 \
  --port 8000 \
  --instance-indices 0 \
  --num-rollouts 1 \
  --output-dir outputs/b1k_eval \
  --write-video
```
2. Custom robot (`evaluation.md:150-156`):
```bash
python -m omnigibson.eval.eval \
  --task-name turning_on_radio \
  --robot-config path/to/my_robot.yaml \
  --host 127.0.0.1 \
  --port 8000
```
3. Baselines evaluation (`baselines.md:301-312`):
```bash
conda activate behavior
cd $PATH_TO_BEHAVIOR_1K
export LOG_PATH=./eval_logs/$TASK_NAME     # where metrics and videos are written

python -m omnigibson.eval.eval \
    --task-name $TASK_NAME \
    --host 127.0.0.1 \
    --port 8000 \
    --output-dir $LOG_PATH \
    --write-video
```
4. Code docstring example (`eval.py:9-16`):
```bash
python -m omnigibson.eval.eval --task-name turning_on_radio --robot-config omnigibson/eval/r1pro.yaml \
  --mode public_test --host 127.0.0.1 --port 8000 --instance-indices 0 --max-steps 500 \
  --output-dir outputs/b1k_eval --write-video
```
5. π0.5 policy server (`baselines.md:171-183`):
```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/$TASK_NAME --repo-id $REPO_ID \
  --policy.config pi05_b1k --policy.dir $PATH_TO_CKPT --control_mode receding_horizon --action_horizon 16 --port 8000
```
6. GR00T server (`baselines.md:286-292`): `python scripts/b1k/serve_b1k.py --model-path $PATH_TO_CKPT --modality-config-path examples/b1k/r1pro.py --embodiment-tag NEW_EMBODIMENT --host 127.0.0.1 --port 8000`. It applies temporal ensembling by default and has a health check at `curl -s http://127.0.0.1:8000/healthz` (`baselines.md:295`).

[INF] None of the documented example commands pass `--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper`, so they run the low-res RGB-only `DefaultWrapper`. They also omit `--mode` (default `public_test`), and #3 omits `--instance-indices`, so only index 0 (ID 301) runs. **[INF] A rules-compliant reporting run per task:**
```bash
python -m omnigibson.eval.eval --task-name $TASK --env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper \
  --mode public_test --instance-indices 0 1 2 3 4 5 6 7 8 9 --num-envs 10 --num-rollouts 1 \
  --host 127.0.0.1 --port 8000 --output-dir $OUT --write-video
```
Alternatively, run 10 separate invocations with `--num-envs 1`. The wrapper choice is the participant's, but only RGB, depth and proprio are permitted (`evaluation.md:144`).

Full flag list (HEAD `eval.py:38-112`): `--task-name` (required), `--host` (127.0.0.1), `--port` (8000), `--robot-config`, `--instance-indices` ([0]), `--mode` (public_test), `--num-rollouts` (1), `--num-envs` (1), `--replay-action-chunk-size` (0), `--max-steps` (None → 1.5× human), `--env-wrapper` (DefaultWrapper), `--policy {websocket,local}` (websocket), `--output-dir` (/tmp/b1k_eval), `--write-video/--no-write-video` (False), `--video-fps` (30), `--headless/--no-headless` (True). 2026-eval adds `--policy-scheme {ws,wss}` and `--policy-endpoints HOST:PORT ...` (`origin/2026-eval:.../eval.py:58-66`). [CODE] `partial_scene_load` is hardcoded True (`eval.py:160`), which loads the task's rooms from `B100_task_misc.csv` (`evaluator.py:253-254`; `updates.md:25`). [CODE]

**Outputs:**
- JSON goes under `<output-dir>/json/` (`evaluation.md:86`; `eval.py:171-172`). The file is named `{task_name}_{instance_id}_{rollout_id}.json`, and `instance_id` is the actual ID (301-340), not the index (`evaluator.py:681`). [CODE]
- MP4 goes under `<output-dir>/videos/` (`evaluation.md:90`; `eval.py:173-175`), named `{task}_{instance_id}_{rollout_id}.mp4`. The frame is 448×672 (rows×cols order not verified), with the 2 wrist views (224×224) stacked beside the head view (448×448) (`evaluator.py:570-573,594-615`). [CODE]
- Videos are "required for challenge submissions" (`evaluation.md:90`; `submission.md:28`). `--write-video` requires the `head`, `left_wrist` and `right_wrist` roles (`evaluation.md:216`). [DOC]

## 12. Policy server protocol

- "The evaluator connects to the policy server at `--host` and `--port` ... The websocket interface is implemented by the evaluation utilities adapted from openpi" (`evaluation.md:50`). "The default port is `8000`. The evaluator waits for the server health check at `/healthz`, then opens the websocket connection." (`evaluation.md:62`) [DOC]
- "The evaluator sends flattened observations to the policy server. The server should return a msgpack-encoded response containing an `action` array with the robot action for the current step. The helper server implementation is `WebsocketPolicyServer` in `OmniGibson/omnigibson/eval/utils/network_utils.py`, and the evaluator-side client is `omnigibson.eval.policies.WebsocketPolicy`." (`evaluation.md:103`) [DOC]
- **Code details (HEAD):**
  - On connect, the server first sends a msgpack metadata dict (`network_utils.py:219`).
  - The client sends `{"reset": True}` on episode reset, with no reply (`network_utils.py:172-179,226-228`).
  - The health endpoint is `/healthz` (`network_utils.py:268-276`).
  - The server replies with `{"action": ..., "server_timing": {...}}` (`network_utils.py:233-243`).
  - Observations are batched with a leading `num_envs` dimension, even when `num_envs=1` (`evaluator.py:411-421,430`). [CODE]
- **Optional action-chunk replay:**
  - HEAD: `--replay-action-chunk-size K` adds `__action_chunk_size__` to the request (`network_utils.py:29`). The response must include `action_chunk` of shape `(..., K, action_dim)` with `action_chunk[...,0,:] == action` (`evaluation.md:105-127`; enforced at `network_utils.py:147-157`).
  - 2026-eval: L ≤ K actions are allowed, and the server may vary L (`origin/2026-eval:docs/challenge/evaluation.md:119-143`).
  - "Do not enable it for temporal ensembling or for a server that repeatedly re-runs or blends the policy on the same stale observation" (`evaluation.md:124-127`). [DOC/CODE]
- **Final evaluation transport:**
  - "Docker-based evaluation | **Recommended.** Submit a Docker image that serves your policy. We run OmniGibson outside the container and connect to your policy through the WebSocket policy client." (`submission.md:84-86`)
  - "IP address-based evaluation | ... We only accept IP address-based submissions that expose at least 50 ports for parallel evaluation." (`submission.md:89-91`) [DOC]

## 13. num-envs, vectorized evaluation, tiled rendering

- `--num-envs`: "Number of logical environments processed together as one evaluation batch. This must equal the number of values passed to `--instance-indices`." (`evaluation.md:70`) This is enforced at `eval.py:132-136`. [DOC/CODE]
- 09/28 features: "Added vectorized environments ... parallel policy evaluation ... Set `--num-envs` to match the number of instance indices evaluated together. Existing single-environment workflows remain supported with `--num-envs=1`." and "batched policy transport and optional action-chunk replay" (`origin/main:docs/challenge/updates.md:77-81`). [DOC]
- 2026-eval: "For vectorized evaluation with one policy server per rollout, pass `--policy-endpoints` with one `HOST:PORT` for each `--instance-indices` value, in the same order." (`origin/2026-eval:docs/challenge/evaluation.md:52,66-69`) [DOC]
- **Tiled rendering:** "**Tiled rendering is not recommended.** Although it can speed up evaluation, it introduces visual artifacts that may be out of distribution for your policy. We will **not** use tiled rendering for final evaluations." (`origin/main:docs/challenge/updates.md:53`) [DOC] In code it is opt-in via the env var `OMNIGIBSON_ENABLE_TILED_RENDERING`, default False, and applies only when `num_envs > 1` (`OmniGibson/omnigibson/macros.py:161-166`; `envs/env_base.py:481`). [CODE]

## 14. History-dependent policies

- "For each task ID and instance ID, we will use the same policy-server IP address and port throughout the rollout to support history-dependent policies." (`origin/main:docs/challenge/updates.md:57`) [DOC]
- The reset message `{"reset": True}` is sent at the start of each episode (`network_utils.py:172-179`). "The current reset message has no acknowledgment." (`origin/2026-eval:docs/challenge/evaluation.md:111`) [DOC/CODE]
- Memory and SLAM are explicitly envisioned: "memory systems, SLAM, and LLM-assisted policies" (`index.md:142`). [DOC]

## 15. Submission and leaderboard logistics

- "**No formal registration is required**" (`submission.md:9`). The 08/24 update nevertheless introduced a participant registration form (https://forms.gle/Kf4ABLmDKbuK5Yhj6) (`updates.md:33`). [DOC]
- Final zip (`submission.md:102-123`):
  - metrics JSON files (up to 1,000)
  - the wrapper `.py`
  - the exact robot `.yaml`/`.json`
  - a README "including the full evaluator command, wrapper path, robot config path, Docker image details or IP address information. For IP address-based submissions, include at least 50 available ports."
  - Video links go through the portal (`submission.md:125`). [DOC]
- "Submitted solutions will remain confidential unless participants explicitly grant permission" (`submission.md:96-98`). [DOC]
- Links must have viewing permissions (`origin/main:docs/challenge/updates.md:55`). [DOC]
- Prizes total $11,000: 1st $5,000, 2nd $3,000, 3rd $2,000, plus $1,000 for an outstanding open-source solution (`index.md:68-89`; `evaluation.md:247-268`). [DOC]
- [INF] The in-repo leaderboard script `docs/gen_leaderboard.py` belongs to 2025. It ranks by `max(hidden_q, public_q)` (`docs/gen_leaderboard.py:67-82`), which conflicts with the 2026 rule that hidden replaces public. The 2026 leaderboard is the HF Space, whose code is not available here (NOT FOUND).

## 16. URLs referenced in the six docs (union across refs)

Checkpoints:
- π0.5 `turning_on_radio`: https://drive.google.com/file/d/1KojwNUz0HVwU3Ww2SVh3NKt-4asuI3y2/view?usp=sharing (`baselines.md:18`)
- GR00T N1.7 `turning_on_radio`: https://drive.google.com/file/d/1OXNm3SPLvWOSJR1e8In6xHMHxOYDp789/view?usp=sharing (`baselines.md:18`)
- Only one task has provided checkpoints (`baselines.md:16-18`).
- Base model: `gs://openpi-assets/checkpoints/pi05_base/params` (`baselines.md:110`)
- https://huggingface.co/nvidia/GR00T-N1.7-3B (`baselines.md:6,189`)
- https://huggingface.co/nvidia/Cosmos-Reason2-2B (gated, `baselines.md:206`)

Datasets:
- https://huggingface.co/datasets/behavior-1k/2026-challenge-demos (LeRobot v3, 20,000 demos, 3.27 TB; `dataset.md:9`, `baselines.md:55`)
- https://huggingface.co/datasets/behavior-1k/2026-challenge-rawdata (raw HDF5, 1.44 TB; `dataset.md:7`)
- https://huggingface.co/datasets/behavior-1k/2026-challenge-demos/discussions/2 (depth fix; `updates.md:46`)
- In code only: HF `behavior-1k/zipped-datasets`, which includes `2026-challenge-task-instances` (`asset_utils.py:435-437,590-592`)

Leaderboard and submission:
- https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard (`index.md:57`; `mkdocs.yml:108`)
- https://behavior-1k-2026-challenge-leaderboard.hf.space/submit (`index.md:44`; `submission.md:4,125`; `origin/main:updates.md:49`)

Forms:
- Registration: https://forms.gle/Kf4ABLmDKbuK5Yhj6 (`updates.md:33`)
- Inference Speed Form: https://forms.gle/5GNqc2eNhFTx4UhN8 (`origin/main:updates.md:41`)

Code:
- https://github.com/StanfordVL/BEHAVIOR-1K.git (`baselines.md:46`)
- https://github.com/wensi-ai/openpi (`baselines.md:5,85`)
- https://github.com/wensi-ai/Isaac-GR00T (`baselines.md:6,198`)
- https://github.com/Physical-Intelligence/openpi (`evaluation.md:50`)
- https://github.com/huggingface/lerobot (`dataset.md:9`; `baselines.md:55`)
- lerobot@c75455a6de5c818fa1bb69fb2d92423e86c70475 (`baselines.md:229`)

Docs and other:
- https://behavior.stanford.edu/omnigibson/controllers.html (`evaluation.md:220`)
- https://www.physicalintelligence.company/blog/pi05 (`baselines.md:5`)
- https://docs.astral.sh/uv/ (`baselines.md:25`)
- Discord https://discord.gg/bccR5vGFEx (`index.md:153`)
- Zoom office hours, Mondays 5-6pm PT (`index.md:154`)
- YouTube and Vimeo videos (`index.md:5,134-148`; `evaluation.md:300`)
- Sponsors (`index.md:180-189`)

Web search: the public site pages exist at https://behavior.stanford.edu/challenge/evaluation.html and https://behavior.stanford.edu/challenge (seen in search results; content not fetchable here). Search found no snippet with additional 2026 rules.

## 17. Discrepancies

Page vs code, and page vs page:

1. **Hidden-instance indexing.** The docs say "indices `20-39` are hidden instances" (`evaluation.md:66`). In code, hidden instances use `--mode hidden_test --instance-indices 0..19`. Passing 20-39 fails the assertion `indices ⊆ range(20)` (`evaluator.py:83-86`; `eval.py:50-65`). `--mode` is not documented in `evaluation.md` at all.
2. **Number of hidden instances.** The docs mention both "20-39 hidden" (20 instances, `evaluation.md:66`) and "hold out 10 more instances for final evaluation" (`evaluation.md:290`). Code has 20 hidden IDs, 321-340 (`eval_utils.py:18-21`).
3. **Scorer divisor.** `score_utils.compute_final_q_score` divides each task's sum by 20 for the "public" testset (`score_utils.py:300,340`), but the docs say to report 10 instances (0-9) (`evaluation.md:66,282`). With the in-repo script, a full 10-instance submission would score at most 0.5. [INF] The organizers' leaderboard probably uses different code (NOT FOUND).
4. **Scorer naming and tracks.** The scorer still expects a folder named `<track>.<testset>.<team>.<affiliation>.<date>` with track ∈ {standard, privileged} (`score_utils.py:236-262`), but 2026 has a single track (`evaluation.md:5`). Expected file names use instance IDs (`{task}_{301..320}_0.json`, `score_utils.py:288-299`). The sample JSON in `submission.md:49` shows `"instance_id": 0`, but the code writes the real ID (`evaluator.py:676`).
5. **Q-score definition.** The docs say "(Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates)" (`evaluation.md:231`). The code counts only predicates **newly** satisfied (false at the start and true at the end), takes the max over goal options, and gives 1.0 on full success (`task_metric.py:6-28`).
6. **Timeout enforcement.** The 600 s per-step limit, the 3 reconnects, the `max_steps × 1 s` budget and the first-query allowance are announced in updates (main) and evaluation.md (2026-eval). They are **absent from the HEAD/main code** and implemented only on `origin/2026-eval` (§8.4).
7. **Contradiction inside 2026-eval.** updates.md says the sim-step cost is split "among rollouts active for that step ... The simulator share changes as rollouts finish or fail" (`origin/2026-eval:docs/challenge/updates.md:15`). evaluation.md says "divided by the total number of environments in the batch. This divisor stays fixed" (`origin/2026-eval:docs/challenge/evaluation.md:115`). The code matches evaluation.md: `share = seconds / len(self.clients)` (`origin/2026-eval:.../policies.py:218-220`).
8. **Single-endpoint group budget.** In 2026-eval code, single-endpoint mode (`--host/--port`) with `num_envs = N` gets one group deadline of `max_steps` seconds of wall-clock (`origin/2026-eval:.../evaluator.py:884`), not N × max_steps. group-time-budget docs said N × max_steps (`origin/group-time-budget:docs/challenge/updates.md:15`), and 2026-eval docs describe per-rollout accounting only for multi-port.
9. **Version tags.** HEAD evaluation.md and baselines.md say `v3.9.3`; HEAD updates.md says `v3.9.2`; main updates say `v3.9.3-post1`, or `-post2` for custom robots. No `v3.9.3` tag exists on the remote (`git ls-remote`).
10. **Registration.** `submission.md:9` says none is required; `updates.md:33` introduced a registration form.
11. **`obs_modalities`.** The robot config field is said to be customizable (`evaluation.md:176`), but the evaluator overwrites it from the wrapper's `camera_spec` (`evaluator.py:269`).
12. **Example commands vs official track.** The documented commands use the default `DefaultWrapper` (RGB 224, no depth) rather than `RGBDFullResWrapper` (`evaluation.md:39-48,82`; `baselines.md:306-312`; `eval.py:88-92`).
13. **Reconnect at HEAD.** The HEAD docstring says "If True, the client will attempt to reconnect indefinitely", but the loop bounds it at `max_retries=2` (`network_utils.py:57,127-170`). `eval.py` never enables reconnect at HEAD.
14. **Demo gallery data.** The gallery has instructions only for tasks 0-49 (`task_data.json`). `updates.md:50` says `meta/tasks.jsonl` covers all 100.

## 18. Open questions

- Which code will the organizers actually run for final hidden-test evaluation: the `v3.9.3-post1` tag (no time budget in code) or the `2026/eval` branch (budget enforced)? Will they use `--policy-endpoints` (multi-port, ≥50 ports) or one endpoint with `num_envs>1`?
- Do hidden runs use `RGBDFullResWrapper` or the participant's submitted wrapper? The docs say the participant's wrapper is inspected and used (`submission.md:37-42`).
- The exact per-task `max_steps` values. They need `2026-challenge-task-instances/metadata/task.jsonl`, which is NOT FOUND locally.
- Is the 2026 leaderboard scorer divisor 10 or 20 per task? Is the Q-score "newly satisfied" or "satisfied at end"?
- Is using the NL task instruction at test time officially allowed? It is implied by the baselines and by `task_id` in the observation, but not stated.
- Is the deadline final? It passed on 2026-10-16. Is any late or hidden-test submission window still open?
- Is data collection on public test instances (indices 10-19) allowed in 2026? The 2025 ban is not restated.

## 19. Implications for a planner + executor agent

- Only RGB, depth (`depth_linear`) and proprio may reach the policy, plus `task_id` and `cam_rel_poses`, which the evaluator adds. The planner must not read sim state, object poses, segmentation or global robot pose. Any wrapper we write is manually inspected (`evaluation.md:144`).
- The BDDL goal can be used at test time (`evaluation.md:28`). A strong-model planner may parse the task's BDDL goal (mapped via `task_id`) to produce subgoals. Planner LLM API calls are allowed, but we must supply credentials and quota, and their latency counts toward the `max_steps × 1 s` budget and the 600 s per-query limit.
- Budget math [INF]: the server is queried once per sim step unless action-chunk replay is used.
  - With 1 s per step on average, slow LLM calls must be amortized. Call the planner rarely, for example at subtask boundaries. Return π0.5 chunks with `--replay-action-chunk-size` (≤ the replanning horizon, e.g. 16), so each query covers many steps.
  - The first query is free, up to 600 s, on 2026-eval. Use it for warm-up and initial planning.
  - Any planning overrun yields Q = 0 for that rollout.
- History-dependent state is supported: the same IP and port are used throughout a rollout (`origin/main:updates.md:57`). Keep per-rollout memory server-side, and reset it on the `{"reset": True}` message. With num_envs>1 on one endpoint, observations arrive batched with a leading env dimension. With `--policy-endpoints`, run one server per port.
- The Docker submission must fit on a single 24 GB GPU (RTX 3090, A5000 or TitanRTX class). A local strong-model planner plus π0.5 will not fit on that GPU [INF]. An external API planner or an IP-based submission with ≥50 ports are the practical options.
- Report exactly one rollout per instance, on public indices 0-9 (`--mode public_test`), for 100 tasks. Missing tasks count as 0, so cover all 100 tasks, even weakly. Do not re-run and cherry-pick. Do not edit JSONs or videos. Include the wrapper, robot config and README.
- Use `v3.9.3-post2` (HEAD) for local work. Use `--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper` if depth or full resolution is needed. Do not enable tiled rendering, because the final evaluation won't use it.

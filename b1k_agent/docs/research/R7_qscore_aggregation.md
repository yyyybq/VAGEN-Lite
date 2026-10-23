# R7 — Q-score and aggregation, exactly as implemented (BEHAVIOR-1K 2026 Challenge)

Scope: per-episode Q-score, success, secondary metrics, the per-episode metrics JSON, aggregation into per-task and overall scores, missing-instance handling, the leaderboard, and reusable aggregation pseudo-code.

Path conventions:
- `B1K` = `/home/user/stanfordvl/behavior-1k` at HEAD = `v3.9.3-post2` (a8247a8).
- `OG` = `B1K/OmniGibson/omnigibson`.
- `2026-eval:` = branch `origin/2026-eval` (38d2c9f). `main:` = `origin/main` (7d6718c). `gtb:` = `origin/group-time-budget` (ad934aa).
- `v3.7.2:` = 2025-era tag, fetched from raw.githubusercontent.com into `scratchpad/research/refs/v372/`.

Status tags:
- **[CODE]** verified in code.
- **[DOCS]** stated in docs.
- **[DATA]** verified against shipped result JSONs.
- **[INFERRED]** my inference.
- **[NOT FOUND]** searched for and not located.

---

## 0. TL;DR

1. **Per-episode Q is computed once, from the state at the step where the episode ends** (success or timeout). It is not a max over time. **[CODE]** `OG/eval/evaluator.py:125-137` and `:669-679`; `OG/metrics/task_metric.py:58-81`.
2. Per-episode formula **[CODE]** (`OG/metrics/task_metric.py:6-28`):
   ```
   Q = 1.0                                                       if success
     = max over grounded goal options o of
         |{p in o : p false at episode start AND p true at end}| / |o|     otherwise
   ```
   - Predicates already true at episode start earn **no credit**, but they **still count in the denominator**.
   - Breaking a predicate that started true carries **no penalty**, except that it may prevent full success.
   - The options come from disjunctions, quantifiers, `forn` and `forpairs`. They are expanded into grounded conjunctions of atomic, possibly negated, literals. The best single option is taken (`bddl3/bddl/condition_evaluation.py:741-791`).
3. **Success** means the `PredicateGoal` termination condition fired. That happens when every top-level BDDL goal condition evaluates true at that step, and the episode then terminates immediately (`OG/tasks/behavior_task.py:237-250`, `OG/termination_conditions/predicate_goal.py:38-44`, `OG/tasks/task_base.py:266-282`).
4. **Per-task aggregation** is the sum of per-rollout `q_score.final` divided by a **fixed** `n_instances_per_task`. Missing files contribute 0 (`OG/eval/utils/score_utils.py:339-345`).
5. **Overall score** is the mean over **all** tasks in `TASK_NAMES_TO_INDICES`, so a missing task counts as 0 (`score_utils.py:347-353`). `task_sr` counts rollouts with `q == 1` (`score_utils.py:341`).
6. **Discrepancy in the denominator.** HEAD `score_utils.py:300` uses `NUM_PUBLIC_TEST_INSTANCES = 20` (`eval_utils.py:19`). The docs say only the **first 10 public instances (indices 0-9)** are reported (`docs/challenge/evaluation.md:66,282`) and the full evaluation is 100 tasks x 10 instances (`docs/challenge/submission.md:10`). The 2025 code hard-coded `/10` (`v3.7.2:score_utils.py:114`). Run as-is on a 10-instance submission, HEAD `score_utils` would **halve** the score. **[CODE vs DOCS]**
7. **Secondary metrics** are computed for **every** rollout, successful or not:
   - `time_score = M/(M-1) - 1/((M-1)*normalized_time)` with `M = 1.5`, which equals `3 - 2*steps/human_mean_len` (`score_utils.py:321-324`, `eval_utils.py:17`).
   - Distance scores are `human_mean / agent` for base and for the left and right end effectors (`agent_metric.py:52-66`).
   - They are used only as tie-breakers (`docs/challenge/evaluation.md:243`).
8. **2026-eval branch.** A policy connection loss, query timeout, policy error or exceeded group time budget writes a JSON with `failure_reason`, `success=false`, `q_score.final = 0.0`, all normalized distances 0 and `normalized_time = 1/1.5`, which gives time_score 0 (`2026-eval:OG/eval/evaluator.py:827-852`). The docs agree: "A failed or over-budget rollout receives zero score" (`2026-eval:docs/challenge/evaluation.md:115`).
9. **Partial-credit Q cannot be recomputed from the metrics JSON alone.** The JSON holds only the scalar `q_score.final`, not per-option or per-predicate masks. The scalar is enough for aggregation, so reuse it verbatim.
10. **2026 leaderboard.** It is a Hugging Face Space (`docs/challenge/index.md:57`). Its scoring code is **NOT FOUND**: huggingface.co is blocked, GitHub code search is blocked for this session, and WebSearch returned nothing usable. The in-repo leaderboard generator `docs/gen_leaderboard.py` builds only the **2025 archive** page (`gen_leaderboard.py:93`).

---

## 1. Where the per-episode score is produced (call chain)

```
python -m omnigibson.eval.eval  (OG/eval/eval.py)
  -> resolve_instance_ids(...)                                     evaluator.py:73-87, eval.py:130
  -> BatchedEvaluator.load_env: human_stats = load_human_stats(task) evaluator.py:249
       max_steps = int(human_stats["length"] * 1.5)                 evaluator.py:281-286
  -> load_metrics -> [AgentMetric(human_stats), TaskMetric(human_stats)]  evaluator.py:374-378
  -> run() -> evaluate_instances_batched(load_fn, step_fn, record_fn)    evaluator.py:636-697
       load_batch: env.reset(); metric.reset()  (TaskMetric snapshots initial option masks)  evaluator.py:551, 564-568
       loop: step_fn -> env.step -> metric.step for ACTIVE envs        evaluator.py:386-408
             if terminated or truncated: record_fn(...) immediately   evaluator.py:125-137
       record_fn: result = {task, instance_id, rollout_id, steps, success} + metric.aggregate()  evaluator.py:669-679
                  json.dump(result) -> <output-dir>/json/<task>_<instance_id>_<rollout_id>.json  evaluator.py:680-682, eval.py:171-172
```

Key code. HEAD `OG/eval/evaluator.py:669-682`:

```python
def record_fn(env_idx, instance, step, terminated, truncated):
    instance_eval_state = self.instance_eval_states[env_idx]
    instance_eval_state.active = False
    self.n_trials += 1
    success = instance_eval_state.env_accessor.success
    ...
    result = {"task": task_name, "instance_id": int(instance), "rollout_id": rollout_id, "steps": step}
    result["success"] = success
    for metric in instance_eval_state.metrics:
        result.update(metric.aggregate())
    if metrics_dir is not None:
        with open(os.path.join(metrics_dir, f"{task_name}_{instance}_{rollout_id}.json"), "w") as f:
            json.dump(result, f, indent=2, default=float)
```

**When the episode ends [CODE].**
- `evaluate_instances_batched` checks `terminated`/`truncated` after every step and calls `record_fn` **in that same iteration** (`evaluator.py:120-137`).
- `terminated` is set when a non-timeout termination condition fires, i.e. `PredicateGoal`, which means success. `truncated` is set by the `timeout` condition (`OG/envs/env_base.py:781-792`).
- `Timeout` fires when `env.episode_steps >= max_steps` (`OG/termination_conditions/timeout.py:23`).
- At HEAD, `run()` does not pass `max_steps` to `evaluate_instances_batched` (`evaluator.py:691-697`). Timeout is therefore enforced only by the task's `Timeout` condition, whose `max_steps` is set in `evaluator.py:281-289`.

Finished envs keep being stepped by the shared simulator with zero actions. They are recorded at finish time, so later steps cannot change their score (`evaluator.py:423-437`; the docstring on the 2026-eval branch says this explicitly at `2026-eval:evaluator.py:116-119`). **[CODE]**

**Q is therefore evaluated at the terminal step only, with no max over time [CODE].** The same holds in 2025: `TaskMetric.end_callback` (`v3.7.2:OG/metrics/task_metric.py:30-45`).

---

## 2. Precise per-episode Q-score definition

### 2.1 The function (verbatim, `OG/metrics/task_metric.py:6-28`)

```python
def compute_q_score(
    success: bool,
    now_satisfied_options: Sequence[Sequence[bool]],
    initial_satisfied_options: Sequence[Sequence[bool]],
) -> float:
    """
    Partial-success (Q-score) for one episode/env: a full success scores 1.0; otherwise the fraction
    of goal predicates that were NOT satisfied at episode start but ARE satisfied now, maximized over
    the alternative goal-state options. ...
    """
    if success:
        return 1.0
    if not now_satisfied_options:
        return 0.0
    option_scores = []
    for now_opt, init_opt in zip(now_satisfied_options, initial_satisfied_options):
        if len(now_opt) == 0:
            option_scores.append(0.0)
            continue
        newly_satisfied = sum(int((not init) and now) for now, init in zip(now_opt, init_opt))
        option_scores.append(newly_satisfied / len(now_opt))
    return max(option_scores) if option_scores else 0.0
```

Call site, `OG/metrics/task_metric.py:43-81`:
- `reset()` stores `self.initial_predicate_states = env.get_goal_option_satisfaction()` (`:48-52`). It runs from `load_batch` **after** `env.reset()`, i.e. after physics settling and instance loading (`evaluator.py:550-568`).
- `_compute_episode_metrics()` calls `compute_q_score(success=env.success, now_satisfied_options=env.get_goal_option_satisfaction(), initial_satisfied_options=self.initial_predicate_states)` (`:64-73`). It returns
  ```
  {"q_score": {"final": q},
   "time": {"simulator_steps": T, "simulator_time": T*rendering_dt, "normalized_time": human_len/T}}
  ```
  (`:74-81`).

### 2.2 How the option masks are built (`OG/tasks/behavior_task.py:206-235`)

```python
def get_goal_option_satisfaction(self, env_idx):
    from bddl.condition_evaluation import evaluate_state
    def evaluate_fn(predicate_name, *entities):
        return self._evaluate_predicate(env_idx, predicate_name, *entities)
    option_masks = []
    for option in self.ground_goal_state_options:
        _, results = evaluate_state(option, evaluate_fn)
        satisfied = set(results["satisfied"])
        option_masks.append([i in satisfied for i in range(len(option))])
    return option_masks
```

- `self.ground_goal_state_options = compiled.ground_goal_state_options` (`behavior_task.py:415`).
- That list is built by `CompiledTask.__init__` via `get_ground_goal_state_options(conditions, object_scope, goal_conditions)` (`bddl3/bddl/knowledge_base/models.py:956-962`), which calls `get_ground_state_options` (`bddl3/bddl/activity.py:155-184`).

### 2.3 "Ground goal options": how disjunctive and quantified goals become options (`bddl3/bddl/condition_evaluation.py`)

Each compiled node has `flattened_condition_options`, a list of options, where each option is a list of atomic literals. The rules:

| Construct | Ground options | Evaluate semantics | Lines |
|---|---|---|---|
| `and` | Cartesian product of children's options, concatenated | all | 110-125 |
| `or` | union of children's options | any | 156-166 |
| `forall ?x - cat` | product over instances | all | 205-220 |
| `exists ?x - cat` | union over instances | any | 254-264 |
| `forn (N)` | for each product option, every **combination of exactly N** children ("minimal solution") | `sum == N` (exactly N) | 300-319 |
| `forpairs` | all L-permutation matchings x product of options | bipartite coverage >= min(M,N) | 365-406 |
| `fornpairs (N)` | all N-pair matchings x product | coverage >= N | 447-477 |
| `not` | De Morgan: one negated literal picked from each child option (product) | not child | 508-530 |
| `imply A B` | (negated-antecedent options) + (consequent options) | `not A or B` | 570-597 |
| `HEAD` (top-level condition) | child's options | child value | 637-667 |

Top level, `get_ground_state_options` (`condition_evaluation.py:741-791`):
- Takes the **Cartesian product across all top-level goal conditions** and flattens each combination into one list of literals (`:763-771`).
- Drops self-contradictory options that contain both `P` and `(not P)` (`:773-785`).
- Re-compiles each surviving option as a list of `HEAD`s, one per literal, **sorted by length, shortest first** (`:787-790`).
- `get_ground_goal_state_options` asserts there is at least one option (`activity.py:183`).

Consequences **[CODE + INFERRED]**:
- One "option" is a full grounded conjunction, i.e. one concrete way to satisfy the **entire** goal. Its literals include negated literals such as `["not", ["open", "fridge_1"]]`. Each literal is one unit of partial credit with equal weight.
- Partial credit is the best single option, so progress spread across incompatible groundings does not add up. Example: placing object A in bin 1 and object B in bin 2 when the options require both in the same bin.
- Shorter options give more credit per literal, e.g. 1 of 2 = 0.5 beats 2 of 5 = 0.4.
- Negated goal literals that are already true at start (e.g. "not cooked") earn no credit but still count in `|o|`. **The maximum partial (non-success) Q is below 1 whenever an option contains initially-true literals.**
- The bddl3 docstring says the same: "you can evaluate each ground option independently and report which fraction of its literals are already satisfied" (`activity.py:163-165`). The older `get_reward` used the **raw** fraction satisfied without the "newly" filter (`activity.py:261-275`). The challenge Q uses the **newly-satisfied** fraction (`task_metric.py:26`).

Toy checks, running `compute_q_score` extracted verbatim from `task_metric.py`:

| Case | now | init | Q |
|---|---|---|---|
| success | - | - | 1.0 |
| 2 of 4 newly true | [T,T,F,F] | [F,F,F,F] | 0.5 |
| 1 literal true at start | [T,T,F,F] | [T,F,F,F] | 0.25 |
| initially-true literal broken | [F,T,F,F] | [T,F,F,F] | 0.25 (no penalty) |
| max over options | [[T,F],[T,T,F,F,F]] | all F | 0.5 |
| option fully true at start, not success | [[T,T]] | [[T,T]] | 0.0 |
| no options | [] | [] | 0.0 |

### 2.4 Success definition [CODE]

- `BehaviorTask._create_termination_conditions` (`behavior_task.py:237-250`) creates `Timeout(max_steps)` and `PredicateGoal(check_goal_fn=lambda env_idx: self.compiled_task.check_goal(...))`.
- `CompiledTask.check_goal` runs `evaluate_state(self.goal_conditions, evaluate_fn)` on the **original, non-grounded** top-level goal conditions. `all_satisfied` is true iff no condition is unsatisfied (`models.py:964-975`; `condition_evaluation.py:720-738`).
- `PredicateGoal._step` sets done=all_satisfied per env (`predicate_goal.py:38-44`). As a `SuccessCondition`, its `success = done & _terminate_is_success` (`termination_condition_base.py:93-100`).
- `TaskBase._step_termination` ORs dones and successes across conditions (`task_base.py:266-282`). `task.success` is updated every step (`task_base.py:383-389`).
- The evaluator reads `bool(shared_env.task.success[env_idx])` (`evaluator.py:163-165`).

In short, **success means all BDDL goal conditions are true at some step. The episode terminates on that step and Q = 1.0.** No "hold for k steps" requirement was found.

**Docs vs code:** the docs describe Q as "(Number of goal BDDL predicates satisfied at episode end) / (Total number of goal predicates)" (`docs/challenge/evaluation.md:231`). The code adds two things: (a) only predicates **newly** satisfied relative to the episode start count, and (b) the max is taken over grounded goal options. The 2025 archive leaderboard text matches the code better: "computing the fraction of completed sub-goals and choosing the best-matched goal clause" (`docs/gen_leaderboard.py:100-107`; also https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html via WebSearch snippet).

### 2.5 Time and efficiency (secondary) terms

**TaskMetric time [CODE]** (`task_metric.py:54-81`):
- `timesteps` is incremented once per env step while the env is active.
- `simulator_time = timesteps * og.sim.get_rendering_dt()`.
- `normalized_time = human_stats["length"] / timesteps`, where `length` is the **mean** human episode length in frames (`score_utils.py:215-233`, mean at `:231`). It is `inf` if `timesteps == 0`.
- The docs' sample shows 500 steps giving 16.667 s, i.e. 1/30 s per step (`docs/challenge/submission.md:66-70`). **[DOCS]**

**AgentMetric distance [CODE]** (`agent_metric.py:1-66`):
- Per step, it records the Euclidean displacement of the base position and of each arm's EEF position (keys `"base"` plus `robot.arm_names`).
- Episode values: `agent_distance[k] = sum(deltas)` and `normalized_agent_distance[k] = human_stats[k] / agent_distance[k]` (`inf` if the agent never moved), with `human_stats = {base: distance_traveled, left: left_eef_displacement, right: right_eef_displacement}` (`agent_metric.py:13-19, 52-66`).
- The human keys are only "left" and "right". A robot whose `arm_names` differ from `["left","right"]` would hit a KeyError there. **[INFERRED]** R1Pro uses left/right, as the 2025 JSON keys and the docs sample show (`submission.md:53-62`).

**Aggregation-time secondary scores [CODE]** (`score_utils.py:320-327`):

```python
q_score[task_name][f"{instance_id}_{rollout_id}"] = result["q_score"]["final"]
normalized_time = result["time"]["normalized_time"]
time_score[task_name][...] = EVAL_TIMEOUT_MULTIPLIER / (EVAL_TIMEOUT_MULTIPLIER - 1.0) - 1 / ((EVAL_TIMEOUT_MULTIPLIER - 1.0) * normalized_time)
base_distance_score[...]  = result["normalized_agent_distance"]["base"]
left_distance_score[...]  = result["normalized_agent_distance"]["left"]
right_distance_score[...] = result["normalized_agent_distance"]["right"]
```

- With `EVAL_TIMEOUT_MULTIPLIER = 1.5` (`eval_utils.py:17`): `time_score = 3 - 2 * steps / human_len`. That gives 1.0 at human speed, 0 at the 1.5x timeout, and at most 3.
- Time and distance scores are computed for **all** rollouts, including failures. A timed-out rollout gets time_score of about 0, not a penalty.
- 2025 used `2 - 1/normalized_time` with a 2x timeout (`v3.7.2:score_utils.py:98`; `docs/challenge/archive/2025/evaluation.md:55`).
- Observed 2025 failure rollouts have time_score = -0.000335, slightly negative **[DATA]** (`docs/challenge_submissions/standard.public.Robot_Learning_Collective...json`, `per_rollout_scores.time_score.turning_on_radio["214_0"]`).
- Distance scores are **not capped**. MR's overall base_distance_score is 57.22 (`privileged.public.MR.MR.20251115.json`, `overall_scores`). An agent that never moves produces `inf`, which Python's `json.dump` writes as `Infinity` and which would make the aggregate `inf`. **[CODE + INFERRED]**
- Role: tie-breakers only. "The success score (Q) is the metric used for ranking submissions. If two submissions achieve the same score, secondary metrics will be used to break ties." (`docs/challenge/evaluation.md:243`). "Secondary metrics will be normalized using human averages from 200 demonstrations per task." (`:241`). **[DOCS]**

No time penalty enters Q itself. **[CODE]**

### 2.6 Failures, timeouts and wall-clock budget (2026-eval branch)

At HEAD (v3.9.3-post2) there is no failure path. A crash or exception means no JSON is written for the affected rollouts, so the result is missing and counts as 0 (see section 4).

On `origin/2026-eval` (`2026-eval:OG/eval/evaluator.py`):
- `evaluate_instances_batched` gains `fail_unfinished(...)`, which records every active env as `truncated=True` with a failure key (`:141-145`). It also gains a group wall-clock deadline (`:157`, `:167-195`) that produces `"group_time_budget_exceeded"`, and per-env `failures` from `MultiWebsocketPolicy` (`:199-217`).
- `record_fn` (`:814-876`) does the following on failure:
  ```python
  success = False if failure else instance_eval_state.env_accessor.success          # :828
  if not failure or step > 0:  result.update(metric.aggregate())                      # :833-835
  if failure:
      result["failure_reason"] = "policy_connection_lost" | "group_time_budget_exceeded" | "policy_error" | "policy_query_timeout"   # :837-844
      result["q_score"] = {"final": 0.0}                                              # :845
      result.setdefault("agent_distance", {"base": 0.0, "left": 0.0, "right": 0.0})  # :846
      result["normalized_agent_distance"] = {"base": 0.0, "left": 0.0, "right": 0.0} # :847
      result.setdefault("time", {"simulator_steps": step, "simulator_time": step * dt})  # :848-851
      result["time"]["normalized_time"] = 1 / EVAL_TIMEOUT_MULTIPLIER                 # :852  -> time_score = 0
  if isinstance(self.policy, MultiWebsocketPolicy):
      result["time"]["policy_time"|"sim_time"|"evaluation_time"|"evaluation_fps"] = ...   # :859-866
  ```
- **Any partial progress is discarded on failure: Q = 0** even when the metrics were aggregated. **[CODE]**
- Docs on that branch:
  - Each rollout has an accounted budget of `max_steps` seconds.
  - There is a 600 s first-query allowance and a 600 s per-query limit.
  - "A failed or over-budget rollout receives zero score; other active rollouts continue." (`2026-eval:docs/challenge/evaluation.md:111-115`). **[DOCS]**
- `main:docs/challenge/updates.md:15-19` states the 1 FPS average and the `max_steps x 1 second` episode timeout. It also says reconnection is attempted up to 3 times, after which the rollout counts as a failure. `gtb:docs/challenge/updates.md` (diff) reframes this as a group budget of `N x max_steps` seconds. **[DOCS]**

---

## 3. Per-episode metrics JSON schema

**HEAD v3.9.3-post2 [CODE]** (`evaluator.py:676-682` plus the metrics):

| Field | Type | Source |
|---|---|---|
| `task` | str | `evaluator.py:676` |
| `instance_id` | int, the **resolved** test instance id (public: 301..320; hidden: 321..340) | `evaluator.py:676`, `eval_utils.py:21`, `evaluator.py:78-87` |
| `rollout_id` | int | `evaluator.py:676` |
| `steps` | int, batch loop step at finish | `evaluator.py:676`, `:123` |
| `success` | bool | `evaluator.py:673-677` |
| `agent_distance` | {base,left,right}: float | `agent_metric.py:56` |
| `normalized_agent_distance` | {base,left,right}: float (human/agent; may be `Infinity`) | `agent_metric.py:58-64` |
| `q_score` | {final: float in [0,1]} | `task_metric.py:75` |
| `time` | {simulator_steps, simulator_time, normalized_time} | `task_metric.py:76-80` |

**2026-eval additions [CODE]:**
- `failure_reason` (on failure).
- `time.policy_time`, `time.sim_time`, `time.evaluation_time`, `time.evaluation_fps` (only with `--policy-endpoints`, i.e. MultiWebsocketPolicy).

These come from `2026-eval:evaluator.py:837-866`.

**Filename [CODE]:** `<output-dir>/json/<task>_<instance_id>_<rollout_id>.json` (`eval.py:171-172`, `evaluator.py:681`), for example `turning_on_radio_301_0.json`.

**Docs sample [DOCS]** (`docs/challenge/submission.md:44-72`) shows the same keys but `"instance_id": 0`. The code writes the resolved id (e.g. 301). See discrepancies.

The 2025 sample JSON had only `agent_distance`, `normalized_agent_distance`, `q_score`, `time` (`docs/challenge/archive/2025/submission.md:12-33`).

**Can Q be recomputed from the JSON? No [CODE].** The JSON carries only the scalar `q_score.final`. The per-option masks (`initial_predicate_states` and the end-of-episode masks) are not serialized (`task_metric.py:74-81`). The step `info` contains `goal_status` (indices of satisfied and unsatisfied **top-level** goal conditions; `behavior_task.py:757-759`), but it is not written either. For aggregation, `q_score.final` is exactly what the official aggregator consumes (`score_utils.py:320`), so nothing needs recomputing. For diagnostics, such as which literals were missed, a dev-only run must call `env.task.get_goal_option_satisfaction(env_idx)` itself. **[INFERRED]**

---

## 4. Aggregation: episode to task to overall (official `score_utils.compute_final_q_score`)

`OG/eval/utils/score_utils.py:271-397` (HEAD; identical on main, 2026-eval and gtb, since `git diff` shows no changes):

1. **Input layout** (`:236-268`, `:283-287`): `<track>.<testset>.<team>.<affiliation>.<date>/json/*.json`, where `track` is `standard` or `privileged` and `testset` is `public` or `hidden`. `sanity_check_filename` appends `_0` if the rollout id is missing (`:261-268`).
2. **Allowed filenames** (`:289-299`): `{task}_{iid}_{0}.json` for every task in `TASK_NAMES_TO_INDICES` and every `iid` in `TEST_INSTANCE_IDS[:20]` (public, i.e. 301..320) or `TEST_INSTANCE_IDS[20:]` (hidden, i.e. 321..340). Only **rollout 0** is allowed (`range(1)`). Any other `.json` file triggers `assert file in possible_filenames` (`:313`).
3. **Per-rollout values** (`:316-327`): q, time_score and the three distance scores as above.
4. **Per task** (`:339-345`):
   ```python
   q_score_avg[t] = sum(q_score[t].values()) / n_instances_per_task
   task_sr[t]     = sum(1 for v in q_score[t].values() if v == 1) / n_instances_per_task
   (time/base/left/right analogous, same fixed denominator)
   ```
   with `n_instances_per_task = NUM_PUBLIC_TEST_INSTANCES if testset == "public" else NUM_HIDDEN_TEST_INSTANCES`. Both are 20 at HEAD (`:300`, `eval_utils.py:18-20`).
5. **Overall** (`:347-353`): `overall = sum(per_task.values()) / len(TASK_NAMES_TO_INDICES)`, an unweighted mean over all challenge tasks (100 per `docs/challenge/task_data.json`, which has 100 entries **[DATA]**; `TASK_NAMES_TO_INDICES` is loaded from `DATA_PATH/2026-challenge-task-instances/metadata/B100_task_misc.csv`, `eval_utils.py:60-77`).
6. **Output** (`:355-387`): `{team, affiliation, date, track, testset, overall_scores{q_score, task_sr, time_score, base_distance_score, left_distance_score, right_distance_score}}`. With `final_score_only=False` it adds `per_task_scores{...6 metrics}` and `per_rollout_scores{q_score, time_score, base/left/right_distance_score}` keyed `"<iid>_<rid>"`. It is written to `<output_dir>/<track>.<testset>.<team>.<affiliation>.<date>.json`.

**Missing instances and tasks [CODE + DOCS + DATA]:**
- The denominators are fixed, so missing rollouts contribute 0 and missing tasks contribute 0.
- Docs: "Partial submissions are allowed. Missing rollout instances count as zero in the final score." (`docs/challenge/submission.md:11`). `main:docs/challenge/updates.md:61` adds that hidden-test evaluation follows the tasks and instances covered by the submission, and unsubmitted ones count as zero.
- Verified on all 23 shipped 2025 submission files: per-task q = sum(per_rollout)/10 for every task, task_sr = count(q==1)/10, and overall = mean over 50 tasks. Examples include `standard.hidden.SimpleAI_Robot...json`, which has rollouts for only 15 of 50 tasks, and `privileged.public.MR.MR...json`, which has 40 empty tasks (`turning_on_radio` = `{}` giving 0.0). **[DATA]**

**`task_sr` is derived from `q == 1`, not from the JSON's `success` field [CODE]** (`score_utils.py:341`). The two are equivalent, since Q = 1.0 iff success except in the degenerate case of a fully newly-satisfied option without PredicateGoal success. **[INFERRED]**

**Denominator: 10 vs 20 [CODE vs DOCS]:**
- HEAD divides by 20 for public. The docs prescribe 10 public instances for reported results (`evaluation.md:66,282`) and 1,000 rollouts total = 100 x 10 x 1 (`submission.md:10`). Final evaluation holds out "10 more instances" (`evaluation.md:290`), while the code has 20 hidden.
- 2025 `score_utils` hard-coded `/10` and used task-specific public ids from `test_instances.csv`, with hidden ids `range(10)` (`v3.7.2:score_utils.py:61-77, 113-119`). The 2025 hidden JSON keys are `"0_0".."9_0"`. **[DATA]**
- Whether the 2026 HF leaderboard divides by 10 or 20 is **NOT FOUND**.
- **Recommendation:** report both, and treat `/10` over instance ids 301..310 as the documented definition. Running HEAD's `score_utils` on a 10-instance public set gives exactly half.

**Leaderboard-side hidden evaluation selection (2025 infra, `B1K/eval-jobqueue`, [CODE]):**
- `generate_jobs.py:212-234` sorts submissions by `overall_scores.q_score` and takes the top 5 standard and top 5 overall. It schedules hidden jobs **only for tasks with public per-task `q_score > 0`**, except two hard-coded teams.
- It skips instances that already have results (`:259-269`).
- Results are written to `/vision/group/behavior/eval-results/<team_slug>/metrics/<task>_<instance_id>.json`, with no rollout id (`README.md:209-213`).
- `check_results.py:136-157` is a monitoring tool. It averages q over **finished** evals only (`eval_q_score = sum/finished`), so it is not the official aggregate. Missing results default to `q_score = 0.0` (`:136-139`), and team slugs are hard-coded to 2025 teams (`:100-105`).
- `eval.sh:32` launches `omnigibson.learning.eval_with_jobqueue`, but `OG/learning` does not exist at HEAD. **This infra is 2025-era and stale for 2026. [CODE]**
- 2026 docs: top-5 (`evaluation.md:290`) or "top few" (`main:updates.md:59`) are evaluated on hidden instances, and hidden scores **replace** public scores for final ranking (`main:updates.md:23`). **[DOCS]**

---

## 5. What the leaderboard displays

**In-repo `docs/gen_leaderboard.py` [CODE]** generates `challenge/archive/2025/leaderboard.md` only (`:93`).
- `load_submissions` (`:28-87`) reads `docs/challenge_submissions/*.json` and requires `overall_scores`. It merges public and hidden entries **per team name** (`:61-64`), so a team's later file of the same testset overwrites the earlier one, in glob order.
- Sort key (descending, `:71-86`):
  1. `max(hidden_q_score, public_q_score)`, missing = 0.
  2. Mean of the public base, left and right distance scores.
  3. `public_time_score`.
- Columns (`:122-174`): Rank, Team, Affiliation, Track, Release (code/report links), **Full Task Success Rate** (Public Validation / Held-out Test), **★ Q Score** (Public Validation / Held-out Test, hidden in bold), Date. Values use 4 decimals. Time and distance scores are **not displayed**; they serve only as tie-breakers.
- The "About Q-score" banner reads: "fraction of completed sub-goals and choosing the best-matched goal clause. It awards partial credit..." (`:100-107`).

**2026 leaderboard:** the Hugging Face Space `huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard` (`docs/challenge/index.md:57`), with submissions via `behavior-1k-2026-challenge-leaderboard.hf.space/submit` (`submission.md:4`). Index page: "Ranking metric: Average task success score with BDDL partial credit" (`index.md:119-120`). **[DOCS]**
- Its source and its aggregation are **NOT FOUND**. huggingface.co is blocked for curl/WebFetch; `gh search code` returned HTTP 403 ("sessions are bound to their configured repositories"); WebSearch "behavior-1k 2026 challenge leaderboard huggingface space q_score" returned no Space code.
- Leaderboard scores were hidden in the last submission week (`main:updates.md:23`). **[DOCS]**

**Submission JSON schema, from the 2025 files [DATA]:**
- Top keys: `team, affiliation, date, track, testset, overall_scores, per_task_scores, per_rollout_scores`.
- `per_task_scores` holds 6 metrics x 50 tasks.
- `per_rollout_scores` holds 5 metrics (no task_sr) x task to `{"<iid>_<rid>": value}`.

Example (`standard.hidden.Robot_Learning_Collective...json`): overall q 0.25985, task_sr 0.124, time_score 0.1078, base/left/right distance 1.276/0.366/0.414. Partial credit made up 0.136 of the 0.260 Q: 62 successes, 169 fractional rollouts with mean 0.402, and 269 zeros out of 500. **[DATA]**

---

## 6. Reusable aggregation pseudo-code (reuses official per-episode metrics; no re-scoring)

Tested: `scratchpad/research/tools/aggregate_official.py`. The self-test `tools/selftest_aggregate.py` rebuilds per-episode JSONs from 2025 `per_rollout_scores` and reproduces the official `overall_scores` exactly (all 6 metrics, 1e-6) for `standard.hidden.Robot_Learning_Collective...` and `standard.hidden.SimpleAI_Robot...` (35 tasks missing). **[DATA]**

```python
EVAL_TIMEOUT_MULTIPLIER = 1.5                       # eval_utils.py:17
FNAME = r"^(?P<task>.+)_(?P<inst>\d+)_(?P<roll>\d+)\.json$"   # evaluator.py:681 naming; task names contain "_", so rsplit("_",2)

def time_score(nt, M=1.5):                          # score_utils.py:321-324
    return M/(M-1) - 1/((M-1)*nt)                   # = 3 - 2*steps/human_len

def aggregate(json_dir, task_names,                 # all 100 challenge tasks (B100_task_misc.csv / task_data.json)
              expected_instance_ids,                # DOCS: [301..310] (public idx 0-9); official HEAD code: [301..320]
              rollout_ids=(0,), strict=True):
    expected = {f"{t}_{i}_{r}.json" for t in task_names for i in expected_instance_ids for r in rollout_ids}
    N = len(expected_instance_ids) * len(rollout_ids)      # FIXED denominator (score_utils.py:300,340)
    per = {m: defaultdict(dict) for m in ("q","time","base","left","right")}
    for fn in listdir(json_dir):
        if not fn.endswith(".json"): continue
        if fn not in expected: (raise if strict else warn)          # score_utils.py:313
        task, inst, roll = fn[:-5].rsplit("_", 2)                    # score_utils.py:316
        r = json.load(open(fn))                                      # accepts Infinity
        key = f"{inst}_{roll}"
        per["q"][task][key]    = r["q_score"]["final"]               # reuse official per-episode Q verbatim
        per["time"][task][key] = time_score(r["time"]["normalized_time"])
        per["base"][task][key], per["left"][task][key], per["right"][task][key] = (
            r["normalized_agent_distance"]["base"], r["normalized_agent_distance"]["left"], r["normalized_agent_distance"]["right"])
        # 2026-eval: if "failure_reason" in r, q is already 0.0 and time_score evaluates to 0 -> no special-casing needed
    per_task = {
      "q_score":   {t: sum(per["q"][t].values())/N for t in task_names},
      "task_sr":   {t: sum(v == 1 for v in per["q"][t].values())/N for t in task_names},
      "time_score":{t: sum(per["time"][t].values())/N for t in task_names},
      "base_distance_score":  {t: sum(per["base"][t].values())/N  for t in task_names},
      "left_distance_score":  {t: sum(per["left"][t].values())/N  for t in task_names},
      "right_distance_score": {t: sum(per["right"][t].values())/N for t in task_names},
    }
    overall = {m: sum(v.values())/len(task_names) for m, v in per_task.items()}  # missing task -> 0
    coverage = {"found": ..., "expected": len(expected), "missing": sorted(expected - found)}
    return overall, per_task, coverage
```

Practical notes:
- **Prefer the official function when possible.** It is `from omnigibson.eval.utils.score_utils import compute_final_q_score`, or the CLI `python -m omnigibson.eval.utils.score_utils -i <dir_of_submission_dirs> -o <out>` (`score_utils.py:400-472`). It needs OmniGibson importable and `gm.DATA_PATH/2026-challenge-task-instances/metadata/B100_task_misc.csv` present, because `eval_utils` raises at import otherwise (`eval_utils.py:60-77`). It also needs the directory name `<track>.<testset>.<team>.<affiliation>.<date>`, with `track` set to `standard` (the 2026 docs have a single track, but the code still asserts `standard`/`privileged`, `score_utils.py:250`). The reimplementation above only re-does the trivial averaging and keeps `N` explicit, because of the 10-vs-20 issue.
- Note that the official CLI iterates over **every subdirectory** of `--input_dir` (`score_utils.py:466-471`).
- Keep one rollout per instance (rollout_id 0). Re-running and picking the best rollout is prohibited (`main:updates.md:25`, `evaluation.md:286`, `submission.md:13`).
- When comparing internal variants, always report coverage alongside Q. A missing JSON (crash at HEAD) silently becomes 0.

---

## 7. Page vs code discrepancies

1. **Q definition.** Docs say "satisfied at episode end / total goal predicates" (`evaluation.md:231`). Code counts **newly** satisfied literals relative to episode start, takes the **max over grounded goal options**, and returns 1.0 on success (`task_metric.py:6-28`).
2. **Denominator.** Docs report 10 public instances (idx 0-9) and 100x10 rollouts (`evaluation.md:66,282`; `submission.md:10`). HEAD `score_utils` divides by `NUM_PUBLIC_TEST_INSTANCES = 20` and allows ids 301..320 (`score_utils.py:289-300`; `eval_utils.py:18-21`). Hidden: docs say "10 more instances" (`evaluation.md:290`) vs code `NUM_HIDDEN_TEST_INSTANCES = 20`.
3. **Hidden indices.** Docs say "indices 20-39 are hidden instances" (`evaluation.md:66`). Code requires `--mode hidden_test` with indices in `range(20)` and asserts otherwise (`evaluator.py:83-87`; `eval.py:56-64`).
4. **Sample JSON.** It shows `"instance_id": 0` (`submission.md:49`). The code writes the resolved test id, e.g. 301 (`evaluator.py:676`; `eval.py:130,183`).
5. **"Each successful rollout produces a JSON"** (`evaluation.md:129`). At HEAD every finished rollout writes a JSON, success or not (`evaluator.py:680-682`). On 2026-eval, failed rollouts also write one with `q_score.final = 0` and `failure_reason`.
6. **Secondary metric "Simulated time"** (`evaluation.md:235`). The leaderboard-side metric is `time_score = 3 - 2*steps/human_len`, computed for failures too, not raw simulated time (`score_utils.py:321-324`).
7. **Human normalization "from 200 demonstrations per task"** (`evaluation.md:241`). The code uses whatever episodes are in `task.jsonl` (the mean over `num_episodes`, `score_utils.py:215-233`). The file itself was not available here.
8. **Leaderboard generator.** `docs/gen_leaderboard.py` targets the 2025 archive page only (`:93`). The 2026 leaderboard is on an HF Space whose code is NOT FOUND.
9. **eval-jobqueue is 2025-era.** It calls `omnigibson.learning.eval_with_jobqueue` (`eval.sh:32`), which is absent at HEAD. Its metrics paths have no rollout id (`README.md:212`), it hard-codes 2025 team slugs (`check_results.py:100-105`), and it selects hidden jobs only for tasks with public q > 0 (`generate_jobs.py:230-234`).
10. **Failure scoring exists only on the 2026-eval branch**, not at v3.9.3-post2 (`2026-eval:evaluator.py:827-852`, `2026-eval:docs/challenge/evaluation.md:115`).

---

## 8. Open questions

- Does the 2026 HF leaderboard divide by 10 (documented) or 20 (HEAD code) instances per task? Does it use `score_utils` from v3.9.3-post2 or a private copy?
- Which evaluator code produced or will produce the official hidden-test JSONs: `2026-eval` (failure means Q=0, group time budget) or v3.9.3-post2? `main:updates.md:51` says v3.9.3-post1 will be used for final hidden evaluations, but the 2026-eval branch postdates it.
- `task.jsonl` human stats (length, distances) for the 100 tasks are in the data bundle, not in the repo, so `max_steps` per task cannot be listed here.
- Ties are broken by "secondary metrics" (docs). For 2026, the exact combination (gen_leaderboard used mean public distance, then public time) is unknown.
- Does a `forn` goal (exactly N when evaluating success, minimal N-combinations for options) ever over-penalize satisfying more than N? `NQuantifier.evaluate` uses `sum == N` (`condition_evaluation.py:305`), which means over-satisfying would break success. This was not verified against the BDDL authors' intent.

---

## 9. Implications for our modular agent

1. **Only the terminal state counts.** The agent must not undo progress late in the episode, e.g. re-grasping an already placed object, reopening a closed appliance, or leaving an object in hand when the goal says "on/inside". Before the step budget runs out, a "stabilize / do no harm" mode is worth more than a risky last attempt.
2. **Commit to one grounding.** Credit is the max over grounded options of the newly-satisfied fraction. The planner should pick one consistent assignment (which objects go to which container or instance) and complete literals in that option, rather than scattering progress across alternatives. Shortest options give the largest credit per literal.
3. **Literals true at start are worth 0 but stay in the denominator.** Effort should go to literals that are false at start. Literals that were true at start (often negated goals such as "not open") are free credit-neutral, but they must remain true for full success.
4. **Success terminates the episode immediately with Q = 1.** No explicit "done" signal is needed. Partial episodes always run to `max_steps = int(1.5 x human mean length)`.
5. **Efficiency only breaks ties.** Time and distance scores do not affect Q. Do not trade Q for speed. The **wall-clock budget does matter**: under the 2026 rules the budget is `max_steps x 1 s`, and a rollout over budget or disconnected gets Q = 0 on the 2026-eval evaluator. Strong-model planner latency must therefore be amortized (action chunks / `--replay-action-chunk-size`, planning infrequently) so the rollout averages at least 1 step/s.
6. **Coverage is score.** Missing rollouts and tasks count 0, and hidden evaluation follows the tasks and instances that were submitted. Run all 100 tasks x 10 instances (ids 301..310, rollout 0) even if many score low. A crash at HEAD produces no JSON, which means 0.
7. **The metrics JSON cannot explain why Q was partial.** For development analysis, add a separate dev-only logger of `task.get_goal_option_satisfaction()` (initial and final masks plus option literals). Keep it outside the submitted wrapper: privileged state must not reach the policy, and the wrapper is manually inspected (`evaluation.md:144`).
8. **Aggregate our own runs with the official per-episode `q_score.final`** and the fixed-denominator averaging above. Report both N=10 (docs) and N=20 (HEAD code) until the leaderboard convention is confirmed.

Sources (web): https://behavior.stanford.edu/challenge/archive/2025/leaderboard.html (WebSearch snippet: "computing the fraction of completed sub-goals and choosing the best-matched goal clause"); https://behavior.stanford.edu/challenge/leaderboard.html (WebSearch result listing only, content not retrieved).

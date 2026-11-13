# R3 — BEHAVIOR-1K evaluator changes on newer branches (v3.9.3-post2 vs 2026/eval, main, challenge/group-time-budget)

Scope: read-only code/doc diff. No simulator was run.
Repos: `B1K` = `/home/user/stanfordvl/behavior-1k` (HEAD = tag `v3.9.3-post2`, `a8247a8c`, 2026-10-05).
`OPENPI` = `/home/user/wensi-ai/openpi` (branch `behavior`, `0cc8e355`).

Citation conventions:
- `HEAD:path:L` means the file at v3.9.3-post2, which is the working tree.
- `2026-eval:path:L` means `git show origin/2026-eval:path`. That is branch `2026/eval`, `38d2c9fb`, Wensi Ai, 2026-10-19, "Fix cross-scene transition removal before deferred object additions".
- `main:path:L` means `origin/main` (`7d6718cb`, 2026-10-14, "Pin pyarrow<26 (#2369)").
- `gtb:path:L` means `origin/group-time-budget`. That is branch `challenge/group-time-budget`, `ad934aa6`, 2026-10-09, "docs: clarify shared evaluation time budgets for challenge rollouts".
- `v3.9.2-raw:` means the file from `https://raw.githubusercontent.com/StanfordVL/behavior-1k/v3.9.2/<path>`, fetched with curl.
- Status tags: **[code]** verified in code; **[docs]** stated in docs; **[inferred]** my inference; **[NOT FOUND]**.

Copies of the 2026-eval files I read are under
`/tmp/claude-0/-home-user-VAGEN-Lite/9977f130-b514-5e2e-ab6e-7d7795066892/scratchpad/research/r3_refs/eval26/`. The v3.9.2 files are under `.../r3_refs/v392/`.

---

## 0. TL;DR

1. **v3.9.3-post2 (HEAD) already has the following** [code]:
   - vectorized / parallel evaluation (`--num-envs`, `BatchedEvaluator`, `evaluate_instances_batched`);
   - **batched policy transport**: one websocket server receives `(num_envs, ...)` observations and returns `(num_envs, action_dim)`;
   - **opt-in action-chunk replay** (`--replay-action-chunk-size K`, where the chunk must be exactly K long).

   v3.9.2 has none of these. It sends un-batched observations (`v3.9.2-raw:.../evaluator.py:247-251`).
2. **HEAD has none of the following** [code]:
   - no global episode timer;
   - no per-action response timer;
   - no reconnect in the official CLI (`allow_reconnect` defaults to False and `eval.py` never sets it);
   - no multi-port fan-out;
   - no failure records.

   A dropped connection raises `RuntimeError` and aborts the evaluator (`HEAD:.../network_utils.py:165-170`, `HEAD:.../eval.py:144-150`). `recv()` has no timeout (`HEAD:.../network_utils.py:133`).
3. **Only `origin/2026-eval` has the following** [code]:
   - a global/group time budget of `max_steps` **seconds**, which starts after an uncharged first query;
   - a 600 s per-query cap;
   - a 600 s initialization allowance;
   - 3 reconnects per rollout, with the same observation resent;
   - `--policy-endpoints` for multi-port evaluation (one server per env, queried concurrently by `MultiWebsocketPolicy`, with per-rollout time accounting);
   - `--policy-scheme ws|wss`;
   - variable-length chunks (`1..K`);
   - `failure_reason` and zeroed scores in the result JSON;
   - a time breakdown in multi-port mode;
   - `num_rerenders_on_reset=10`;
   - the `r1pro_surround` base-mass override;
   - a multi-env transition-rule fix in `simulator.py`.
4. **origin/main** differs from HEAD only in docs (the 09/28, 10/07 and 10/09 update posts, CoRL venue, contact email) and in the `pyarrow<26` pin. Its eval code is identical to HEAD.
   - The 10/09 post on main *announces* the timers, but main does not implement them [code + docs].
5. **challenge/group-time-budget** differs from HEAD only in `docs/challenge/{index,updates}.md`.
   - Its updates.md proposes a **group budget of `N × max_steps` seconds**. No fetched branch implements that [code].
   - 2026/eval replaced that wording with per-rollout accounting.
6. **For faithful local reproduction, use `origin/2026-eval@38d2c9fb` for the evaluator.** Reasons:
   - It implements the announced final-evaluation rules.
   - It is the newest ref, committed after the 2026-10-16 deadline.
   - Obs, action, wrappers, metrics and scoring are byte-identical to post2.

   Run it with the `RGBDFullResWrapper`, the default `max_steps`, `--write-video` and no tiled rendering. Test both single-port `--num-envs 1` and multi-port `--policy-endpoints`.
7. **To be compatible with both versions, a policy server must:**
   - serve `/healthz`;
   - send the metadata frame immediately (within 10 s);
   - never reply to `{"reset": True}`;
   - return exactly one binary msgpack dict containing `action`, with shape `(B, action_dim)`;
   - if it supports chunks, return exactly K actions;
   - answer each query in < 600 s and average ≤ about 1 s per step overall;
   - keep episode state across reconnections;
   - never block the asyncio loop (so health checks and pings keep working).

   Details are in §7.

---

## 1. Refs, tags and diff stats

### 1.1 Ref inventory
- `git ls-remote --tags origin`, run 2026-10-21, lists `v3.9.0`, `v3.9.1`, `v3.9.2`, `v3.9.3-post1` (`bd049de3`) and `v3.9.3-post2` (`a8247a8c`). **There is no plain `v3.9.3` tag on the remote [NOT FOUND]**, even though every version of `docs/challenge/evaluation.md:34-35` says "Use the `v3.9.3` tag".
- `git ls-remote origin` shows `refs/heads/2026/eval = 38d2c9fb` and `refs/heads/challenge/group-time-budget = ad934aa6`. Both match the local fetched refs.
- All three non-HEAD refs are shallow single commits. `git merge-base` is unavailable, and the GitHub API and atom feeds return 403 from this sandbox. **The intermediate commit history of 2026/eval was NOT FOUND.**

### 1.2 Diff stats
`git diff --stat HEAD origin/2026-eval` (whole tree):
```
 OmniGibson/omnigibson/eval/eval.py                |  53 +-
 OmniGibson/omnigibson/eval/evaluator.py           | 270 +++++++++--
 OmniGibson/omnigibson/eval/policies.py            | 138 +++++-
 OmniGibson/omnigibson/eval/utils/network_utils.py | 209 +++++---
 OmniGibson/omnigibson/simulator.py                |  12 +-
 OmniGibson/omnigibson/transition_rules.py         |   7 +-
 OmniGibson/setup.py                               |   2 +
 OmniGibson/tests/test_evaluator.py                | 565 +++++++++++++++++++++-
 docs/challenge/evaluation.md                      |  26 +-
 docs/challenge/index.md                           |   6 +-
 docs/challenge/updates.md                         |  78 +++
 docs/overrides/index.html                         |   2 +-
 12 files changed, 1251 insertions(+), 117 deletions(-)
```
Files with **no** change between HEAD and 2026-eval [code]:
- `eval/utils/{score_utils,eval_utils,obs_utils}.py`
- `eval/wrappers/*`
- `eval/r1pro.yaml`
- `envs/`
- `metrics/`

The observation content, wrappers, metrics and scoring are therefore identical.

`git diff --stat HEAD origin/main`:
```
 OmniGibson/setup.py       |  2 ++      (pyarrow<26 pin)
 docs/challenge/index.md   |  6 ++--    (CoRL 2026, Nov 9-12, Austin)
 docs/challenge/updates.md | 78 +++    (09/28, 10/07, 10/09 posts)
 docs/overrides/index.html |  2 +-     (contact email)
```
`git diff --stat HEAD origin/group-time-budget`:
```
 docs/challenge/index.md   |  6 ++--
 docs/challenge/updates.md | 80 +++
```
No code changes. Relative to main, gtb lacks the pyarrow pin and the email change: `git diff --stat origin/main origin/group-time-budget` shows `setup.py -2` and `index.html`.

`git diff --stat origin/main origin/2026-eval` consists of the eval code, the simulator/transition change, tests, `evaluation.md`, and a 6-line `updates.md` change. 2026-eval therefore = main + the eval/timer work + the updates.md rewording.

### 1.3 v3.9.3-post1 vs v3.9.3-post2
Fetched with `git fetch --depth 1 origin tag v3.9.3-post1`.
- post1 is `bd049de3`, 2026-09-29, "Fix toggle state access in evaluation light synchronization (#2357)".
- post2 is `a8247a8c`, 2026-10-05, "Add config option to specify presampled_pose_key (#2361)".

```
$ git diff --stat v3.9.3-post1 v3.9.3-post2
 OmniGibson/omnigibson/eval/evaluator.py      | 16 +++++-----
 OmniGibson/omnigibson/tasks/behavior_task.py | 48 +++++++++++++++++++++-------
 docker/gh-actions/Dockerfile                 |  2 +-
 docs/challenge/evaluation.md                 |  2 ++
 4 files changed, 48 insertions(+), 20 deletions(-)
```
Functional content:
- A new helper `get_presampled_robot_poses(presampled_poses, robot_model, pose_key=None)` in `behavior_task.py`, plus a `presampled_pose_key` task kwarg.
- The evaluator passes `cfg["task"]["presampled_pose_key"] = self.robot_eval_config.get("use_presampled_pose_key")` (`HEAD:.../evaluator.py:259-260`) and uses the helper at `HEAD:.../evaluator.py:486-491`.
- Docs add the `eval.use_presampled_pose_key` paragraph (`HEAD:docs/challenge/evaluation.md:218`).
- Dockerfile: GH runner version 2.320.0 → 2.337.0.

The change only matters for custom robots whose `model` is not one of fetch/r1/r1pro/stretch/tiago. The policy protocol is **not** changed [code].
- Docs: "If your submission uses a custom robot, please update to `v3.9.3-post2`. Otherwise, no update is required" (`2026-eval:docs/challenge/updates.md:33`, identical on main).

---

## 2. Feature matrix — what exists where

| Behaviour | v3.9.2 | v3.9.3-post1 / **post2 (HEAD)** | origin/main | **origin/2026-eval** |
|---|---|---|---|---|
| `--num-envs` vectorized batch eval | no [code] | **yes** (`HEAD:eval.py:67-72`, `HEAD:evaluator.py:206-211,292-294`) | same as HEAD | yes (unchanged) |
| Obs sent with leading batch dim | no, un-batched (`v3.9.2-raw:evaluator.py:247-251`) | **yes**, `(num_envs, …)` (`HEAD:evaluator.py:411-421,430`) | same | yes; per-port `(1, …)` (`2026-eval:policies.py:200`) |
| Batched transport (1 server, B envs) | n/a | **yes** (`HEAD:evaluator.py:423-437`) | same | yes, the default when `--policy-endpoints` is absent |
| Multi-port fan-out (1 server per env) | no | no | no | **yes**: `--policy-endpoints`, `MultiWebsocketPolicy` (`2026-eval:eval.py:61-66,172-180`; `policies.py:126-233`) |
| Action-chunk replay | no | **yes**, chunk must be **exactly K** (`HEAD:network_utils.py:146-157`) | same | yes, chunk of **1..K** (`2026-eval:network_utils.py:225-243`) |
| Global episode / group timer | no | **no** | no (announced only) | **yes**, `max_steps` seconds (`2026-eval:evaluator.py:157,167-195,884`; `policies.py:154-177,218-228`) |
| Per-action response timer | no | **no**; `recv()` has no timeout (`HEAD:network_utils.py:133`) | no | **yes**, 600 s (`2026-eval:network_utils.py:32,190-192,204`) |
| Init allowance + uncharged first query | no | no | no | **yes**, 600 s each (`2026-eval:evaluator.py:58,767-791`) |
| Reconnect in the official CLI | client has `allow_reconnect=False`, never set (`v3.9.2-raw:policies.py:54`) | **no**: class default False, CLI never sets it (`HEAD:policies.py:63`, `HEAD:eval.py:144-150`) | no | **yes**, 3 per rollout, resends the same obs (`2026-eval:eval.py:178,187`; `network_utils.py:31,151-171,208-210`) |
| Failure records (`failure_reason`, q=0) | no | no; the evaluator crashes | no | **yes** (`2026-eval:evaluator.py:836-858`) |
| Per-rollout time breakdown in JSON | no | no | no | multi-port only (`2026-eval:evaluator.py:859-866`) |
| `wss` scheme flag | no | no CLI flag | no | `--policy-scheme` (`2026-eval:eval.py:58-60`) |
| `num_rerenders_on_reset` | default | default 3 (`HEAD:envs/env_base.py:1033`) | 3 | **10** in eval (`2026-eval:evaluator.py:378`) |
| Multi-scene transition-rule ordering fix | n/a | per-scene remove+add (`HEAD:simulator.py:1427-1430`) | same | global remove, then add (`2026-eval:simulator.py:1427-1438`) |
| `pyarrow<26` pin | — | no | yes | yes |

---

## 3. Code changes HEAD → 2026-eval, explained

### 3.1 `network_utils.py` (WebsocketClientPolicy). The server class is unchanged.
The diff hunks stop before `class WebsocketPolicyServer`. On 2026-eval it is at `2026-eval:network_utils.py:277+`, and the hunks are all ≤ line 277 [code].

New constants and exceptions (`2026-eval:network_utils.py:31-40`):
```python
MAX_RECONNECTS_PER_ROLLOUT = 3
POLICY_RESPONSE_TIMEOUT = 600
class PolicyConnectionError(RuntimeError): ...
class PolicyTimeoutError(RuntimeError): ...
```
**Deadline plumbing** (`:86-97`):
- `set_deadline()` stores an absolute monotonic deadline.
- `_remaining(deadline, cap)` raises `PolicyTimeoutError` if the deadline has passed and `PolicyConnectionError` if the client is closed. Otherwise it returns `min(remaining, cap)`.

**`_wait_for_server(max_attempts=None, deadline=None)`** (`:102-149`):
- The health-check timeout is `min(remaining, 2)`; the sleep between attempts is `min(remaining, 5)`.
- The websocket uses `ping_interval=60`, **`ping_timeout=None`**, and `open_timeout=min(remaining,10)`.
- **The metadata frame must arrive within `min(remaining,10)` s** (`:142`).
- It catches `OSError` / `EOFError` / `WebSocketException`.

HEAD by contrast:
- loops forever on the health check (`HEAD:network_utils.py:83-92`);
- uses `ping_timeout=300` (`HEAD:...:103-104`);
- blocks on `conn.recv()` with no timeout (`:106`).

**Reconnect** (`:151-171`):
```python
def _reconnect(self, error, deadline=None):
    self._ws = None
    if not self._allow_reconnect:
        raise PolicyConnectionError(...)
    while self._reconnect_attempts < MAX_RECONNECTS_PER_ROLLOUT:
        self._reconnect_attempts += 1
        try:
            self._ws, self._server_metadata = self._wait_for_server(max_attempts=1, deadline=deadline)
            return
        except PolicyConnectionError as e:
            error = e
            if self._reconnect_attempts < MAX_RECONNECTS_PER_ROLLOUT:
                time.sleep(self._remaining(deadline, 5))
    raise PolicyConnectionError(f"Policy connection failed after {MAX_RECONNECTS_PER_ROLLOUT} reconnect attempts: {error}")
```
- Each reconnect attempt is **one** health check (2 s timeout) plus one ws connect (10 s open, 10 s metadata).
- The counter spans the whole rollout and is reset only in `reset()` (`:253`). Test: `2026-eval:tests/test_evaluator.py:482-507`.

`_ensure_connected` (`:173-182`):
- After a previous connection existed, reconnection goes through `_reconnect`, which is capped at 3.
- The very first connection loops until the deadline.

**`act()`** (`:184-250`):
- Chunk replay is served first, without the network and without a deadline check (`:185-188`).
- `query_deadline = min(now + 600, self._deadline)` (`:190-192`).
- `recv(timeout=remaining)` (`:204`). On `TimeoutError` or `PolicyTimeoutError`, the client **closes the socket and raises `PolicyTimeoutError`**, with no reconnect (`:205-207`).
- On `OSError` / `EOFError` / `ConnectionClosed`, it calls `_reconnect(...)` and then **`continue`, which resends the same packed observation** (`:208-210`). Test: `2026-eval:tests/test_evaluator.py:418-442` asserts `first_socket.sent == second_socket.sent`.
- A text-frame response raises `RuntimeError` (`:214-215`).
- A missing `action` key is retried up to 2 times by resending (`:218-223`).
- Chunk validation allows `1 <= L <= K` and requires `chunk[...,0,:] == action` exactly (`:225-243`).
  - HEAD requires `chunk.shape == (*action.shape[:-1], K, action_dim)`, i.e. **exactly K** (`HEAD:network_utils.py:149-153`).

**`reset()`** (`:252-265`):
- Zeroes the reconnect counter, clears the chunk, ensures connection, and sends `{"reset": True}` with **no acknowledgment read**.
- Retries via `_reconnect` on a socket error.

**`close()`** (`:272-274`): new. It is called from `BatchedEvaluator.__exit__` (`2026-eval:evaluator.py:906-907`).

### 3.2 `policies.py`
- `LocalPolicy.set_deadline` is a no-op (`2026-eval:policies.py:61-62`).
- `WebsocketPolicy` gains `scheme` and `set_deadline` (`:73-112`). `forward` is unchanged (`:114-118`), including the legacy `need_new_action` cache.
- **New `MultiWebsocketPolicy`** (`:126-233`):
  - one `WebsocketClientPolicy` per endpoint and a `ThreadPoolExecutor(max_workers=len(clients))` (`:132-142`);
  - per-env `elapsed`, `policy_time` and `sim_time` (`:146-149`);
  - `set_time_budget(seconds)` resets all counters and clears client deadlines (`:154-160`);
  - `_run_client` sets the client deadline to `start + (budget - elapsed)`, charges the call duration to `elapsed` and `policy_time`, and raises `PolicyTimeoutError` when the budget is spent (`:162-177`);
  - `reset()` resets all clients concurrently and stores per-port exceptions in `self.failures` (`:179-189`);
  - `forward(obs_list, active_env_indices)` queries active, non-failed envs concurrently. Each obs gets `unsqueeze(0)` (`:200`). It accepts `(1, A)` or `(A,)` and requires the shape to be `(action_dim,)` (`:209-212`). **Any** exception becomes a per-env failure (`:214-215`).
  - `charge_simulation(seconds, active)` uses **`share = seconds / len(self.clients)` (a fixed divisor)** and charges only the active envs (`:218-228`).

### 3.3 `evaluator.py`
- `POLICY_INITIALIZATION_TIMEOUT = 600.0` (`2026-eval:evaluator.py:58`).
- `BatchStepResult(terminated, truncated, failures, advanced, failures_after_step)` (`:94-100`).
- **`evaluate_instances_batched`** gains `time_limit_seconds`, `start_fn` and `initialize_fn` (`:103-221`). The order is:
  1. `load_fn` (scene and instance load; uncharged) (`:147`).
  2. `initialize_fn()`. A `PolicyConnectionError` or `PolicyTimeoutError` fails all rollouts (`:149-155`).
  3. `deadline = monotonic() + time_limit_seconds` (`:157`). `start_fn(deadline)` is called (`:158-165`).
  4. Loop:
     - if the group has timed out, fail all as `"group_time_budget_exceeded"` (`:170-172`);
     - call `step_fn`. Any raised `PolicyTimeoutError` or `PolicyConnectionError` fails **all active** rollouts (`:174-183`);
     - advance `step` only if `advanced` (`:191-193`);
     - **check the group deadline again before processing terminations** (`:194-195`);
     - per-env failures are recorded at `previous_step`, or at `step` if the failure happened after the sim step (`:198-215`);
     - then process normal term/trunc (`:216-219`).
- `self.max_steps` is stored (`:372`). `cfg["env"]["num_rerenders_on_reset"] = 10` (`:378`).
- The base-mass override is extended to `r1pro_surround` (`:431-434`). That model is **not present in any robots file of either ref** (`git grep -i surround` → only evaluator.py) [NOT FOUND], so presumably it is an external or custom model [inferred].
- **`_step_fn`** (`:506-538`):
  - Multi-port uses `policy.forward(list_of_obs, active)` and reads per-env failures. Single-port uses `policy.forward(obs=self._batch_obs())`.
  - It accepts a prefetched `policy_result`, the uncharged first query.
  - Only healthy envs receive actions; others get zeros.
  - `_apply_actions` is timed and passed to `charge_simulation` (`:529-535`). The timed span includes `env.step`, light sync, obs preprocessing and metrics.
  - If no env is healthy, the simulator is **not** stepped (`advanced=False`) (`:537-538`).
- `policy.reset()` is **removed from `load_batch`** (HEAD has it at `HEAD:evaluator.py:563`). It now happens in `initialize_fn`.
- **`run()`** (`:739-887`):
  - `initialize_fn` (`:767-791`):
    - set the 600 s init deadline (single-port) or a 600 s budget (multi-port);
    - `policy.reset()`;
    - clear the deadline/budget, keeping per-port init failures;
    - then **one uncharged first query** with all envs, bounded by the client's 600 s cap.
  - `start_fn` (`:793-798`): multi-port calls `set_time_budget(self.max_steps)`; single-port calls `set_deadline(deadline)`.
  - `step_fn` (`:800-812`) consumes the prefetched first result once. Video frames are skipped for failed or non-advanced envs.
  - `record_fn` (`:814-876`), in failure cases:
    - `success=False`;
    - metrics are aggregated only if `step > 0`;
    - `failure_reason ∈ {policy_connection_lost, group_time_budget_exceeded, policy_error, policy_query_timeout}`;
    - `q_score.final = 0.0`, `normalized_agent_distance = 0`;
    - `time.normalized_time = 1/EVAL_TIMEOUT_MULTIPLIER` (= 1/1.5) (`:836-858`).

    For multi-port only, it also adds `time.policy_time`, `time.sim_time`, `time.evaluation_time` and `time.evaluation_fps` (`:859-866`).
  - `time_limit_seconds = self.max_steps if isinstance(self.policy, WebsocketPolicy) else None` (`:884`). The **single-port path therefore has one wall-clock group deadline of `max_steps` seconds, whatever `num_envs` is**. The multi-port path has no group deadline, only per-rollout accounting.

### 3.4 `eval.py`
- `parse_policy_endpoints` requires exactly one `HOST:PORT` per env (`2026-eval:eval.py:37-50`).
- New flags `--policy-scheme {ws,wss}` (`:58-60`) and `--policy-endpoints HOST:PORT ...` (`:61-66`).
- The `--replay-action-chunk-size` help now reads "maximum chunk length … servers may return 1 to this many actions" (`:99-108`).
- **Both websocket paths now set `allow_reconnect: True`** (`:178,187`).

### 3.5 `simulator.py` + `transition_rules.py`
- `TransitionRuleAPI.step()` is split into `get_transition_results()`, which evaluates rules, and `execute_transition()` (`2026-eval:transition_rules.py:156-185`).
- In `Simulator._non_physics_step`, results are gathered for **all** scenes first. All removals are then batch-removed globally, and only after that are additions executed per scene (`2026-eval:simulator.py:1427-1438`):
```python
transitions = [(scene.transition_rule_api, scene.transition_rule_api.get_transition_results()) for scene in self.scenes]
# Remove across all scenes before adding objects, whose initialization is deferred.
removed_objs = [obj for _, result in transitions for obj in result.remove]
if removed_objs:
    self.batch_remove_objects(removed_objs)
for api, result in transitions:
    api.execute_transition(added_obj_attrs=result.add, removed_objs=[])
```
- With one scene (`num_envs=1`) the order is the same as HEAD's (remove, then add) [inferred from code]. The fix only matters for `num_envs > 1`.
- The commit date (2026-10-19, after the 10-16 deadline) suggests the organizers run multi-env evaluation with transition rules [inferred].

### 3.6 `setup.py`
- Only the `"pyarrow<26"` pin, with the comment "pyarrow 26 drops numpy 1.x support, but isaacsim-kernel pins numpy==1.26.0" (`git diff HEAD origin/2026-eval -- OmniGibson/setup.py`). It is also on main.

### 3.7 `tests/test_evaluator.py` (+565 lines). The tests are the executable specification:
- connection failure after a successful step does not advance the step (`2026-eval:tests/test_evaluator.py:61-82`);
- the group deadline starts after load **and** after init: `observed_deadlines == [1154]`. Completed rollouts are preserved and others get `group_time_budget_exceeded` (`:99-133`);
- a query timeout fails only the active rollouts (`:136-157`);
- a one-port failure does not stop other envs. The failed env's step stays 0 (`:160-191`);
- all ports failing means no sim advance (`:194-209`);
- the wss scheme is preserved (`:212-230`);
- multi-port queries run concurrently in env order (`:233-267`);
- the sim share is charged to survivors with a fixed divisor ("still dividing by both envs") (`:270-313`);
- a healthy port is stepped after another port fails (`:316-364`);
- budget = own query time + sim share (`:367-404`);
- endpoint parsing (`:407-415`);
- reconnect resends the same observation (`:418-442`);
- failure after 3 reconnects (`:445-462`);
- server unavailable at reset: attempts `[None,1,1,1]` (`:465-479`);
- the reconnect budget spans steps (`:482-507`);
- the shorter group deadline wins and a timeout does not trigger reconnect (`:510-536`);
- the 600 s cap (`:539-553`);
- reconnect uses the same query deadline (`:556-582`);
- a failed-result JSON is written: `steps=0`, `failure_reason=policy_connection_lost`, `normalized_time=2/3` (`:585-614`).

### 3.8 `docs/challenge/evaluation.md`, 2026-eval additions
- `--policy-endpoints` paragraph and table row (`2026-eval:docs/challenge/evaluation.md:52,66-69`).
- Initialization allowance (`:111`): "After scene initialization and instance loading, initial policy connection and sending the rollout reset have a separate 600-second allowance per endpoint, excluded from the episode budget. The current reset message has no acknowledgment. The evaluator does not restart participant servers."
- First-query allowance (`:113`): the first action query must finish within 600 s and is excluded from the budget. Its actions and chunk are executed normally and every simulator step is charged. A failed first query fails the rollout.
- Accounting (`:115`):
  - accounting starts after the first query completes;
  - multi-port: per-rollout budget of max_steps seconds = own policy time (including reconnections) + sim step time / total envs (fixed divisor);
  - waiting for other ports is not charged;
  - each query must finish within 600 s and the remaining time;
  - up to 3 reconnects, which resend the current observation without advancing the step;
  - a failed or over-budget rollout scores 0;
  - recorded evaluation time and FPS exclude init and the first query.
- Chunks: `L` with `1 <= L <= K`, which may vary per inference. Each port replays its own chunk, and budget checks remain in effect (`:119-143`).

---

## 4. Requested topics, one by one

| Topic | v3.9.3-post2 (HEAD) | 2026/eval only |
|---|---|---|
| **Global episode timer** | Absent [code]. Only the step cap `max_steps` (`termination_config`, `HEAD:evaluator.py:281-289`) applies. | Single-port: wall-clock deadline `max_steps` s for the whole batch, starting after the first query (`2026-eval:evaluator.py:157,884`). Multi-port: per-rollout accounted budget `max_steps` s (`policies.py:154-177,218-228`; `evaluator.py:796`). On expiry: `failure_reason="group_time_budget_exceeded"` (single-port) or `policy_query_timeout` (multi-port `PolicyTimeoutError`), with q=0. |
| **Per-action response timer** | Absent; `self._ws.recv()` blocks forever (`HEAD:network_utils.py:133`). There is only a 300 s keepalive `ping_timeout` (`:103-104`). | 600 s cap, further limited by the remaining budget (`2026-eval:network_utils.py:32,190-192,204`). A timeout closes the socket and fails the rollout (single-port: all active rollouts in the batch). |
| **Time budgets** | none | Init 600 s (`evaluator.py:58,767-781`); first query uncharged ≤600 s (`:783-791`); episode = `max_steps` s. Docs: about 10.85 h for the longest task (`2026-eval:updates.md:15`) ⇒ about 39,060 s ⇒ max_steps ≈ 39,060 for that task [inferred arithmetic]. |
| **Reconnection** | Client supports `allow_reconnect` (2 retries, then waits indefinitely), but **the CLI never enables it**. A disconnect raises `RuntimeError("Websocket connection error")` and the run aborts with no JSON (`HEAD:network_utils.py:165-170`; `eval.py:144-150`). | Enabled. Max 3 per rollout, each a single health-check + connect attempt, inside the query deadline. The same obs is resent with no reset. Failure gives `policy_connection_lost` and q=0 (`network_utils.py:151-171,208-210`; `evaluator.py:837-838`). |
| **Batched policy transport** | **Exists**: one server receives `(num_envs, …)` and must return `(num_envs, A)` (`HEAD:evaluator.py:411-437`). Finished envs keep their last obs in the batch so positions stay aligned (`:411-421`). | Unchanged; plus the multi-port alternative. |
| **Action-chunk replay** | **Exists**, opt-in, exactly K (`HEAD:network_utils.py:114-117,122-125,146-162`; docs `HEAD:evaluation.md:105-127`). | 1..K (`2026-eval:network_utils.py:225-243`). Replayed actions are not timed (`:185-188`) but sim steps are. |
| **Vectorized / parallel eval (`--num-envs`)** | **Exists**. `len(instance_indices)` must equal `num_envs` (`HEAD:eval.py:132-136`). Finished envs get zero actions, keep stepping, and are recorded at their finish step (`HEAD:evaluator.py:98-103,423-437`). | Same; plus failure isolation per port and the no-advance rule when every env has failed. |
| **Obs / action format** | Flattened obs + `<robot>::cam_rel_poses` + `task_id` (`HEAD:evaluator.py:576-592`), with a leading batch dim. | **Unchanged** (`_preprocess_obs` and wrappers untouched). Multi-port sends `(1, …)` per port. |
| **Metrics output** | `{task, instance_id, rollout_id, steps, success, q_score, time{simulator_steps, simulator_time, normalized_time}, agent_distance, normalized_agent_distance}` (`HEAD:evaluator.py:676-679`; metrics `task_metric.py:74-81`, `agent_metric.py:55-65`) | Plus `failure_reason` and forced zeros on failure; multi-port also adds `time.{policy_time, sim_time, evaluation_time, evaluation_fps}` (`2026-eval:evaluator.py:836-866`). `score_utils` is unchanged. A failed rollout gets a time score of 1.5/0.5 − 1/(0.5·(2/3)) = 0 (`HEAD:score_utils.py:321-324`) [inferred arithmetic]. |

---

## 5. Docs-only branches: what the `updates.md` text says

| Item | HEAD | main (`main:docs/challenge/updates.md`) | group-time-budget (`gtb:…`) | 2026-eval (`2026-eval:…`) |
|---|---|---|---|---|
| Latest post | 08/24 (`HEAD:updates.md:7`) | 10/09, 10/07, 09/28 added | same + "We're excited…" (`:27`) | same as main except items 2-4 of 10/09 |
| 10/09 item 2 | — | "**Average evaluation speed:** Each rollout must average at least **1 FPS** … enforce … episode timeout of `max_steps × 1 second`, excluding scene startup … approximately **10.85 hours** … longest task" (`:15`) | "the group has a wall-clock budget of `N × max_steps` seconds … group of 10 approximately **108.6 hours**" (`:15`) | "budget of `max_steps × 1 second` per rollout … each rollout is charged for its own policy queries … plus an equal share of each shared simulator step among rollouts **active** for that step … The simulator share changes as rollouts finish or fail" (`:15`) |
| 10/09 item 3 | — | 600 s per round trip and within the remaining episode time (`:17`) | … remaining **group** budget (`:17`) | … that **rollout's** remaining budget (`:17`) |
| 10/09 item 4 | — | up to 3 reconnects; reconnect time counts (`:19`) | counts toward the group budget (`:19`) | counts toward that rollout's budget (`:19`) |
| 09/28 rules | — | Use `v3.9.3-post1`; hidden-test evaluation will use `v3.9.3-post1` (`:51`). No tiled rendering for final evaluation (`:53`). Same IP/port per task-instance for history-dependent policies; 3 reconnects (`:57`). "We will introduce a global episode timer and a per-action response timer" (`:63`). | same | same |
| 09/28 features | — | vectorized envs; `--num-envs`; "batched policy transport and optional action-chunk replay" (`:77-81`) | same | same |
| 10/07 | — | custom robots → `v3.9.3-post2` (`:33`) | same | same |

`index.md` on main, gtb and 2026-eval sets the event to CoRL 2026, November 9-12, 2026, Austin, Texas (`main:docs/challenge/index.md:52-54`).

`submission.md` is unchanged across refs:
- Docker-based evaluation: "We run OmniGibson outside the container and connect to your policy through the WebSocket policy client" (`HEAD:docs/challenge/submission.md:85`); single 24 GB GPU (`:86`).
- IP-based submissions must "expose at least 50 ports for parallel evaluation" (`:90`).
- The README must include "the full evaluator command" (`:120`).

---

## 6. Page-vs-code / doc-vs-doc discrepancies

1. **Fixed vs changing simulator-share divisor.**
   - `2026-eval:docs/challenge/updates.md:15` says "an equal share … among rollouts active for that step … The simulator share changes as rollouts finish or fail."
   - Code uses `share = seconds / len(self.clients)` with the comment "the divisor stays fixed" (`2026-eval:policies.py:218-220`). The test asserts the fixed divisor (`2026-eval:tests/test_evaluator.py:304-306`), and `2026-eval:evaluation.md:115` says the divisor stays fixed.
   - Code and evaluation.md agree; updates.md does not.
2. **Group budget N × max_steps (gtb docs) vs code.** `gtb:updates.md:15` promises a group wall-clock of `N × max_steps` s. No fetched ref implements it.
   - The 2026-eval single-port path gives the whole batch **`max_steps` s of wall clock** (`2026-eval:evaluator.py:884`).
   - The multi-port path uses per-rollout accounting.
3. **Single-port batched N>1 is stricter than the docs imply.** With `--num-envs N` and no `--policy-endpoints`, all N rollouts share one wall-clock deadline of `max_steps` s. That deadline covers batched inference, N-env sim stepping and video writing (`2026-eval:evaluator.py:157,884`).
   - 2026-eval `evaluation.md:115` only specifies the multi-port budget.
   - `updates.md:15` describes per-rollout charging [code vs docs].
4. **"Use the v3.9.3 tag"** (`evaluation.md:34-35` on every ref) conflicts with two facts:
   - no plain `v3.9.3` tag exists on the remote;
   - updates say `v3.9.3-post1` will be used for hidden-test evaluation (`updates.md:51`) and post2 for custom robots (`:33`).

   The timers announced for final evaluation (`updates.md:63`, 10/09 items 2-4) exist **only** on the untagged `2026/eval` branch.
5. **"Each successful rollout produces a JSON result…"** (`evaluation.md:145`). On 2026-eval, failed rollouts also produce JSONs with `failure_reason`. On HEAD, a connection failure produces **no** JSON for the batch, because the exception aborts `run()`.
6. **Success lost on the boundary step** [code]. The group-deadline check after `step_fn` (`2026-eval:evaluator.py:194-195`) runs *before* the termination loop. In multi-port mode, `failures_after_step` is checked before `terminated` (`:198-215`). A rollout that reaches success on the same step its budget expires is therefore recorded as a timeout with q=0. This edge case is undocumented.
7. **Charged time differs between paths** [code].
   - Single-port charges *all* wall-clock time: video writing in `step_fn` and loop overhead.
   - Multi-port charges only the `_apply_actions` duration (env.step, light sync, obs preprocessing, metrics) plus the policy calls (`2026-eval:evaluator.py:529-535,807-811`).
   - Neither doc mentions video-writing cost.
8. **Single-port vs multi-port error handling** [code].
   - Single-port: any non-`Policy*` exception escapes `evaluate_instances_batched` and aborts the evaluator. Examples: `RuntimeError` from a text frame, a bad chunk shape, a chunk/action mismatch, or a missing `action` after retries.
   - Multi-port: every exception becomes `policy_error` for that port only (`2026-eval:policies.py:214-215`; `evaluator.py:201-207`).
9. **evaluation.md does not mention the `--policy-scheme` flag** (grep of `2026-eval:docs/challenge/evaluation.md`). It exists in code (`eval.py:58-60`).
10. **`time.evaluation_time` and `evaluation_fps` are recorded only for multi-port** (`2026-eval:evaluator.py:859-866`). The docs say recorded evaluation time and FPS exclude init and the first query, without saying it is multi-port only.

---

## 7. What a faithful local reproduction should use

**Decision: evaluator code from `origin/2026-eval` at `38d2c9fb`, pinned by hash because the branch is untagged and moving.** The simulator, assets and robot config should be v3.9.3-post2-equivalent; they are identical except for the changes listed in §3.5-3.6.

Reasons:
1. The 10/09 rules (1 FPS-equivalent budget of `max_steps` s, 600 s per query, 3 reconnects) and the 09/28 promise of "a global episode timer and a per-action response timer" (`2026-eval:updates.md:15-19,63`) are implemented only on 2026-eval [code]. HEAD and main would never surface timeout failures.
2. 2026-eval is the newest ref (2026-10-19, after the 2026-10-16 deadline) and contains a multi-env bug fix. That suggests it is the branch the organizers run [inferred].
3. Obs, action, wrappers, metrics and score_utils have zero diff vs post2 (§1.2). A policy validated on 2026-eval sees the same inputs and outputs as on post2. The only behavioural differences are timing, reconnection and failure records, plus `num_rerenders_on_reset=10` (fresher first frame) and the multi-env transition ordering.
4. The submission page requires ≥50 ports for IP-based submissions "for parallel evaluation" (`submission.md:90`). That matches the multi-port `--policy-endpoints` path, which exists only on 2026-eval [docs + inferred].

Recommended local configuration [docs + inferred]:
- `--env-wrapper omnigibson.eval.wrappers.RGBDFullResWrapper` (official RGB+D; `HEAD:docs/challenge/evaluation.md:82,139-140`).
- Omit `--max-steps`, so the default is 1.5 × the human mean (`HEAD:eval/utils/eval_utils.py:17`; `HEAD:docs/challenge/evaluation.md:77-79`). The time budget is derived from it (`2026-eval:evaluator.py:372,884`).
- `--write-video` (required for submissions; `HEAD:docs/challenge/evaluation.md:89-91`).
- **No tiled rendering**, i.e. do not set `gm.ENABLE_TILED_RENDERING` (`updates.md:53`).
- The official instance protocol: `--mode public_test`, indices 0-9 for reporting and 10-19 as a held-out check (`evaluation.md:282`). Hidden instances live in `scene_test/private` (`HEAD:utils/asset_utils.py:137-146`), which is presumably not distributed [inferred].
- Leave `--replay-action-chunk-size` at its default of 0. The organizers run the evaluator and the default is per-step querying (`eval.py:99-108`). Who chooses K for the final run is not documented [NOT FOUND]; the README's "full evaluator command" (`submission.md:120`) is presumably where a participant would request it [inferred].
- Run two modes:
  - **(a)** single-port `--num-envs 1`, the strictest per-rollout wall clock (sim + video + policy);
  - **(b)** multi-port `--num-envs N --policy-endpoints h:p1 … h:pN` with N independent server processes.
- Add a safety margin (for example, target ≤ 0.7-0.8 of `max_steps` seconds), because of the boundary-step issue (§6.6) and slower final-eval GPUs: RTX 3090 / A5000 / TitanRTX (`submission.md:86`) vs the RTX 4090 benchmarks (`HEAD:docs/challenge/evaluation.md:310-371`) [inferred].
- Sim speed reference: RGB+depth at 720/480 ≈ 13.52 FPS on an RTX 4090 with random actions (`evaluation.md:361-368`). That is roughly 74 ms/step of sim [inferred arithmetic], leaving about 0.9 s/step on average for policy plus transport if a rollout used every step.

Also run the HEAD evaluator once as a compatibility check. HEAD is stricter in two ways: chunks must be exactly K, and any disconnect aborts the run.

---

## 8. Policy-server contract that works with BOTH v3.9.3-post2 and 2026/eval

Verified wire facts:
- Both clients use the msgpack + `__ndarray__` extension and the same `Packer`/`unpackb` (`HEAD:network_utils.py:298-342`; unchanged on 2026-eval).
- Both wait for `/healthz` before connecting.
- Both read one metadata frame on connect.
- Both send `{"reset": True}` with no reply expected.
- Both send the observation dict and expect one reply dict containing `action`.

Requirements, with the reasons for each:
1. **Serve HTTP `GET /healthz` → 200 on the websocket port** (`HEAD:network_utils.py:80-92`; `2026-eval:network_utils.py:109-126`). The health check must answer **within 2 s even while an inference is running**: on 2026-eval each reconnect attempt is a single 2 s health check (`:115-124,163`).
   - So run inference off the asyncio loop (`await asyncio.to_thread(...)` or an executor) [inferred]. The OpenPI fork and OmniGibson's reference server both call `self._policy.act(obs)` synchronously inside the async handler (`OPENPI:src/openpi/serving/websocket_b1k_server.py:73`; `HEAD:network_utils.py:233`), which blocks the loop.
2. **Send the metadata frame immediately on connect** (any msgpack dict), within 10 s on 2026-eval (`2026-eval:network_utils.py:140-142`). Do not load models before sending it. Do lazy warm-up in the first query instead: up to 600 s, uncharged, once per rollout (`2026-eval:evaluator.py:783-791`; `evaluation.md:113`).
3. **`{"reset": True}` resets episode state and sends nothing back** (`HEAD:network_utils.py:172-179`; `2026-eval:network_utils.py:252-265`). A reply would be consumed as the next action response and desynchronize the stream: missing `action` → resend [inferred from `:218-223`]. Reset is sent **once per rollout/batch**, not per env (`HEAD:evaluator.py:563`; `2026-eval:evaluator.py:775`). For a batched single port, reset all B slots.
4. **Reply to each observation with exactly one binary frame** containing `{"action": float32 ndarray}`, with shape `(B, action_dim)` where B is the obs leading dim.
   - B = num_envs for single-port; B = 1 per port for multi-port (`2026-eval:policies.py:200,209-212`).
   - Also accept un-batched obs (v3.9.2 style) and return `(action_dim,)` (`v3.9.2-raw:evaluator.py:247-251`).
   - `action_dim` must equal `robot.action_dim` from the submitted robot config (`submission.md:41`).
   - Extra keys such as `server_timing` are ignored.
   - **Never send text frames**: both clients raise `RuntimeError` (`HEAD:network_utils.py:134-135`; `2026-eval:network_utils.py:214-215`).
5. **Chunk protocol (optional).** If the request contains `"__action_chunk_size__": K` (an int in the obs dict), either omit `action_chunk`, which makes the client fall back permanently to per-step requests, or return `action_chunk` with **exactly K** actions: `(B, K, A)`. Exactly K satisfies both HEAD's exact-K check and 2026-eval's 1..K check.
   - `action` must be bit-identical to `action_chunk[..., 0, :]` after the float32 cast, so send both as float32 (`HEAD:network_utils.py:145-155`).
   - If chunks are not supported, ignore or strip the reserved key; it arrives as an extra obs entry.
6. **Latency.** Every reply must arrive within 600 s (hard cap on 2026-eval), and the time per episode must stay within `max_steps` s.
   - Single-port: wall clock including sim and video.
   - Multi-port: own query time + sim share.

   Episode time starts after the first query (`2026-eval:evaluator.py:157,793-798`). On HEAD there is no cap, but a blocked server loop for more than about 300 s after a ping risks a keepalive close (`ping_interval=60, ping_timeout=300`, `HEAD:network_utils.py:103-104`). That raises `RuntimeError` with no reconnect [inferred from websockets keepalive semantics].
7. **Episode state must survive reconnection.**
   - 2026-eval reconnects up to 3 times per rollout and **resends the same observation without a reset** (`2026-eval:network_utils.py:208-210`; test `:418-442`).
   - Keep policy and planner state outside the per-connection handler (OPENPI does this: the shared `self._policy` is in `websocket_b1k_server.py:35,67,73`).
   - Make a resent observation idempotent, e.g. by caching the last obs fingerprint and its response. Otherwise internal chunk cursors or planner step counters advance twice [inferred]. The obs contains no step counter (`HEAD:evaluator.py:576-592`).
8. **Tolerate abrupt client closes mid-inference** (a timeout closes the socket, `2026-eval:network_utils.py:205-207`; the evaluator closes at exit, `evaluator.py:906-907`).
   - Discard the stale result.
   - Accept a new connection while the old handler may still be running: serialize with a lock [inferred].
   - The next rollout's `reset()` reconnects automatically (`2026-eval:network_utils.py:176-177,253-256`).
9. **One process (or isolated state) per port** for multi-port and IP-based evaluation, with ≥50 ports (`submission.md:90`). The same port is used throughout a given task-instance rollout (`updates.md:57`).
10. **Budget arithmetic for a slow (LLM) planner** [inferred]. The average step must stay at or under about 1 s including sim (`updates.md:15`; `2026-eval:evaluator.py:884`). An LLM call of T seconds must be amortized over roughly T/(1 − t_sim − t_exec) cheap steps.
    - Do expensive planning in the uncharged first query when possible.
    - Use asynchronous or background planning with fast low-level executor steps (for example, a pi0.5 receding-horizon cache served per step, as OPENPI does with `action_horizon=16`: `OPENPI:scripts/b1k/serve_b1k.py:35-37,68-75`).

OPENPI server compatibility snapshot [code]:
- It serves `/healthz` (`websocket_b1k_server.py:106-114`), sends metadata on connect (`:59`), handles `reset` with no reply (`:66-68`), and returns `{"action", "server_timing"}` (`:76-86`). It handles batched and un-batched obs (`OPENPI:src/openpi/shared/eval_b1k_wrapper.py:123,160-162`).
- **It does not** implement `action_chunk`, so it is per-step only.
- **It does** run inference synchronously on the event loop (`:73`). That is a risk for 2026-eval reconnect health checks and for HEAD keepalive pings on long inferences [inferred].

---

## 9. Open questions
- Which mode will the organizers actually use for hidden-test evaluation? Options are single-port `--num-envs 1`, single-port batched N, or `--policy-endpoints` with N ports. The value of N and whether `--replay-action-chunk-size` is honoured from the participant README are also unknown. NOT FOUND in docs; `submission.md:90,120` only hints at multi-port.
- Will the final evaluation use the `2026/eval` branch, which carries the timers, or the `v3.9.3-post1`/`post2` tags as the docs state? The docs contradict each other (§6.4).
- Will the organizers fix the boundary-step success-loss (§6.6) or the updates.md divisor wording (§6.1)?
- Will the gtb `N × max_steps` group budget ever be implemented? It is not on any fetched ref.
- `r1pro_surround`: no definition was found in either ref. Is it a participant model or a forthcoming robot?
- The live website text could not be verified: behavior.stanford.edu is blocked, and WebSearch returned only 2025 pages, e.g. https://behavior.stanford.edu/challenge/evaluation.html and https://behavior.stanford.edu/challenge/updates.html, with snippets not matching the 2026 text.
- 2026/eval commit history (individual commits and PRs) is NOT FOUND: shallow clone, and the GitHub API and atom feeds returned 403.

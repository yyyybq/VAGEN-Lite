# R1 Canonical Dev-32 Floor-Effect Diagnostics

Date: 2026-09-16

## Outcome

The requested diagnosis is complete. The valid batch baseline was not rerun. Both
allowed diagnostics completed with zero infrastructure errors on the same frozen
32-source development-regression set and the same two pinned models.

The strongest supported explanation for the 0/32 floor is **weak local spatial
action selection**. On formally verified states exactly one action from success,
pretrained selected an optimal first action on 1/32 sources and v46@250 on 2/32.
At exact distance two, the corresponding figures were 4/32 and 6/32. These scores
use the complete set of optimal first actions to any canonical-success state, not
agreement with one audit certificate.

Changing only execution feedback from historical batched execution to
first-action-only replanning did not lift the floor: pretrained remained 0/32 and
v46@250 remained 0/32. This rules out “execute only one action before observing
again” as a sufficient repair under the frozen prompt, model and protocol. It does
not prove that batch execution has no effect: the read-only baseline analysis shows
that multi-action turns frequently co-occurred with visibility loss.

This remains a development-exposed diagnostic. It does not establish general
model equivalence, show that v46 training was harmful, or support a final
generalization conclusion.

## Frozen validity and resource record

- Valid baseline: `policy_eval_h1_render_v2`; pretrained 0/32, v46@250 0/32,
  64/64 complete, zero infrastructure errors.
- The earlier wrong-K run remains invalid evidence and was not used.
- Camera/runtime fix: `d940372a5d710fddb6275ee2b91866777fbf138a`.
- Baseline report commit: `7cfa08dd`.
- Diagnostic implementation commit:
  `8d0d1dd9dfc3a0d710f2434d23e22ca6fa08a4e7`.
- Models were reused without download or substitution. Frozen manifests:
  pretrained `46f05ff...02c8`; v46 step 250 `1fae9d...374f`.
- Both used actual BF16 inference with no quantization. The requested environment
  set `TORCH_SDPA`; vLLM 0.11.0 actually logged Flash Attention and the native
  PyTorch sampler, matching the valid baseline. Requested and actual backends are
  both retained in the evidence.
- SCO job `pt-zymond97`, name `r1-dev32-floor-diag-h800-r1-0916`, pool `h800`,
  resource `N4lS.Iq.I80.1`: one H800 80GB, 14 CPUs and 240GB RAM.
- Worker: `pt-bf86937b3fc0401791c612eac951d06c-worker-0`; Python 3.12.13;
  official renderer port 8898; InteriorGS root
  `/mnt/umm/users/yinbaiqiao/InteriorGS`.
- Job ran from 15:31:24 to 16:28:59 UTC, succeeded, stopped its renderer and
  released the H800 resource.

All 64 local-state initial images and all 64 first-action-only rollout initial
images matched their frozen official-render evidence pixel-for-pixel. Initial
poses and effective H1 K matched. The policy inputs were oracle-free; certificates,
optimal actions, terminal poses and planner state remained audit-only.

## Existing 64 trajectories: read-only analysis

| Metric | Pretrained | v46@250 |
|---|---:|---:|
| Primitive actions | 384 | 384 |
| Turn actions | 283 (73.7%) | 275 (71.6%) |
| Actually executed multi-action turns | 74 | 66 |
| Multi-action turns containing an out-of-frame state | 67 | 62 |
| Episodes ever out of frame | 32/32 | 32/32 |
| First out-of-frame at primitive 1 | 24/32 | 21/32 |
| First out-of-frame occurred in a multi-action turn | 9/32 | 9/32 |
| Later actions remained in that same turn | 7/32 | 7/32 |
| Episodes with margin improvement and visibility degradation | 24/32 | 20/32 |
| Episodes ever canonical-success mid-primitive | 0/32 | 0/32 |
| Adjacent repeated actions | 83 | 100 |
| Adjacent inverse actions | 102 | 89 |

The complete first-out distributions were pretrained: step 1/2/3/4 =
24/3/2/3; v46: step 1/2/3/5/9 = 21/5/4/1/1. Both policies lost useful framing
very early. Multi-action execution is therefore a plausible aggravating factor,
but it is not the full explanation: most first visibility losses did not begin in
a multi-action turn, and no batch continued after an already-achieved canonical
success. Certificate first-action match was not treated as correctness evidence.

## Diagnostic A: exact one-/two-step local action selection

For every parent source, the preparation pass found at most one `d*=1` and one
`d*=2` state along the verified certificate, producing exactly 64 derived states
from 32 parents. Distance was then independently certified by exhaustive formal
runtime enumeration to the **entire** canonical success region:

- `d*=1`: initial state is not successful and at least one legal primitive action
  reaches success;
- `d*=2`: no legal one-step successor is successful and at least one legal
  two-step path reaches success.

No RGB quality or observability predicate filtered the runtime state graph. Every
optimal first action was saved, and the policy saw only the normal task text,
current RGB and current pose.

| Exact distance | Pretrained optimal first action | v46@250 optimal first action |
|---|---:|---:|
| 1 | 1/32 (3.125%) | 2/32 (6.25%) |
| 2 | 4/32 (12.5%) | 6/32 (18.75%) |

Paired at `d*=1`: both hit 0, pretrained-only 1, v46-only 2, both miss 29.
Paired at `d*=2`: both hit 1, pretrained-only 3, v46-only 5, both miss 23.
There were no empty/illegal first actions. Pretrained had one collision attempt;
v46 had none. The selected action actually completed 1/32 pretrained and 2/32
v46 `d*=1` states, agreeing with the formal optimal-action labels.

The first action often increased a scalar diagnostic while damaging the complete
gate set. On `d*=1`, pretrained improved score on 16 states but lost inside-frame
on 17; v46 improved score on 15 but lost inside-frame on 14. On `d*=2`, the
corresponding counts were 26/14 and 20/11. This is why scalar score movement is
not a substitute for canonical success.

Representative local successes are audit examples, not independent episodes:

- pretrained `id_test:1285`, `d*=1`: `move_left`, canonical success;
- v46 `id_test:7433`, `d*=1`: `move_left`, canonical success;
- v46 `validation_proxy:7`, `d*=1`: `move_right`, canonical success.

The low hit rate at exact distance one is direct evidence that the floor already
exists at the local perception/action-selection layer. It cannot be attributed
only to long-horizon credit assignment or a particular expert certificate.

## Diagnostic B: first-action-only replanning

The 32 original certified-Medium initial states were rerun once per model. The
prompt, no-concat current-RGB interface, decoding, per-turn seeds, model, H1
camera, canonical runtime and 12-primitive limit stayed frozen. The sole behavior
change was: parse the normal response, execute only its first action, discard the
rest, render a new RGB and query again. Invalid first actions retained baseline
semantics; no later valid action was substituted.

| Metric | Pretrained | v46@250 |
|---|---:|---:|
| Complete / success | 32 / 0 | 32 / 0 |
| Model calls | 384 | 384 |
| Calls added versus valid batch baseline | +77 | +67 |
| Executed primitive actions | 384 | 382 |
| Raw multi-action outputs | 97 | 76 |
| Discarded later actions | 104 | 80 |
| Collision attempts | 1 | 2 |
| Invalid turns | 0 | 2 |
| Model inference time | 233.54s | 245.25s |

Paired result: both success 0, pretrained-only 0, v46-only 0, both fail 32. The
first decision was identical to the valid batch baseline on 32/32 sources for
each model, as required by identical first input and seed. Every episode again
timed out at the 12-primitive limit.

Pretrained still rotated on 321/384 actions (83.6%); v46 rotated on 311/382
executed actions (81.4%). Thirty-two pretrained and 31 v46 episodes went out of
frame. Pretrained first lost framing at step 1 on 24 sources; v46 did so on 21.
Only 5/32 episodes per model ever reached the relation gate, and neither model
combined it with all visibility and margin gates.

Thus the additional feedback and +144 total inference calls did not produce a
single paired recovery. This does not prove that feedback frequency is never
useful; it shows that first-action-only replanning is insufficient for these
models under this frozen protocol.

## Manual RGB review

The representative contact sheet covers pretrained and v46 trajectories for
`id_test:6597` and `id_test:1095`, showing initial, first-out, best-score and
final frames. It confirms:

- the H1 initial frames are valid, informative official renders;
- the target pair is meaningful at the start;
- the policies rotate/translate until a task object is cropped or displaced from
  the useful joint view;
- a relation score can improve while inside-frame remains false;
- the failures are not empty renders, near-wall fabricated tasks or a termination
  mismatch.

There are no successful full rollouts to cherry-pick. The only successes in this
diagnostic are the explicitly labeled one-step local decisions above.

## Cost

- H800 SCO wall clock: 57m35s, including model hash verification, two sequential
  model loads, local-action inference, both 32-source replans and renderer work.
- Hash verification: 65.55s pretrained, 130.97s v46.
- Model load stage: 573.25s pretrained, 218.89s v46.
- Local diagnostic inference: 57.81s pretrained, 48.01s v46.
- Replan inference: 233.54s pretrained, 245.25s v46.
- Sum of per-episode replan wall time: 288.18s pretrained, 296.97s v46.

SCO allocation time is not reported as pure rendering or pure GPU compute. Both
models were loaded sequentially on the same one-H800 worker; no precision or
quantization shortcut was introduced.

## Supported conclusion and limits

Supported:

1. The model/environment interface is functional and pixel-consistent after the
   H1 fix; infrastructure and rendering do not explain the floor.
2. Local action selection is weak even one or two formal actions from the entire
   canonical success region. v46 is numerically higher than pretrained, but the
   development diagnostic is too small and exposed for a general claim.
3. Historical batching frequently accompanies visibility loss, but did not erase
   any already-achieved success.
4. More frequent visual feedback alone does not lift either model above 0/32.

Not supported:

- that v46 and pretrained are generally equivalent;
- that RL is harmful;
- that batching alone causes the failures;
- that one frozen certificate is the unique correct behavior;
- a generalization claim from these 32 R1-development sources.

The evidence is consistent with a local grounding/control weakness that then
compounds over a trajectory. It remains insufficient to cleanly separate visual
relation grounding from action-coordinate understanding without an independent
diagnostic set.

## Exact next step

Prioritize a **small frozen independent-scene local-action evaluation** before
collecting targeted training data. Reproduce the `d*=1/2` oracle-free protocol on
unseen scenes, grouped by parent source, with all optimal first actions and exact
formal distances. This is the minimum test of whether the observed local-control
failure generalizes beyond the development canary.

If that independent diagnostic reproduces the low hit rate, the next data effort
should target train-only examples of relation grounding plus formal first-action
selection, including negative alternatives that improve margin while losing
inside-frame. Only after freezing the independent evaluation should such data be
used in a training comparison. No data generation or training was started in
this turn.

## Artifacts and SHA256

Primary output:

`r1_canonical_dev_eval32_20260915/floor_effect_diagnostics_v1_20260916/`

Key hashes:

- frozen protocol: `33bb0b75...ad9c22`;
- local policy rows: `1c1d8879...5f5e70`;
- exact-distance audit: `2d4eac27...f5e70`;
- paired summary: `102b733e...2a7f9`;
- read-only baseline summary: `632405b0...6d35`;
- complete worker SHA256 ledger: `ed7248cc...67d4a`;
- injected code archive: `4a479434...93fd`;
- representative contact sheet: `9aeb9c4f...fc17`.

The machine-readable final summary is
`R1_CANONICAL_DEV_EVAL32_FLOOR_DIAGNOSTICS_SUMMARY_20260916.json`. Original
worker outputs remain read-only; the report and manually reviewed contact sheet
are separate versioned analysis artifacts.

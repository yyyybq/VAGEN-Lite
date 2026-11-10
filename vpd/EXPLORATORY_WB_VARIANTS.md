# WalkerBench limited exploration: Q3B recipe variants (EXPLORATORY)

**This is exploratory work. It is not part of the pre-registered WalkerBench pilot (`PREREG_WB_PILOT.md`).** The pilot's G1 result stands (`runs/pilot/GATE.md`, `GATE.json`). Nothing here writes under `OUT/walkerbench/runs/pilot/` and no gate is decided here: `analyze` prints numbers only. The orchestrator applies the rules below to those numbers and records each decision in `OUT/plan/DECISIONS.md`.

Authority: the user's round-3 brief (`OUT/plan/ROUND3.md` section 3, WalkerBench: "不关，限量探索"), DECISIONS G4 (the exploration rules) and G6 (the B1 single-demo rebuild), both decided by the orchestrator under the 2026-11-10 authorisation.

## 1. Purpose

Find, with Q3B only, a WalkerBench recipe whose top tier could pass the round-3 ruler criteria A2-A4 on a selection set that never touches s2test. If one is found, it is frozen as `PREREG_WB_PILOT_V2.md` and only then run with three starts on s2test. If not, within 6 variants or 30 GPU-h, a one-page memo goes to the user.

## 2. The G4 rules (intent, as decided before any variant number)

- **(a) Dev carve D60.** Continue the frozen md5(city) order of `split_s2.json` after the 6 s2val cities and take whole train-rest cities until there are at least 60 tasks. Result: Yokohama, Johannesburg, Curitiba, Sao Paulo, Chicago: 74 tasks (36 nav / 38 vis), sharing 0 routes, geofences or panos with the remaining 409. **SEL = s2val + D60 = 131 tasks (65 nav / 66 vis).**
- **(b) Pools.** The exploration training pool is train_rest minus D60 = 409 tasks. The frozen v2 s2test pilot trains on all 483 (the dev cities go back).
- **(c) s2test.** s2test and every number derived from it (including the 11-10 x483 diagnostics) take no part in recipe selection or in the choice of ruler. The code refuses any s2test task set (`variants.refuse_s2test`).
- **(d) Variant budget.** At most 6 counted variants or 30 GPU-h, Q3B only. V0 (the current recipe on 409) is the control and does not count. Counted: V1, V1T, V2, V5, V4, V6.
- **(e) Free short THOUGHT.** The user's "free short THOUGHT in training and eval" is not a separate variant: the RM expert demos contain no reasoning to train on; the zero-shot free-THOUGHT setting is covered by V1's raw-Q3B probes, and V2 tests the THOUGHT channel with its most decision-relevant content.
- **(f) Selection.** Score on SEL = nav k + vis k. A variant replaces the incumbent only if it scores at least 3 more. Ties go to the simpler recipe.
- **(g) Target reached for kind K** (nav or vis; ROUND3 A2-A4 on SEL):
  - A2: Wilson lower bound (the frozen formula, `pilot_gate.wilson_lb`) above the SEL floor (nav ODO, vis FWD; investigation values 20/65 and 9/66, so >= 28/65 and >= 15/66);
  - A3: paired exact test, same adapter with vs without images on kind K: images better and two-sided p < 0.05;
  - A4: kind rate <= 80%.
  - The floors are recomputed in-job (job A) before use; `analyze` uses the recomputed ones.
- **(h) Stop.** Write the memo at 30 GPU-h or after 6 variants. Also stop after V5 if the best nav is <= 20/65 and the vis image:blind discordance is below 4:1; in that case run V4 on the best first.

## 3. Variants

All training: `n2_train_vpd.py --start Q3B --init-adapter runs/pilot/Q3B/adapters/0W --seed 1001` (the pilot's 0W, adapter.pt sha256 prefix `5cdbfd4366a3`, read only), 640x400 demos already rendered (no re-render), full history, merged evals at 640x400, temperature 0, 8 workers, on SEL with and without images.

| variant | data (`X/<V>/data/train.jsonl`) | epochs | eval |
|---|---|---|---|
| V0 (control) | `tiers/seed1001/demos_all.jsonl` filtered to the 409, tier order, lines verbatim | 3 | `--constrained empty` |
| V1 (probes, no training) | - | - | V0 merged + `--odometer line`; raw Q3B `--constrained off --max-tokens 1024` without / with `--odometer line`; raw `--constrained empty` |
| V1T | V0 + `- Walked so far: X m` after the `- Pitch:` line of every user turn | 3 | `--constrained empty --odometer line` |
| V2 | V0 + THOUGHT `walked X of N m` (nav with a stated total) / `walked X m` (otherwise) in every reply | 3 | `--constrained short --thought-max-chars 32` |
| V5 BASE | BASE's data + per assistant message `loss_weight`: STOP 5, ROTATION or MOVE(k != 1) 3, MOVE(1) 1 | 3 | BASE's |
| V4 BASE | a byte copy of BASE's data | 10 | BASE's |

The odometer (`vpd/walkerbench/odometer.py`) is the single source for the eval line and the demo rewrite: walked = the sum of the shown distances (0.1 m, as the stock prompt prints them) of the MOVEs executed before the decision. It uses the transcript only (no lat/lng, no pano or target ids), the quantity the blind ODO baseline sums. V1 breaks train/eval prompt parity (A1) by construction: it is a probe and can never be frozen. V5 needs the trainer's per-message `loss_weight` (package wbtrain); the job's init-check refuses V5 otherwise, and adapter-ok requires `meta.loss_weights.mode == "weighted"`. V6 (recovery demos) is not implemented here; it runs only if the failure analysis shows off-path drift dominating.

## 4. Task sets and outputs

- `OUT/walkerbench/split/s2dev74/` and `OUT/walkerbench/split/s2sel/`: sha-checked copies of `split/s2train/tasks` files plus `selection.json` (rule, cities, ids, per-file sha256, `split_s2.json` sha256, leakage counts, the 409 pool's ids sha; s2sel also `members` = s2val / s2dev74). Registered in `run_vpd.TASK_SETS` in-process only (`run_vpd.py` is unchanged). `split.check` walks only s2test / s2val / s2train, so these dirs change nothing of the frozen split.
- Root X = `OUT/walkerbench/runs/explore/wbv/Q3B/` (`X/EXPLORATORY.md` marks it):
  - `X/<V>/data/{train.jsonl, manifest.json}`, `X/<V>/adapter/{adapter.pt, meta.json, log.jsonl}`, `X/<V>/evals/wbv__Q3B__<V>__s2sel__640__<mode>[__odo][__blind]/` (each with `explore.json`: exploratory, odometer mode, code sha256);
  - `X/V1/evals/` (the probes: model `V0` or `raw`);
  - `X/floors/floors_s2sel.json` (+ `_subsets.json`: SEL / s2val / s2dev74), `X/floors/logs/`;
  - `X/b1/{frozen_before.json, B1_CHECK.json}` (G6; not exploratory);
  - `X/analysis/{ANALYZE.json, REPORT_EXPLORATORY.md}`.
- `OUT/walkerbench/demos/s2train_640x400_B1_vis0578/`: the single rebuilt val demo vis_0578 (G6), made by `demos.py --stems ... --strict` then `--merge --allow-partial`. The frozen `demos/s2train_640x400/demos_all.jsonl` (sha256 `5aea1bfd...`) is checked before and after and never written.

## 5. Jobs

`vpd/jobs/wb_variants.sh VARIANT RUN [BASE]` (as root, one H800, code snapshot RUN/code):

- **Job A, vpd-wb-v0v1** (`V0`): B1 rebuild first (precheck that `baselines.rm_expert` wraps -164.8 to 195.2; frozen sha before; render; merge; verify), setup, floors on s2sel, V0 data, init-check, train, merge, V0 SEL with / without images, V1 probe V0 + odometer line, remove the merged copy, the three raw probes, analyze, expect.
- **Job B, vpd-wb-v1t** (`V1T`) and **Job C, vpd-wb-v2** (`V2`): setup (verify), data, init-check, train, merge, SEL with / without images, analyze, expect.
- **Later: vpd-wb-v5** (`V5 BASE`) and **vpd-wb-v4** (`V4 BASE`), BASE = the incumbent after Decision 1 / 2.

Every step is a `vpd_step` in `RUN/steps.jsonl`; `RUN/expect.txt` lists the outputs (`variants expect --variant V [--base B] --check`); `RUN/DONE` only when every step passed. Idempotent: data is written once (other existing content is refused), a finished adapter is reused when `variants adapter-ok` passes (exit 3 = other trainer code: the job stops), finished evals are skipped (stale or invalid ones are kept and flagged).

Before deleting a job record, check: `RUN/DONE`, every `steps.jsonl` rc 0, `variants expect --variant V [--base B] --check` exit 0 (every `results.json`, not `results_invalid.json`), for job A `X/b1/B1_CHECK.json` `"ok": true`, then `sco status` once more.

## 6. How to read `analyze`

`python -m vpd.walkerbench.variants analyze` (login node, PY_EVAL, 1 core) writes `X/analysis/ANALYZE.json` and `REPORT_EXPLORATORY.md`:

- `floors.subsets`: STOP0 / FWD / ODO / RM per kind on SEL, s2val, s2dev74 (in-job recompute); `floors.g4_floor` = SEL nav ODO, vis FWD.
- `evals[<owner>/<name>]`: per kind success with Wilson 95% and oracle on SEL, s2val and s2dev74 separately; `behaviour_SEL` = learn483's stop / oracle / turn statistics (ended in STOP, cap hits, P(STOP | <= 15 m), turn-node accuracy, the own-path "stop at walked >= stated" counterfactual, ...); for short-THOUGHT evals `thought` = THOUGHT format rate and odometer error against the true walked distance.
- `variants[<V>]`: `selection_score_nav_k_plus_vis_k` (G4(f)) and `g4g[kind]` with A2 (`wilson_lb`, `floor`, `bar_k`, `A2_lb_above_floor`), A3 (`A3_paired` image-only : blind-only and the two-sided exact p, `A3_images_better_p_lt_0.05`), A4 (`A4_rate_le_0.80`) and `all_three`; the data rule and the adapter's losses.
- `g4h_inputs`: the best nav k and each variant's vis image:blind discordance (the G4(h) futility inputs).

These are numbers, not verdicts. Every number that goes into a report must be recomputed by an independent script from the raw files (ROUND3 section 6).

## 7. Caveats to carry into the memo or the v2 prereg

- SEL is small (Wilson half-width about +-12 points) and selection among several variants carries a winner's curse; the +3 adoption margin and the untouched s2test gate are the mitigations.
- The vis FWD floor on SEL (about 14%) is far above s2test's (3/94): passing on SEL does not carry over one-to-one.
- s2val serves both Q3B recipe selection here and, later, confirmatory LR selection; the v2 prereg must disclose this.
- The pilot's per-start numbers (e.g. vis passing at T200 for all starts) must not be used to choose a vis-only ruler; v2 needs fresh runs.
- Training on 409 instead of 483 tasks makes the variant numbers slightly pessimistic against the 483-trained v2 pilot.
- WalkerBench loads only `perception_whitelist.json` as its geofence file, so nav / vis moves are not fenced (upstream behaviour, the same in demos and evals).

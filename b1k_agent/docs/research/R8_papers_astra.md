# R8: Checking the reference papers and the "Astra" model

Date of research: 2026-10-22. Researcher: subagent R8 (read-only, no simulator).

## 0. How this was checked

**Where the evidence came from:**
- **arxiv.org, huggingface.co, alphaxiv.org and project pages.** None of these could be fetched. curl through the proxy returned "CONNECT tunnel failed, response 403". WebFetch to alphaxiv.org returned `getaddrinfo ENOTFOUND`.
- **WebSearch.** This worked. It returns snippets and model-written summaries only. Every claim drawn from it is tagged **[search]** with its URL.
- **GitHub.** `git clone --depth 1` over https worked. api.github.com and github.com issue pages returned 403. Every paper repo I could find was cloned read-only into the scratch area below. Claims from those clones are tagged **[repo]** with file:line.
  - Scratch area: `/tmp/claude-0/-home-user-VAGEN-Lite/9977f130-b514-5e2e-ab6e-7d7795066892/scratchpad/research/r8repos/`. Short names used below: `RR/`, with `RR/<owner>_<repo>/...`.
- **Cracking Mobile Manipulation PDF.** The authors' repo ships the paper PDF. I extracted its text with `pdftotext -layout` to `/tmp/claude-0/-home-user-VAGEN-Lite/9977f130-b514-5e2e-ab6e-7d7795066892/scratchpad/research/r8txt/cracking.txt`. Line numbers for it are cited as **[CMM-pdf:L<n>]**.
- **Official rules.** These are cited from the local B1K clone at HEAD (v3.9.3-post2, a8247a8) as **[B1K:docs/...:line]**.

**Status labels:**
- **verified_code**: I read it in source code.
- **stated_docs**: stated in a README, docs file or the paper PDF in an author repo.
- **search_snippet**: only seen in search results.
- **inferred**: my own reasoning from the above.
- **NOT FOUND / UNVERIFIED**: I could not find it.

**Repos cloned (commit and date):**

| Short name | Repo | Commit | Commit date |
|---|---|---|---|
| MobiAgent | github.com/kaiknower/MobiAgent | ffce2443abf7 | 2026-10-11 |
| Cracking | github.com/PKU-VaLuE-Lab/Cracking-Mobile-Manipulation | 2702ab119840 | 2026-10-21 (README, PDF, images only) |
| RAX main | github.com/robot-agent-exploration/rax (main) | bf0439f9a61c | 2026-10-21 |
| RAX BEHAVIOR | same repo, branch `robodojo-behavior/native` | 1449974392e1 | 2026-10-21 |
| GalaxeaVLA | github.com/OpenGalaxea/GalaxeaVLA | 89f2322b4ad0 | 2026-08-13 |
| SERF-VLA | github.com/ExistentialRobotics/SERF-VLA | ea27b7aa753c | 2026-07-02 |
| BinceQu RoboHarness | github.com/BinceQu/RoboHarness | dc71e141d990 | 2026-10-20 |

The BinceQu RoboHarness repo belongs to a **different** paper (2610.14018), not 2607.18060. See §4.

---

## 1. Summary table

| arXiv ID | Exists? | Exact title | First author / org | Date | Code / weights | BEHAVIOR-1K results? |
|---|---|---|---|---|---|---|
| 2610.03476 | YES | MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation | Chenzhi Liu; HKU, Astribot, SUSTech | early Oct 2026 | Code: yes. Weights: RoboCasa checkpoint only, gated on HF. No BEHAVIOR weights. | Yes: 4 tasks, 65.0% vs 42.5% for pi0.5-TA [search only] |
| 2610.12089 | YES | ManiUnit: A Manipulation Skill Dataset and Benchmark for Long-Horizon Tasks | Guoting Wei + 12; org NOT FOUND | 2026-10-08 | Code NOT FOUND | Built entirely on BEHAVIOR-1K (50 activities) |
| 2606.04968 | YES (titled differently) | Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement. The method is named ForesightFlow. | Yunpeng Mei + 14; org NOT FOUND | 2026-06-03 | Code NOT FOUND | Yes: 5 BEHAVIOR-1K tasks, numbers not seen |
| 2607.18060 | YES | RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning | Jinbang Huang + 8; Huawei Noah's Ark, UBC, UofT, McGill [search, secondary] | 2026-07-20 (v2 07-28) | Official code NOT FOUND | Yes: 6 tasks, SR 0.43, progress 0.66 [search only] |
| 2610.19600 | YES | Cracking Mobile Manipulation: From Failure Diagnosis to Agentic Robot System | Zixuan Wang; PKU, Alibaba DAMO, CASIA, Tsinghua, UTS, Liverpool | 2026-10-14 | Repo has the PDF; code "Coming soon" | Yes: 5 tasks from the 2026 Challenge, 37/100 vs 14/100 (Harness VLA) vs 7/100 (pi0.5) [read in PDF] |
| 2609.20791 | YES | StageGuard: Learning Stage Transitions for Long-Horizon Robot Tasks via Agentic Distillation | Jinbang Huang + 8; Huawei Noah's Ark | 2026-09-17 | Code NOT FOUND | Yes: evaluated on BEHAVIOR-1K; ">90%" figure is UNVERIFIED |
| 2606.12956 | YES | SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation | Sunghwan Kim; Atanasov group (ExistentialRobotics) | 2026-06-11 | Code and weights released | Yes: 3 tasks, progress 44.0% to 58.7% [search]. Its eval relies on **privileged inputs** [code] |
| 2609.35249 | YES (ID plausible) | Spatial Grafting: Grounding 3D Features for Flow-Matching Robot Policies | Dingsheng Liu + 7; org NOT FOUND | 2026-09-28 | Code NOT FOUND | Yes: 6 tasks, beats the 2025 winner on 5 of 6 [search] |
| 2610.22496 | YES | Scaling Online Exploration for Robot Agents (RAX) | Runhan Huang, Xiangcheng Zhang; Harvard, NVIDIA, UCB, MIT, Stanford | 2026-10-15 | Code released (Apache-2.0); BEHAVIOR branch exists | Yes: 6 tasks, Q 46.5 vs 8.2 [search] |
| 2608.11739 | YES | G0.5: One Autoregressive Stream for Robot Reasoning and Action (Galaxea) | Galaxea; author list NOT FOUND | 2026-08-12 | Code yes. Open weights under a **non-commercial** license. | README claims 0.3136 on BEHAVIOR-1K; no BEHAVIOR checkpoint or entrypoint released |

**"Astra":** this is OpenAI **GPT-6 Astra**.
- Announced 2026-09-03 at https://openai.com/index/gpt-6-astra/
- API model ID `gpt-6-astra`, per the official model page https://developers.openai.com/api/docs/models/gpt-6-astra
- Takes text and image input and returns text. Structured outputs are listed as supported. [search]
- Rollout started with a limited set of organizations. Whether it is now generally available is unclear.
- I found no dated snapshot ID. Do not guess one.

There are two other "Astra"s. Google's **Project Astra** is a research prototype that powers Gemini Live API features. It is not a model ID. ByteDance **Astra** (arXiv 2506.06205) is a robot navigation system, not an API.

---

## 2. MobiAgent (arXiv 2610.03476): skill data curation, execution check and replanning

### Identity
- **Title and authors:**
  - The README title is "MobiAgent: Dual-Loop Recursive Policy Self-Improvement for Long-Horizon Mobile Manipulation" [repo: RR/kaiknower_MobiAgent/README.md:8].
  - Authors: Chenzhi Liu, Zhang Yue, Jiehong Lin (project leader), Jianan Wang, Bo Wang, Zhongrui Wang, Xiaojuan Qi.
  - Affiliations: 1 The University of Hong Kong, 2 Astribot, 3 Southern University of Science and Technology [repo: README.md:10-14].
  - The bibtex carries `eprint={2610.03476}` and `primaryClass={cs.RO}` [repo: README.md:181-189].
- **Venue and date:** accepted at CoRL 2026 and posted in early October 2026 [search: https://www.alphaxiv.org/abs/2610.03476, https://github.com/luohongk/Embodied-AI-Daily/issues/1028]. This is unverified beyond the snippets.
- **Name clash:** this is NOT the 2025 SJTU IPADS "MobiAgent" smartphone GUI agent (arXiv 2509.00531) [search: https://arxiv.org/html/2509.00531v1].

### Code and weights
- **Code** is public on GitHub at commit ffce2443 (2026-10-11).
- **Weights:**
  - The only released checkpoint is "the RoboCasa joint-training checkpoint at step 25,000". It sits on HF `Liukaikai/MobiAgent`, and access is granted only after manual approval [repo: README.md:31-36].
  - For BEHAVIOR: "The default base checkpoint URI is the upstream OpenPI pi0.5 checkpoint; no trained weights are included" [repo: docs/training.md:22-24].
  - So **there are no BEHAVIOR-trained MobiAgent weights** (verified in docs).
- **License:** there is no root LICENSE file. NOTICE.md says `policy/openpi/` keeps OpenPI's Apache-2.0 license [repo: NOTICE.md:1-8]. The license for the rest of the repo is NOT FOUND.

### Method relevant to "skill data curation + execution check/replanning" (verified in code)

**Inner loop: online execution.** Code is in `src/mobiagent/execution/orchestrator.py`. Docstring at lines 1-20:

```
Flow per tick (one tick == one "attempt"):
  1. If there is no current subtask, ask the planner for ONE next subtask
     (passing completed history + last failure context).
  2. Route the subtask to the policy server matching its `stage_hint`,
     request CLAW_CHUNKS_PER_ATTEMPT (default 10) chunks back-to-back, ...
  3. Ask the judge for a verdict on the final obs, also showing it the final
     frames of the previous two attempts (attempt-resolution comparison).
  4. complete -> mark subtask done ... incomplete -> retry same subtask (no planner call) until attempts == max_retries.
     retries done -> clear current, stash failure context for the planner's next call.
  5. The orchestrator owns termination, NOT the planner. Stop when ANY of:
       - env.is_success() returns True (BDDL goal predicate) ...
```

- **Planner output.** The planner emits one JSON object per call. Its fields are `id`, `prompt`, `stage_hint` (the exact skill name), `success_check`, `max_retries`, `target_object_name`, `failure_cues`, `rationale` and `plan_sketch`. The prompt also says: "Never treat an attempted action as a completed action" and "Do not invent object locations, fixed task scripts, hidden state, or synthetic success evidence" [repo: src/mobiagent/execution/PROMPTS.md:3-24].
- **Judge input.** The judge (`judge_vlm.py`) gets head and both wrist images from three moments: "two attempts ago", "previous attempt" and "current". It also gets `success_check`, `failure_cues` and proprioception text.
- **Judge output.** It returns `verdict ∈ {complete, incomplete, error}` and `recommended_followup ∈ {next, retry, replan_keep_pose}`. Any unknown verdict falls back to `incomplete`.
- **Judge prompt:** "Return complete only when visible evidence supports the requested outcome ... Do not infer success from elapsed time, a task name, or an earlier verdict" [repo: src/mobiagent/execution/judge_vlm.py:10-60].
- **When it replans.** On `error`, or a followup of `replan_keep_pose`/`replan_plan_deviated`, the orchestrator replans early. Otherwise it replans once `attempts > st.max_retries` [repo: orchestrator.py:339-371].
- **Gripper state for the judge.** Robot info is decoded from the 256-D R1Pro proprio. Left gripper is `a[193:195]`, right is `a[232:234]` [repo: orchestrator.py:274-288].
- **Skill vocabulary on BEHAVIOR:** `move_to, pick_up_from, place_in, place_on, open, close`, in that order [repo: docs/deployment.md:19-21; policy/openpi/src/openpi/training/config.py:340].
- **LLM requirement.** `OPENAI_MODEL` must be "A model supporting image inputs and JSON output". Each stage can be overridden with `MOBIAGENT_PLANNER_MODEL`, `MOBIAGENT_JUDGE_MODEL`, `MOBIAGENT_FRAME_MODEL` or `MOBIAGENT_NAMING_MODEL` [repo: README.md:150-161].

**Outer loop: offline skill discovery and curation.** Code is in `src/mobiagent/discovery/behavior/`.
- **Video annotation.** A multimodal video model annotates per-demo `skill_timeline` entries (`segment_id, skill_description, start_time_sec, end_time_sec`). The defaults are `gemini-3.1-pro-preview` and `qwen3.6-plus`. Cluster naming falls back to `OPENAI_MODEL` [repo: pipeline.py:31-35; skills/mobiagent-discovery/SKILL.md:66-79].
- **Frame filter.** Near-static frames are dropped by thresholding frame and state deltas [repo: frame_filter.py:10-30].
- **Clustering.** Skill descriptions are clustered by normalised string match, then named by an LLM [repo: clustering.py:17-40].
- **Segment export.** Segments are `[start_idx_30hz, end_idx_30hz)`. Short action chunks repeat the final frame inside the same segment and "never extend into the next skill" [repo: docs/training.md:37-39].
- **Training recipe.** Six routed action experts (`Pi0SixHeadConfig`) sit on a pi0.5 base. The PaliGemma backbone is frozen. Sampling is per-expert (`per_expert=4`), with per-expert normalization and `skill_prompt_style="skill_only"` [repo: config.py:314-337; docs/training.md:33-35].

**My summary (inferred from the code above):** MobiAgent is the closest public template for our planner–executor–checker loop.
- A VLM planner emits one subtask at a time with an explicit `success_check`.
- That subtask runs on a skill-routed pi0.5 for N chunks.
- A VLM critic compares frames before and after, then chooses next, retry or replan.
- The orchestrator, not the LLM, owns retry budgets and termination.

### BEHAVIOR-1K results
- **Headline [search only, not verified against the paper]:**
  - "65.0% mean success rate on BEHAVIOR-1K, 22.5 points above pi0.5-TA (42.5%)", over **four** BEHAVIOR-1K tasks [search: https://www.alphaxiv.org/fr/abs/2610.03476, https://note.com/yasuhitoo/n/nf49e69164f75].
  - RoboCasa: 7.50% to 27.50%. Astribot S1: 32.5% to 57.5% after data recycling [search: https://www.alphaxiv.org/abs/2610.03476].
  - Which four tasks were used is NOT FOUND.
- **Protocol caveat (verified_code + inferred).** The BEHAVIOR adapter is MobiAgent's own `OmniGibsonEnv`. It is built from "BEHAVIOR gello configuration helpers" [repo: src/mobiagent/execution/env_omni.py:1-8, 442-484].
  - It forces 224x224 RGB: "Force 224×224 RGB (training-data resolution). gello defaults to 1080." [env_omni.py:475].
  - The official evaluator's challenge wrapper is `RGBDFullResWrapper`: 720x720 head camera and 480x480 wrist cameras [B1K:docs/challenge/evaluation.md:140].
  - So MobiAgent's BEHAVIOR numbers are **not official-evaluator numbers** (inferred).

---

## 3. ManiUnit (arXiv 2610.12089): training on skill segments

- **Exists.** "ManiUnit: A Manipulation Skill Dataset and Benchmark for Long-Horizon Tasks". Submitted Oct 8, 2026. Guoting Wei and 12 co-authors [search: https://arxiv.org/abs/2610.12089, https://arxiv.org/abs/2610.12089v1].
- **Affiliation:** NOT FOUND in snippets.
- **Code and data release:** NOT FOUND. There is no GitHub or dataset link in any snippet [search: https://arxiv.org/html/2610.12089v1]. I searched "ManiUnit BEHAVIOR-1K skill dataset github code release Guoting Wei" in extended mode.

**Method (search snippets):**
- **How it is built.** ManiUnit cuts long-horizon BEHAVIOR-1K demonstrations into single manipulation operations, using the skill annotations that ship with the BEHAVIOR-1K demos. Each annotation gives an action label, the objects, and start and end frames. Every segment is paired with an explicit **subtask instruction**.
- **Scale.** 50 BEHAVIOR-1K activities, giving **137,899 segments, 21 skill types and 417 subtasks**. The benchmark has **1,260 test instances**.
- **How it scores a skill.** The benchmark **restores intermediate simulator states** and defines **local success conditions**, so each skill is scored without running the earlier stages. It also perturbs the starting **base position** or **joint configuration**.
- **Findings:**
  - Joint perturbations cut success by about 56% compared with demonstrated starting states.
  - On two long-horizon activities, a skill policy trained on segments reaches **78.7%** local success. A task policy trained on full demos reaches **49.3%**.
  - Coordinating the two policies through a planner raises full-task success from **4.0% to 18.0%**.
  - [search: https://arxiv.org/html/2610.12089v1, https://arxiv.org/abs/2610.12089]
- **Reports BEHAVIOR-1K results:** YES. It is a BEHAVIOR-1K derived benchmark. Whether it ran under the official 2026 evaluator is unknown.
- **Related but different paper:** MUCO, arXiv 2610.19913, "A Mani-Unit-Structured Manipulation Framework ..." [search: https://arxiv.org/html/2610.19913].

**Note for us (inferred):** start-state sensitivity of about 56% under joint perturbation suggests the planner or approach module should hand the skill executor near-demonstration start configurations. MobiAgent's "replan_keep_pose" and Cracking's base adjustment are possible mechanisms for this.

---

## 4. ForesightFlow (arXiv 2606.04968): scoring action and stage success from mixed-quality data

- **Exists, under a different title.** "Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement". The method is named **ForesightFlow**. Submitted 3 June 2026. Authors are Yunpeng Mei, Jiakai He, Hongjie Cao, Chenyu Wang, ..., Gang Wang (15 total) [search: https://arxiv.org/abs/2606.04968, https://arxiv.org/html/2606.04968v1, https://arxiv.org/pdf/2606.04968].
- **Affiliation:** NOT FOUND.
- **Code:** NOT FOUND. Warning: the GitHub org `github.com/ForesightFlow` is an **unrelated** prediction-market project by Maksym Nechepurenko (arXiv 2605.00493) [search: https://github.com/ForesightFlow, https://arxiv.org/abs/2605.00493].

**Method (search snippets):**
- **One flow does both jobs.** A self-guided flow policy adds a learned **success-potential trajectory** to every generated action chunk. The same flow proposes candidates and scores them, which gives best-of-K inference "without an external critic".
- **Decoupled training.** Action velocities get clipped exponentiated advantage weights. Potential velocities are trained **uniformly on the full mixed-quality dataset**, because weighting them too "would hide corrective gradients from overconfident failures".
- **Cheap scoring at inference.** A one-step estimator takes a single Euler step from the noise endpoint, using the CFM conditional-mean identity. It then averages the success potential over the horizon (Eq. 14).
- **Results:** 5 BEHAVIOR-1K simulation tasks and 5 real bimanual tasks. It improves over imitation baselines and matches the strongest separate-critic baseline in simulation. It also uses 38% less training compute.
- **Ablations** show that decoupling prevents "value hallucination".
- [search: https://arxiv.org/html/2606.04968v1, https://awesomepapers.io/robotics/papers/2606.04968, https://arxiv.org/html/2607.27782]
- **BEHAVIOR-1K results:** YES (5 tasks). Per-task numbers and task names were NOT FOUND in snippets.

**Note for us (inferred):** this is a policy-internal success scorer, which needs retraining of the VLA's flow head. It is not a drop-in stage checker for an off-the-shelf pi0.5.

---

## 5. RoboHarness (arXiv 2607.18060): capability descriptions, experience retrieval, Memory Bridge

- **Exists.** "RoboHarness: Memory-Driven Orchestration of Heterogeneous Robot Policies for Long-Horizon Planning". Jinbang Huang and 8 co-authors. Submitted July 20, 2026, revised July 28 [search: https://arxiv.org/abs/2607.18060, https://huggingface.co/papers/2607.18060].
- **Affiliations:** Huawei Noah's Ark Lab, UBC, U of Toronto, McGill. This comes from a **secondary** news digest [search: https://note.com/yasuhitoo/n/n9cad6a51f2af].
- **Project page:** robo-harness.com, named in a secondary source. It was unreachable (proxy 403).
- **Official code:** NOT FOUND [search: extended query "RoboHarness" "Memory-Driven Orchestration..." github].

**Method (search snippets):**
- **Skills.** "encapsulates independently developed robot control systems as reusable agentic skills", instantiated with VLAs, RL policies and TAMP.
- **Memory skills.** They keep a **policy-specific episodic memory** and an **execution log**. The log is used for "capability assessment, failure diagnosis, and evolution".
- **Memory Bridge.** It "retrieves execution trajectories associated with the next policy, estimates its in-distribution state region, and guides the robot toward that region without joint policy retraining". The steps are: retrieve top-K anchors, fit a local progress estimator, then generate a bridge trajectory.
- **Scale.** "Three public benchmarks" in the abstract but "five" in the HTML (a discrepancy), plus 500 custom tasks across 10 classes and 135 real-robot trials.
- **BEHAVIOR result.** One snippet reports six BEHAVIOR-1K long-horizon mobile manipulation tasks, with **SR 0.43 and progress 0.66** for RoboHarness.
- **Criticism.** Pith, an AI-generated review site, says Table 2 "may measure in-task adaptation rather than zero-shot composition".
- [search: https://arxiv.org/html/2607.18060, https://pith.science/paper/2607.18060, https://www.besthub.dev/articles/why-one-model-can-t-win-roboharness-orchestrates-heterogeneous-robot-policies-37fdea2494d8]
- **"Capability descriptions" specifically:** UNVERIFIED. No snippet uses that phrase. The closest is the execution log being used for "capability assessment".

### Name collisions (important)
- **arXiv 2610.14018, "RoboHarness: A Simple Harness Outperforms VLA and World Action Models".** Code at `github.com/BinceQu/RoboHarness` [search: https://arxiv.org/abs/2610.14018]. From the clone:
  - It is training-free: an LLM agent clicks visual keypoints, which are tracked by optical flow and depth back-projection and turned into geometric constraints [repo: RR/BinceQu_RoboHarness/README.md:18-27].
  - BEHAVIOR results use "Claude Code 2.1.259 with `Qwen3.8-Flash-Next-FP8`" [README.md:25-26].
  - BEHAVIOR is pinned to **v3.9.1** [README.md:33].
  - Its protocol is the **2025** challenge evaluator: 5 instances per task (IDs 301, 304, 306, 308, 310), a step cap of **2x** mean human demo length, and human-written task-specific prompts [README.md:203-221].
- **arXiv 2603.24060**, "RoboHarness: A Memory-Augmented Policy Harness for VLA Model Robustness via In-Context Adaptation" [search: https://arxiv.org/pdf/2603.24060].
- **arXiv 2609.29389**, "Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation" [search: https://arxiv.org/pdf/2609.29389].

---

## 6. Cracking Mobile Manipulation (arXiv 2610.19600): approach and recovery from failure analysis

The paper PDF is in the authors' repo, so these facts are read from the paper itself.

### Identity
- **Title and authors.** "Cracking Mobile Manipulation: From Failure Diagnosis to Agentic Robot System". Zixuan Wang, Jia-Ning Li, Yinzhou Tang, Yanzhao Yu, Biao Wu, Meng Fang, Xin Li, Yiwu Zhong.
- **Affiliations.** PKU School of Intelligence Science and Technology; State Key Laboratory of General Artificial Intelligence, PKU; Alibaba DAMO Academy; CASIA; Tsinghua; UTS; Univ. of Liverpool [CMM-pdf:L21-29].
- **arXiv ID** 2610.19600, submitted 14 Oct 2026 [search: https://arxiv.org/abs/2610.19600].
- **Code:** "🚧 Coming soon". Every listed component is unchecked: subtask VLA training, evaluation, checkpoint, agentic system and system evaluation [repo: RR/PKU-VaLuE-Lab_Cracking-Mobile-Manipulation/README.md:20-29].

### Failure analysis

The baseline is an agentic system adapted from Harness VLA (arXiv 2607.08448). Its planner is Kimi K3, and one subtask VLA handles both navigation and manipulation. Each VLA call runs 350 control steps [CMM-pdf:L1067-1074].

Primary failure attribution over 81 failed RoboCasa365 episodes [CMM-pdf:L273]:

| Failure class | Share | Main subtypes | Counts |
|---|---|---|---|
| Task planning | 21.0% | target/receptacle assignment; subtask order leading to an unrecoverable state | 9; 8 |
| Navigation and base positioning | 25.9% | ineffective approach or poor base pose; collision or stall; orientation | 11; 7; 3 |
| Local manipulation | 53.1% | incorrect placement; grasp misalignment or empty grasp; slippage | 20; 10; 6 |

Simulator states are used "only for diagnosis and are not available to the online agent" [CMM-pdf:L235].

### Interventions (the method for our "approach/recovery" module)
- **Skill cards from demonstrations.** Gemini 3.5 Flash retrieves a training-split demo by matching annotations to the instruction. It summarises the demo into a skill card holding the goal, ordered subtasks, keyframes and completion criteria. Cards are fixed before evaluation [CMM-pdf:L351-358, D.1].
- **Adaptive visual memory.** The planner calls `recall_memory(q)`. Qwen3.8-27B ranks up to 256 past frames at 448x448 and returns the **top k=16** plus a trajectory summary. Only past observations are used [CMM-pdf:L360-366; L1230-1240].
- **Accessibility check (`check_stance`).**
  - Grounding DINO plus SAM find the target mask. Depth pixels are back-projected into the robot frame.
  - The stance is classed **reachable if d ≤ 0.75 m**, **marginal if 0.75 < d ≤ 1.00 m**, otherwise out of reach. Thresholds come from successful training trajectories [CMM-pdf:L374-381; L1250-1254].
- **`approach_xy`.** For a marginal stance: a local base XY shift of **at most 0.25 m**, heading fixed [CMM-pdf:L1252].
- **`goto_stance`.** For out of reach: picks a new stance, then runs A* on a 2D occupancy grid built from current depth. Grid is **0.05 m**, obstacles inflated by **0.10 m**, followed by a PD controller [CMM-pdf:L384-389; L1255-1257].
- **End-effector adjustment.** Target position is the median of the grounded 3D points. Approach direction comes from the local surface normal. Each correction is bounded to **0.30 m**, and the VLA itself stays fixed [CMM-pdf:L391-399; L1258-1260].
- **Recovery prompt:** "After an empty grasp or an off-target placement, inspect what happened before retrying. A closed gripper alone does not confirm a successful grasp. Localize the target from the current RGB-D observations rather than using fixed coordinates." [CMM-pdf:L1294-1297].

### Results
- **RoboCasa365:** 19% to 47% on dev seeds; 18% to 49% on held-out seeds [CMM-pdf:L41-43].
- **BEHAVIOR-1K, Appendix I, Table 14.** Five **2026 Challenge** tasks, 20 matched episodes each [CMM-pdf:L1495-1507]:

| Method | Pick Up Trash | Make Rose Centerpieces | Composting Waste | Installing Smoke Detectors | Make Microwave Popcorn | Overall |
|---|---|---|---|---|---|---|
| pi0.5 | 1 | 2 | 1 | 1 | 2 | 7/100 |
| Harness VLA | 2 | 3 | 3 | 2 | 4 | 14/100 |
| Ours | 7 | 9 | 7 | 6 | 8 | **37/100** |

- **BEHAVIOR training detail.** Low-level policies use "only the training data provided by the BEHAVIOR-1K challenge". The subtask VLA trains on trajectories segmented at the provided subtask boundaries. The pi0.5 baseline trains on full trajectories. The planner is Kimi K3 [CMM-pdf:L1509-1515].
- **What the paper does not say for BEHAVIOR:** the BEHAVIOR-1K version, the evaluator used, the instance IDs, and whether the metric is success or Q-score. For RoboCasa they say "the official RoboCasa365 evaluator" [CMM-pdf:L418-419]; no equivalent statement exists for BEHAVIOR.

---

## 7. StageGuard (arXiv 2609.20791): teacher-labelled stage judgements distilled into a light checker

- **Exists.** "StageGuard: Learning Stage Transitions for Long-Horizon Robot Tasks via Agentic Distillation". Jinbang Huang, Yuanzhao Hu, Zhiyuan Li and others (9 authors). Huawei Noah's Ark Lab. Submitted 2026-09-17 [search: https://arxiv.org/abs/2609.20791, https://arxiv.org/html/2609.20791, https://github.com/set-path/DailyArXiv/issues/354].
- **Code:** NOT FOUND [search: extended query on title + "code github student VLM"].
- **Name clash:** a different "StageGuard" on sleep staging is arXiv 2607.23284 [search: https://arxiv.org/pdf/2607.23284].

**Method (search snippets):**
- **Problem.** Completion checkers are hard to get in real execution. Large VLMs' decision boundaries "don't inherently match task completion criteria". Cloud reasoning adds latency.
- **Teacher.** A teacher VLM inside an agentic workflow writes structured reasoning traces. These "explicitly link observations to ground-truth transition decisions in demonstration trajectories".
- **Student.** A lightweight student VLM is trained by SFT, with self-explanation, and serves as the **execution monitor** for hierarchical control.
- **Evaluation:** LIBERO, BEHAVIOR-1K and a real robot. Reported gains are better transition prediction, low latency and better closed-loop success.
- [search: https://arxiv.org/html/2609.20791, https://pith.science/paper/2609.20791]
- **BEHAVIOR-1K results:** YES, it is evaluated there. The "over 90% transition completion on BEHAVIOR-1K, well above GPT-5.6 Luna" figure comes only from Pith, an AI-generated review site. Status: **UNVERIFIED**. Pith also flags "teacher-as-judge circularity".

**Note for us (inferred):** BEHAVIOR-1K demos ship skill/stage annotations (ManiUnit and Cracking both use them). These give the "ground-truth transition decisions" needed to reproduce StageGuard-style distillation for a fast checker.

---

## 8. SERF (arXiv 2606.12956): spatial memory and prior-map assumptions

### Identity, code and weights
- **Exists.** "SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation". Sunghwan Kim, Byeonghyun Pak, Kehan Long, Yulun Tian, Nikolay Atanasov. Submitted 11 June 2026 [search: https://arxiv.org/abs/2606.12956; repo: RR/ExistentialRobotics_SERF-VLA/README.md:3-7, 229-236].
- **Venue:** the project page lists an ICRA 2026 workshop Best Paper; a citing paper lists CoRL 2026 [search: https://existentialrobotics.org/serf/]. Unverified.
- **Code:**
  - `ExistentialRobotics/SERF-VLA` holds BEHAVIOR-1K policy training and evaluation [repo: README.md:9-13].
  - The mapping component is a separate repo, `SERF-mapping` [README.md:15-17].
- **Weights:**
  - HF `byeonghyunpak/SERF-VLA`, with pi0.5 baselines and SERF-VLA checkpoints for task-0021, task-0022 and task-0026 [repo: README.md:87-143].
  - Initialization is `IliaLarchenko/behavior_50t_checkpoint`, from the 2025 first-place solution [repo: README.md:60-85].
- **Task names.** The three IDs map to collecting_childrens_toys, putting_shoes_on_rack and assembling_gift_baskets. This uses the task-ID table in MobiAgent's env_omni.py:52,53,57 (verified_code, cross-repo).

### Method (search) and BEHAVIOR results
- **The map.** Environment and the articulated robot body are stored as neural points in a shared latent space. The map is updated online from egocentric observations and proprioception, and fed into a VLA.
- **Results on 3 BEHAVIOR-1K tasks:** average task progress **44.0% to 58.7%** over image-only baselines; **28.0% to 51.0%** in unvisited regions [search: https://arxiv.org/html/2606.12956v1].

### Prior-map and privileged-input assumptions (verified in code; important)
1. **Per-instance prebuilt map loaded at evaluation.**
   - `_map_dataset_path` resolves `<root>/<task>/eval/scene_{instance_id}.hdf5` [repo: src/serf_b1k/learning/eval_4d_env_feat_serf_b1k.py:363-371; src/serf_b1k/utils/neural_point_paths.py:27-33].
   - The tracker initialises from `initial_points`, `initial_instance_ids` and `initial_features` in that file [repo: src/serf_b1k/learning/online_3d_tracker.py:438-451].
   - So the policy starts with a **prior 3D map of the specific evaluation instance**.
2. **Ground-truth instance segmentation.**
   - Code: `cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb", "depth_linear", "seg_instance"]`, commented "Depth and instance segmentation are consumed by the online tracker" [eval_4d_env_feat_serf_b1k.py:431-434].
   - The `{og_instance_id: object_name}` mapping is read from `info["obs_info"]...["seg_instance"]` [eval_4d_env_feat_serf_b1k.py:688-713].
3. **World camera pose.** `_get_camera_poses_c2w` returns "the absolute camera-to-world matrix". It uses `camera.get_position_orientation()` or the camera view transform [eval_4d_env_feat_serf_b1k.py:638-669].
4. **Modified task goal.** For task 21 the SERF evaluation **patches the BDDL goal** so that all dice, teddy bears, board games and train sets must be inside the bookcase [repo: README.md:195-208; scripts/setup/patch_collecting_childrens_toys_bddl.py:1-40].
5. **Uses 2025 challenge data** (`datasets/2025-BEHAVIOR-1K-CHALLENGE`) and `2025-challenge-task-instances` metadata [README.md:42-54; eval_4d_env_feat_serf_b1k.py:398-405].

**These conflict with the 2026 rules (inferred, by comparison).**
- The 2026 rules forbid "ground-truth segmentation, object state, target object pose, full-scene point cloud, robot global pose, or other simulator-only privileged information during evaluation" [B1K:docs/challenge/evaluation.md:21-24].
- Allowed inputs are "RGB + depth + proprioception" [B1K:docs/challenge/evaluation.md:14-16].
- So SERF as released **cannot be used as-is** in the official 2026 evaluation, and its numbers are not comparable to official Q-scores.
- A challenge-legal spatial memory would have to be built online. Candidate inputs: RGB-D, camera intrinsics, camera poses relative to the robot base, and odometry integrated from proprioceptive base velocity. That last approach is what RAX does [repo: RR/rax_rb_native/docs/BEHAVIOR_BACKEND.md:137-140].
- SERF's README also warns that "A single episode can take several hours, and a full 20-episode task evaluation may take several days" [README.md:176-178].

---

## 9. Spatial Grafting (arXiv 2609.35249): spatial features (not spatial memory)

### Is the ID plausible?
Yes, and the paper exists.
- The arXiv abstract pages `arxiv.org/abs/2609.35249` and `.../2609.35249v1` both appear in search results with the exact title [search: https://arxiv.org/abs/2609.35249, https://arxiv.org/html/2609.35249v1].
- Other five-digit September 2026 IDs above 35249 also appear in searches:
  - 2609.37583 (RoboHarn-Evo) [search: https://arxiv.org/html/2609.37583v1]
  - 2609.35690 (Agent Priors-guided Policy Learning) [search: https://arxiv.org/html/2609.35690]
  - 2609.34276 (NavHarness) [search: https://arxiv.org/html/2609.34276]
- A secondary source says monthly submissions reached **40,363 in September 2026** [search: techjacksolutions.com via "arXiv monthly submissions September 2026"]. This is unverified against arxiv.org/stats.
- For comparison, the official arXiv blog record for October 2024 was 24,226 [search: https://blog.arxiv.org/2024/11/04/arxiv-sets-new-record-for-monthly-submissions-again].
- A sequence number of 35249 is therefore consistent with 2026 volumes.

### Identity and method (search)
- **Title and authors.** "Spatial Grafting: Grounding 3D Features for Flow-Matching Robot Policies". Dingsheng Liu, Yangzheng Wu, Mahboubeh Asadi, Zhiyuan Li, Jinbang Huang, Yixin Xiao, Tongtong Cao, Yingxue Zhang. Submitted 28 Sep 2026 [search: https://arxiv.org/abs/2609.35249].
- **Affiliation:** NOT FOUND. Jinbang Huang and Zhiyuan Li overlap with StageGuard and RoboHarness, which are Huawei Noah's Ark (inferred and unverified).
- **Code:** NOT FOUND.
- **Method.** A lightweight module binds **frozen reconstruction features** to **metric, robot-relative 3D geometry**. It injects those spatial tokens into the **flow-matching action expert via cross-attention**, leaving the host's perception pathway unchanged. It adds **146.8M parameters (4.4% of 3.3B)** for the pi0.5 host.
- **RoboTwin 2.0:** 94.0% on clean scenes and 92.4% on randomized ones, vs WAM4D at 93.8% and 89.9%.
- **BEHAVIOR-1K:** six tasks. It "surpasses the 2025 challenge winner on five of six tasks, by up to 0.47 Q-score". It "exceeds a map-conditioned spatial policy on average across the three tasks both report". The map-conditioned policy is plausibly SERF, which has 3 tasks, but that is inferred.
- [search: https://arxiv.org/html/2609.35249v1]
- **Fit with the assigned module:** this is a **perception-to-action** spatial grounding method that requires fine-tuning the VLA action expert. It is **not** a persistent spatial memory or a prior map.

---

## 10. RAX (arXiv 2610.22496): online exploration

### Identity and code
- **Title and authors.** "RAX: Scaling Online Exploration for Robot Agents"; the bibtex title is "Scaling Online Exploration for Robot Agents". Runhan Huang, Xiangcheng Zhang (project lead), ..., Pieter Abbeel, Antonio Torralba, Dawn Song, Weirui Ye, Yilun Du.
- **Affiliations:** Harvard, NVIDIA, UC Berkeley, MIT, Stanford [repo: RR/robot-agent-exploration_rax/README.md:5-14, 261-267].
- **Release dates.** Paper on arXiv 2026-10-15. RoboDojo, BEHAVIOR-1K and real-robot branches released 2026-10-13 [repo: README.md:33-35].
- **License:** Apache-2.0 [README.md:249-252]. Real-robot session logs are on HF [README.md:217-223].

### Method (stated in docs)
- **Setup.** A general coding agent with a frozen model (Codex CLI, Claude Code, and since 2026-10-21 DeepSeek Harness and Kimi Code) drives the robot through tools. It uses "No action demonstration data, hand-engineered action primitives or value-function guidance" [README.md:57-70, 18-26].
- **Three steps** [README.md:117-129]:
  1. **Explore:** solve by hand via tool calls and record preconditions, actions and effects in memory.
  2. **Compile:** turn the successful sequence into `controller.py:main(ctx)` with step functions that check what they observe.
  3. **Evaluate:** the frozen program, or a live agent copy, plays held-out layouts.
- **BEHAVIOR backend (branch `robodojo-behavior/native`):**
  - **Pinned version.** "BEHAVIOR-1K `v3.9.3-post1` (`bd049de`), the 2026 challenge evaluation tag" [RR/rax_rb_native/docs/BEHAVIOR_BACKEND.md:9-11]. I confirmed `git rev-parse --short v3.9.3-post1` = `bd049de` in the local B1K clone (verified). Our HEAD is v3.9.3-post2 (`a8247a8`).
  - **Evaluator.** "The simulator is built by the official `BatchedEvaluator` (`num_envs=1`) ... Every episode is a fresh process." [BEHAVIOR_BACKEND.md:162-164].
  - **Agent inputs.** "Only challenge-track inputs ...: RGB, linear depth and the official 61-D proprioception, plus odometry integrated from the proprioceptive base velocity (start-relative; no world pose, map, object state or segmentation)" [BEHAVIOR_BACKEND.md:137-140].
  - **Tools:** `behavior_step` (1..150 native 23-D rows at 30 Hz), `behavior_base_move` (odometry closed loop), `behavior_step_eef` (closed-loop IK), `behavior_free_space_move` (cuRobo plan, self-collision only), `behavior_pixel_to_position`, and `behavior_preview_run` (a separate robot-only OmniGibson process) [BEHAVIOR_BACKEND.md:142-153].
  - **Instance split.** Exploration uses "the 300 train instances per task". Formal evaluation uses "the 20 public test instances (IDs 301-320, held out)", 10 episodes by default [BEHAVIOR_BACKEND.md:110-113].
  - **Step limit.** "1.5 x the mean human demo length at 30 Hz", from 3,224 to 39,090 steps [BEHAVIOR_BACKEND.md:105-108]. This is stated in RAX docs and not checked against B1K code here.
  - **Episode reviews (exploration only).** A Gemini analyst reads a ground-truth goal report written by the simulator. It gives the agent a text review with object names removed. "Formal episodes are never reviewed" [BEHAVIOR_BACKEND.md:205-245].

### BEHAVIOR-1K results (search)
- Q-score **46.5 vs 8.2** for the strongest prior challenge entry. Baselines are the five highest-ranked 2025 entries, on **ten public test instances**.
- Per task: boxing books 26.7, getting organized 9.0, sorting items 75.0, clearing food 40.0, clean-up desk 28.2, loading the car 100.0. Their mean, 278.9/6 = 46.48, is consistent with 46.5 (inferred arithmetic).
- "Boxing-books is evaluated as a standalone program, but the other five tasks use a fresh agent at evaluation whose workspace stays frozen."
- Other headline gains: RoboDojo +53.4, DexJoCo +31.4, LIBERO-Pro +21.2.
- [search: https://x.com/du_yilun/status/2111143198731149683, https://arxiv.org/html/2610.22496, https://assets.robot-agent-exploration.com/paper/rax.pdf]
- **Possible rules question (inferred):** the tools expose cuRobo motion planning and the simulator's IK. Planning "works relative to the holonomic base's virtual root, so base-frame targets are converted to world coordinates inside the simulator" [BEHAVIOR_BACKEND.md:279-281]. The 2026 wrapper must expose only RGB, depth and proprioception, and must not manipulate the environment [B1K:docs/challenge/evaluation.md:144]. Whether these simulator-side planning tools would pass the organizers' manual wrapper inspection is an **open question**.

---

## 11. Galaxea G0.5 (arXiv 2608.11739): VLA backbone and open weights

- **Exists.**
  - The GalaxeaVLA README says: "[Aug 12, 2026] Our G0.5 paper, **G0.5: One Autoregressive Stream for Robot Reasoning and Action**, is now available on arXiv" (links arXiv 2608.11739) [repo: RR/OpenGalaxea_GalaxeaVLA/README.md:4, 18].
  - The search result title is "Galaxea G0.5: One Autoregressive Stream for Robot Reasoning and Action" [search: https://arxiv.org/html/2608.11739v1].
  - The author list was NOT FOUND (org: Galaxea / OpenGalaxea).
- **Open weights: YES.**
  - `g05-base` was released 2026-06-16 on HF `OpenGalaxea/G05`. Other variants are `g05-so101`, `g05-droid`, `g05-libero` and `g05-robotwin20` [repo: README.md:20-28, 154-200].
  - Each model checkpoint is about 11 GB; the full set is about 55 GB [README.md:192].
- **License:** "G0.5 COMMUNITY LICENSE AGREEMENT (NON-COMMERCIAL + LIMITED PATENT LICENSE)", release date 2026-06-16 [repo: LICENSE-G0.5:1-2, 41].
- **Architecture (stated_docs):**
  - One transformer decoder emits reasoning and action tokens in one autoregressive stream.
  - Inputs: multi-view RGB, an embodiment ID, the instruction and proprioception.
  - Actions use a shared 27-D action space, `left_control(9)|left_gripper(1)|right_control(9)|right_gripper(1)|lower_body(7)`, with a residual-VQ tokenizer.
  - Chain-of-thought fields: `Subtask`, `BBox`, `Trace`, `ActionHint`.
  - Visual memory: 6 frames over 5 s.
  - Initialized from **Qwen3.5 2B**, pretrained on 14 embodiments [README.md:43-73].
- **BEHAVIOR-1K:** the README claims "a 0.3136 task success score on BEHAVIOR-1K" [README.md:73] (stated_docs). However, there is **no BEHAVIOR checkpoint or evaluation entrypoint**. `experiments/` holds only droid, libero, r1lite, r1pro, robotwin and so100 (verified by `ls`). The README says "Evaluation support for more simulation benchmarks will be released soon" [README.md:22].
- **Compute:** inference needs more than 8 GB of GPU memory; full fine-tuning needs more than 70 GB [README.md:114-122].
- **Real-robot path:** a policy server speaks WebSocket + msgpack, which mirrors the OpenPI pattern [README.md:216].
- **Note for us (inferred):**
  - G0.5 natively supports R1 Pro (an `experiments/r1pro` entrypoint), so it is a candidate alternative executor.
  - It would need BEHAVIOR fine-tuning (more than 70 GB per GPU for full fine-tuning).
  - Its non-commercial license must be acceptable for a challenge entry.
  - Its BEHAVIOR number cannot be reproduced from released artifacts.

---

## 12. The "Astra" model

### OpenAI GPT-6 Astra (the only API-served LLM/VLM named "Astra" found)
- **Official announcement:** "GPT-6 Astra: A new generation of intelligence", https://openai.com/index/gpt-6-astra/, dated **September 3, 2026**.
  - Also: https://openai.com/index/gpt-6-astra-next-generation-work/
  - Safety overview: https://openai.com/index/safety-overview-gpt-6-astra/
  - System card: https://deploymentsafety.openai.com/gpt-6-astra
  - [search]
- **Model ID:** `gpt-6-astra`.
  - The official model page is https://developers.openai.com/api/docs/models/gpt-6-astra. Per the snippet it lists "Text, Image ... Input only" for image and "Structured outputs: Supported".
  - The official vision guide (https://developers.openai.com/api/docs/guides/images-vision) and structured-outputs guide (https://developers.openai.com/api/docs/guides/structured-outputs) show examples with `model: gpt-6-astra`: an `input_image` content part, and a strict JSON schema via the Responses API text format.
  - [search]
  - **No dated snapshot ID found.** Do not guess one.
- **Endpoints:** Chat Completions, Responses and Batch. The free API tier is not supported [search: https://developers.openai.com/api/docs/models/gpt-6-astra snippet, https://router.one/blog/gpt-6-astra-api-guide].
- **Context, output and price:**
  - 1,050,000-token context and 128,000 max output.
  - $10 per 1M input tokens and $50 per 1M output tokens.
  - Mostly from aggregators; OpenRouter lists `openai/gpt-6-astra` with the same values [search: https://openrouter.ai/openai/gpt-6-astra/api, https://llm-stats.com/models/gpt-6-astra].
- **Availability:**
  - It started as a limited rollout to select organizations: "Astra is not yet generally available".
  - Reports describe a "messy rollout" and delayed developer access [search: https://thenewstack.io/gpt6-astra-developer-access-delayed/].
  - GA is **unconfirmed** as of my searches.
  - Azure and AWS Bedrock availability is mentioned in snippets only.
  - An alias `~openai/gpt-astra-latest` appears only on aggregators (cloudprice.net). It is UNVERIFIED.
- **Related models:** OpenAI also introduced GPT-6 Sol and Luna (2026-09-22) [search: https://openai.com/index/introducing-gpt-6-sol-and-luna/]. StageGuard compares against "GPT-5.6 Luna" (per the Pith snippet).

### Other "Astra"s (not usable as an LLM API model ID)
- **Google DeepMind Project Astra.** A research prototype whose capabilities power Gemini Live API features (2025 coverage) [search: https://techcrunch.com/2025/05/20/project-astra-comes-to-google-search-gemini-and-developers/]. No standalone model ID was found.
- **ByteDance "Astra".** Robot navigation: Astra-Global is a multimodal LLM for localization; Astra-Local handles planning and odometry. arXiv 2506.06205 [search: https://arxiv.org/pdf/2506.06205]. Not an API.

### Relevance to the challenge (stated in B1K docs)
- **LLM-based querying is allowed.** "Additional components like SLAM or LLM-based querying are also permitted ... If a submission depends on external model-query APIs, participants must provide the credentials, quota, and serving configuration needed for evaluation; the organizers will not cover external API usage costs." [B1K:docs/challenge/evaluation.md:30].
- **MobiAgent's requirement matches.** Its planner and judge need "A model supporting image inputs and JSON output" [repo: MobiAgent README.md:154]. GPT-6 Astra matches that description per the docs snippets (inferred).

---

## 13. Differences between what pages claim and what code shows

1. **MobiAgent.** The paper claims BEHAVIOR-1K results (65.0% on 4 tasks, from search). The repo releases **no BEHAVIOR-trained weights**: "no trained weights are included" (docs/training.md:24). Only a gated RoboCasa checkpoint is linked.
   - Its BEHAVIOR adapter is its own gello-config environment at 224x224, not the official `RGBDFullResWrapper` (720/480).
2. **SERF.** The paper presents "BEHAVIOR-1K results". The evaluation code uses `seg_instance`, world camera poses and per-eval-instance prebuilt maps, and patches task 21's BDDL goal. All of this is outside the official 2026 challenge-track rules.
3. **Galaxea G0.5.** The README claims 0.3136 on BEHAVIOR-1K, but no BEHAVIOR checkpoint or evaluation entrypoint is released.
4. **RoboHarness (2607.18060).** The abstract says "three public benchmarks"; the HTML version says "five" (search snippets).
   - The name is shared with 2610.14018 (BinceQu/RoboHarness, a training-free LLM click/keypoint harness). That repo uses B1K v3.9.1 and the **2025** evaluator with a **2x** step multiplier and 5 instances per task. This differs from the 2026 protocol, where RAX docs state 1.5x and 10 or 20 test instances.
5. **Version pins.** RAX pins B1K v3.9.3-post1 (bd049de) as "the 2026 challenge evaluation tag". The local clone's HEAD is v3.9.3-post2 (a8247a8). An `origin/2026-eval` branch also exists (38d2c9fb). Which ref the organizers will actually use is a question for another research topic.
6. **Cracking.** The README and the abstract both say code will be released; the repo currently holds only the README, PDF and images. The BEHAVIOR numbers in the README table (14% to 37%) match Table 14 in the PDF.
7. **StageGuard.** The headline number (">90%") appears only on an AI-generated review site. It is unverified against the paper.
8. **ForesightFlow.** The paper's title does not contain "ForesightFlow". The `ForesightFlow` GitHub org is an unrelated project.

---

## 14. Open questions
- **Exact BEHAVIOR protocols.** For MobiAgent (which 4 tasks), ForesightFlow (5 tasks), StageGuard, RoboHarness 2607.18060 (6 tasks) and Spatial Grafting (6 tasks): which evaluator version, which instances, and whether the metric is SR or Q? This needs full-text access, and arxiv.org is blocked here.
- **Affiliations not found:** ManiUnit, ForesightFlow and Spatial Grafting. Spatial Grafting's is probably Huawei, but that is unverified.
- **Code not found:** ManiUnit, ForesightFlow, RoboHarness 2607.18060, StageGuard and Spatial Grafting. No public repo was found for any of them.
- **GPT-6 Astra for our account:**
  - Is `gpt-6-astra` generally available to our org now, and at what rate-limit tier?
  - Does a dated snapshot ID exist? Check with `GET /v1/models` before hard-coding anything.
  - Can it combine image input and a strict JSON schema in one request? The docs show each separately.
- **RAX tool legality.** Would the organizers accept simulator-side planning tools such as cuRobo or IK inside a submitted wrapper? The 2026 rules allow TAMP but forbid direct environment manipulation and privileged inputs [B1K:docs/challenge/evaluation.md:30, 144].
- **Galaxea license.** Is G0.5's non-commercial license acceptable for a challenge submission? Is a BEHAVIOR-tuned G0.5 checkpoint planned?

---

## 15. What this means for our agent design
1. **Executor.** Train or serve a **skill/subtask-conditioned pi0.5** on the segmented BEHAVIOR-1K demos, using the provided skill/subtask annotations, rather than a full-task policy.
   - ManiUnit reports 78.7% vs 49.3% local success [search].
   - Cracking trains its BEHAVIOR subtask VLA on subtask-segmented challenge data, while its pi0.5 full-trajectory baseline gets 7/100 [CMM-pdf:L1505-1513].
   - MobiAgent uses six skill experts on a frozen pi0.5 backbone [repo].
2. **Planner contract.** Copy MobiAgent's one-subtask-per-call JSON schema: `stage_hint`, `prompt`, `success_check`, `failure_cues`, `max_retries`, `target_object_name`, `plan_sketch`. Keep termination and retry budgets in deterministic orchestrator code, not in the LLM [repo: PROMPTS.md:10-24; orchestrator.py:339-371].
3. **Checker.**
   - Compare before and after frames from all three cameras. Return `complete/incomplete/error` plus `next/retry/replan_keep_pose`, and default to `incomplete` [repo: judge_vlm.py].
   - Treat "gripper closed" as not proof of a grasp [CMM-pdf:L1296].
   - For latency and cost, distill the big-VLM checker into a small student on demo stage boundaries (StageGuard idea; no code available).
4. **Approach and recovery.** Implement Cracking's geometric skills using only RGB-D and proprio, which the challenge allows:
   - grounding plus depth back-projection;
   - reach thresholds of 0.75 m and 1.00 m, to be re-estimated from BEHAVIOR train trajectories for R1Pro;
   - a local XY shift of at most 0.25 m;
   - A* on a 0.05 m depth occupancy grid with 0.10 m inflation;
   - end-effector correction of at most 0.30 m.
5. **Spatial memory.** Build it online from RGB-D, relative camera poses and proprio-integrated odometry, as RAX does. **Do not** copy SERF's evaluation (prebuilt per-instance maps, `seg_instance`, world camera poses). Cracking's adaptive frame recall (top-16 of ≤256 past frames) is a cheap, legal alternative.
6. **Exploration.** RAX-style exploration can be run **offline on the 300 train instances** to write skill notes and programs, consistent with "collect additional data yourself" [B1K:docs/challenge/evaluation.md:28]. Any simulator-side tool (cuRobo, IK) used at evaluation must pass the organizers' wrapper inspection [B1K:docs/challenge/evaluation.md:144].
7. **LLM provider.** Keep the provider pluggable. `gpt-6-astra` is a documented OpenAI model ID with image input and structured outputs (search snippets), but GA is unconfirmed and no snapshot ID was found. Check `/v1/models` at startup and fail closed. Per the rules, our submission must supply API credentials and quota [B1K:docs/challenge/evaluation.md:30].
8. **Do not quote paper numbers as official BEHAVIOR scores.** Most of these papers use subsets of tasks and episodes, older evaluator versions, or modified goals (SERF task 21). The only reference with full official 2025 per-task numbers is the RAX-generated table in `RR/rax_rb_native/docs/BEHAVIOR_TASKS.md` (stated_docs, generated by their script).

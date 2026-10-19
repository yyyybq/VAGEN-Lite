# 转移题、混合 SFT、输出格式开关、诊断、实验矩阵：总报告

2026-10-19（第二版，加上当天第二批要求的结果）。任务说明：用户 2026-10-17 的提示词（第 0–8 节）和 2026-10-19 的两批要求。

- 代码和文档都在分支 `transition-qa`（工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data`），只在本地，没有推送，没有合并到 main。
- 分支和 main 保持同步（最近一次合入 main `71372b7e`）；合并到 main 由你在冻结期触发。
- 逐步记录见 `PROGRESS.md`（2026-10-19 第二批在第 10 节），改动清单见 `CHANGES.md`。
- 字段和规则见 `DATA_CONTRACT.md`，题型见 `P1_QTYPES.md`。

> 名字提醒：`v10_p1` / `v10_p2` 是 RL 修正包，和数据池 P1（转移题）/ P2（动作轨迹）无关。

## 结论

1. **转移题题库（P1）按 2026-10-19 的规则重建了，训练题从 35,039 道增加到 90,101 道。**
   - 规则：状态题全保留；锚点放开到每条轨迹的全部可用帧；前向题 ≤ 状态题 1.2 倍；逆向题 ≤ 15%；单场景 ≤ 8%。
   - 训练集：状态 34,812（38.6%）/ 前向 41,774（46.4%）/ 逆向 13,515（15.0%）；测试集 6,035 道；探针集 600 道。
   - 字母和均匀分布最多差 2.9 个百分点（验收线 5）；train 和 val 场景没有交叉；全部 11,110 条轨迹的终点成功、终点前不成功。
   - 状态题全保留以后，"当前是否已满足要求"（`goal_now`）占状态题的 70.9%。旧规则下这类题最多占一半；你定了维持现在这样（第 2 节）。
2. **探针集换成第二版，诊断 3 重跑：动作训练基本不改变回答转移题的能力。**
   - 探针集：600 道，15 个题类各 40 道，来自第十二轮 v2 验证集的 97 个场景。遮挡题不够，按预授权给 v2 新增的 15 道遮挡题渲染了 58 张帧。
   - 23 个检查点（原始 7B、SFT 10 个 epoch、P-S0 和 P-D 各 6 个）在 42.3%–48.5%；原始 7B 45.8%，随机猜 36.1%，标准误约 ±2 个百分点。
   - 所有模型在"当前是否已满足要求"上几乎总答 No。训练后的模型在"A 被 B 挡住了吗"上低于随机，因为多答"没有"。
   - **格式不合规率**：训练过的模型 48%–80%，几乎全部是写成 `(B)`，读得出，没有一个用动作格式；原始 7B 15.8%，全是先推理再答。
3. **C0（主线，不新训练）加进矩阵，SFT 配方和主线逐项一致。**
   - C0 = 第十一轮正式 SFT 第 9 个 epoch + 成对 RL，就是第十二轮正在跑的复现对（R12-P1-S0 / -D），结果直接取那条训练链。
   - B / C / D 的 SFT 超参数和主线完全相同；P2 训练行逐条相同、顺序相同；不同的只有数据和验证集。
   - RL 标准名做成了变量（`configs/matrix_v2/vars.yaml`），换 p2 只改一行。
4. **E 组的两件事做了。**
   - RL 日志记录含思考文字的回答比例：`transition/think_rate`。
   - 回答上限只对 E 升到 6,144：`rl-config --max-response-length`。
   - 为此改了 5 个现有文件，都是加选项或加指标，默认行为不变（`CHANGES.md` 第 1 节）。
5. **30 道人工核对还没有结果。** 核对页加了每道题的过滤检查和"没出的候选题"表：https://claude.ai/artifact/NszETEpoarL6CXzbAZKdfQ 。
6. **测试**：和 main 同步后 308 个测试全部通过；之后改的诊断包 11 个通过。
7. **用卡**：本任务累计约 4.8 个 H800 卡时，都是 SCO 单卡任务，跑完的都已删除。
   - 2026-10-18 约 3.4 卡时：渲染 7,528 + 14 张、干跑、36 个检查点的旧探针；
   - 2026-10-19 约 1.4 卡时：渲染 58 张、23 个检查点的新探针。

## 术语

- **状态 / 锚点状态**：状态是 SFT 轨迹上的一个相机位姿和它的图；锚点状态是拿来出题的状态。
- **门**：任务成功判据里的一项条件，例如"目标在画面里的部分不少于 95%"；全部通过才算成功。
- **状态题 / 前向题 / 逆向题**：
  - 状态题问当前这张图，例如"现在 A 在 B 的左边还是右边"；
  - 前向题问某个动作之后，例如"左转 40 度后哪个更近"，不另外渲染；
  - 逆向题给前后两张图，问相机做了什么动作。
- **P0 / P1 / P2**：三个 SFT 数据池。P0 是仿 8 个 benchmark 的 QA（现在只有 230 道样题），P1 是转移题，P2 是第十一轮的 SFT 动作轨迹。
- **正式题 / 后备题**：生成器给每个场景出的两批题。SFT 只用正式题，RL 只用后备题（第十一轮第 7 条决定）。
- **格式不合规**：探针题的回答不是题目要求的"只写选项字母 / 只写 Yes 或 No"；宽松读法仍可能读得出。

## 1. 每一步

| 步 | 做了什么 | 结果 | 详见 |
| --- | --- | --- | --- |
| 1 | 读代码，写"题型 → 复用函数"对照表，估渲染预算 | 几何判定全部复用现有函数；5 件事你都选了推荐项 | `PROGRESS.md` 第 1 节 |
| 2 | 出状态题、前向题、逆向题；按预授权渲染 7,528 个目标区域内部的成功状态 | 2026-10-18 的题库 | 第 2 节 |
| 3 | 混合配方构建器 `data_gen/sft_mixture/` | 干跑通过；配方表 | 第 3 节、`RECIPES.md` |
| 4 | 诊断 1（失败拆解）和诊断 3（P1 探针）的基线 | 失败主要是"差最后一步"；探针平 | 第 4 节、`DIAG_BASELINE.md` |
| 6 | 输出格式开关 | 历史轨迹上两种写法读出的动作 99.98% 相同 | 第 6 节、`CHANGES.md` |
| 7 | 矩阵配置 `configs/matrix_v2/` | 按第十二轮更新：成对 RL、10 个 epoch、v2 题集 | 第 7、9 节 |
| 10.2 | 第二版探针集，诊断 3 重跑 | 结论第 2 条 | 第 10.2 节、`DIAG3_PROBE.md` |
| 10.3 | 数据契约、题型目录 | | `DATA_CONTRACT.md`、`P1_QTYPES.md` |
| 10.4 | 新的比例规则，重建题库和混合配方 | 结论第 1 条 | 第 10.4 节、`P1_STATS.md` |
| 10.5 | C0、SFT 配方核对、RL 标准名做成变量 | 结论第 3 条 | 第 10.5 节、`configs/matrix_v2/README.md` |
| 10.6 | E 组：思考文字比例、回答上限 6,144 | 结论第 4 条 | 第 10.6 节、`CHANGES.md` |

**P1 题库（2026-10-19，`exps/transition_qa_20261019`）**

| 大类 | 训练 | 测试 | 探针 |
| --- | --- | --- | --- |
| 状态题 | 34,812（`goal_now` 24,672、左右 3,515、距离 2,397、视野 1,711、哪个更近 1,444、高度 806、遮挡 267） | 2,332 | 280 |
| 前向题 | 41,774（`goal_after` 20,887，其余 6 类 20,887） | 2,798 | 280 |
| 逆向题 | 13,515 | 905 | 40 |
| 合计 | 90,101 | 6,035 | 600 |

**矩阵六组（训练条数；一条 = 一段对话，P2 一条是一整条多回合轨迹）**

| 组 | P0 | P1 | P2 | 输出格式 | 之后 |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 只输出动作 | 原始 7B 直接 RL |
| B | 177（230 道样题） | 0 | 0 | 只输出动作 | RL |
| C0 | 0 | 0 | 9,802（主线，第 9 个 epoch） | 只输出动作 | 第十二轮的复现对，不新训练 |
| C | 177 | 0 | 9,802 | 只输出动作 | RL |
| D | 177 | 9,802 | 9,802 | 只输出动作 | RL |
| E | 同 D（共用 D 的 SFT 模型） | | | RL 时 `free_think`，回答上限 6,144 | RL（探索性，前 50 次更新解析失败率超过 20% 就放弃） |

所有组按第十二轮的决定设置：
- SFT 训 10 个 epoch，按 val 挑最好的 epoch；
- RL 只用后备题，每组成对跑（`{rl_package}_S0` 和 `_D`，现在 `v10_p1`，只差奖励，种子 20261019），各训 300 次更新；
- 评测集是第十二轮的 v2：val 413 道、test 292 道。

## 2. 等你决定的事（按影响排序，推荐项放第一个）

**已定（2026-10-19）：状态题里的 goal 题不再设上限。**

- "状态题全保留"和 2026-10-18 保留的"goal 题最多占所在大类一半"冲突。你定了维持现在这样：
  - 状态题全保留，`goal_now` 占状态题 70.9%（24,672 / 34,812）；
  - `goal_after` 最多占前向题一半这条仍然保留。
- 题库不用重建：`exps/transition_qa_20261019` 就是这个规则。
- 另一个选项（没有采用）：状态题里也限一半，训练集约 5.2 万道。

还没定的：

1. **训练链的 7 个字段**（`CHANGES.md` 第 5 节）：
   - (a) **推荐**：在 `train_chain.py` 里加上，矩阵各组就能无人值守地跑；
   - (b) 按各组 `manual_steps` 手工准备 progress.json。
2. **合并到 main**：由你在冻结期触发。分支已和 main 同步，308 个测试通过。
   - 本分支改了 5 个现有文件：`training_prep.py`、`env.py`、两个 agent loop、`ray_trainer.py`；都是加选项或加指标。
   - 合并会把 `transition/think_rate` 带进所有之后的 RL 日志。
3. **30 道人工核对**：在核对页上点完即可，我读回后写通过率。其中 22 道在新题库里，题干、选项、答案不变。
4. **修正包 p1 还是 p2**：第十二轮定了以后改 `configs/matrix_v2/vars.yaml` 的 `rl_package` 一行；C0 自动对应 r12a 或 r12b。
5. **P0 全量题库什么时候生成**：由你通知。现在 B 组只有 177 条记录。
6. **（可选）E 组正式跑之前，先用 `--trial` 在 GPU 上试 6,144 的回答上限**：一条序列最长从 8,192 变成 10,240 个词元，显存没试过。
7. **（可选）诊断 2，历史打乱**：代码已有，约 1 卡时。

## 3. 风险和局限

- **P0 很小**：只有 230 道样题，B 组几乎等于不做 SFT；P0 的验证集来自 train 场景。
- **goal 题占比高**：`goal_now` 占状态题 70.9%、`goal_after` 占前向题 50%，两类合计占训练集 50.6%（45,559 / 90,101）。探针显示现有模型在这类题上只会答 No；D 组练了以后能否改过来，看 D 的探针结果。
- **遮挡题少、答案不均**：
  - 遮挡族只占正式题的 2.8%：训练集里 `occluded_now` 267 道、`occluded_after` 577 道。
  - 探针的 `occluded_now` 答案是完全 16 / 部分 17 / 没有 7，总答"部分"能得 42.5%。
- **左右前向题的答案几乎从不改变**：转身不改变两个物体的左右顺序。这是物理事实，照实保留。
- **逆向题的动作分布**：动作来自最短路径，后退多、前进少。训练集字母均匀，动作种类不均匀。
- **探针的局限**：
  - 每类 40 道，题类之间的差别要超过约 15 个百分点才可靠。
  - "相机做了什么动作"两版探针的结论相反（抽法不同），不下结论。
  - D 组练过 P1，探针题和 P1 同一种格式，D 在探针上会直接占优；探针主要看 QA 能力在 RL 中保持还是丢失，迁移要看评测成功率。
- **A 组的风险**：从原始 7B 起步（验证集成功 5.8%），"8 条轨迹里有成有败"的题会很少，每次更新可能凑不满。
- **资源池**：2026-10-19 `zoetrope` 满，单卡任务在那里排了约 50 分钟没起来；`h800` 立刻起来。矩阵的 plan 用的是 `h800`。
- **没有在 GPU 上跑过的代码**：
  - 诊断 2（历史打乱）；
  - 诊断 4（迁移表，等矩阵结果）；
  - `rl-config --output-format think_action` / `--max-response-length` 生成的配置；
  - `transition/think_rate`（只有单元测试）。

## 4. 产物

| 什么 | 在哪（工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data` 下） |
| --- | --- |
| 转移题生成器、测试 | `data_gen/transition_qa/`（`generate.py` raw / probe / build、`extra_tasks.py`、`interior.py`、`review.py`、`report.py`） |
| 题库（现行） | `exps/transition_qa_20261019/`（`p1_train/test/probe.jsonl`、`stats.json`、`probe_stats.json`、`raw/`、`extra_oa15/`） |
| 题库（2026-10-18，保留） | `exps/transition_qa_20261018/`（含内部状态的图 `interior/`、核对题 `review/`） |
| 混合配方构建器、配置、矩阵 plan 生成 | `data_gen/sft_mixture/`（`build.py`、`matrix_plan.py`） |
| 矩阵的混合数据 | `exps/sft_mixture_matrix_v2_20261018/{B,C,D}`（2026-10-19 用新题库重建） |
| 诊断代码、测试 | `eval/diagnostics/` |
| 诊断结果 | `exps/diagnostics_20261018/`（失败拆解、旧探针、`format_check`）、`exps/diagnostics_20261019/probe_v2_all/`（新探针、`curve.md`、`curve.png`） |
| 矩阵配置 | `configs/matrix_v2/`（A–E、C0、`vars.yaml`、README） |
| 改动的现有文件 | `CHANGES.md` 第 1 节 |
| 文档 | `docs/transition_qa/`：本报告、`PROGRESS.md`、`DATA_CONTRACT.md`、`P1_QTYPES.md`、`P1_STATS.md`、`RECIPES.md`、`DIAG_BASELINE.md`、`DIAG3_PROBE.md`、`CHANGES.md`、两张探针曲线图 |

## 5. 复现

在工作树下执行。`PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python`；
`SFT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft`；`META=/mnt/umm/users/yinbaiqiao/InteriorGS_meta`；
`R12=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019`；`OUT=exps/transition_qa_20261019`。

```bash
# P1 题库：候选题（CPU，12 进程约 10 分钟）、内部状态题（图用 2026-10-18 的渲染）
$PY -m data_gen.transition_qa.generate raw --sft-dir $SFT --scenes-root $META --out $OUT --workers 12
$PY -m data_gen.transition_qa.interior questions --sft-dir $SFT --out $OUT --interior exps/transition_qa_20261018/interior
# v2 新增的 15 道遮挡题：最短路径帧（58 张）、渲染（单卡 1 分钟；zoetrope 满时用 h800）、候选题
$PY -m data_gen.transition_qa.extra_tasks frames --tasks $R12/eval_val_v2/tasks.jsonl --ids-from $R12/eval_val_v2_oa/selection.json --sft-dir $SFT --out $OUT/extra_oa15
$PY data_gen/interiorgs_v10/pilot_sco.py prepare --pilot $OUT/extra_oa15 --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS_ply   # 再提交
$PY -m data_gen.transition_qa.extra_tasks trajectories --dir $OUT/extra_oa15
$PY -m data_gen.transition_qa.generate raw --sft-dir $OUT/extra_oa15 --scenes-root $META --out $OUT --tag oa15
# 探针集、构建、统计
$PY -m data_gen.transition_qa.generate probe --out $OUT --split-manifest $SFT/split_manifest.json --scenes-from $R12/eval_val_v2/tasks.jsonl
$PY -m data_gen.transition_qa.generate build --out $OUT --split-manifest $SFT/split_manifest.json
$PY -m data_gen.transition_qa.report --out $OUT --md docs/transition_qa/P1_STATS.md
# 混合配方、矩阵 plan
$PY -m data_gen.sft_mixture.build --config data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml
$PY -m data_gen.sft_mixture.matrix_plan --group D --commit <main 的提交> --out plan.json
# 诊断
$PY -m eval.diagnostics.sco_probe --run <目录> --probe $OUT/p1_probe.jsonl --from-eval <评测目录> --names <结果目录名> --pool h800 --submit
$PY -m eval.diagnostics.qa_probe rescore --probe $OUT/p1_probe.jsonl --out <结果目录>
$PY -m eval.diagnostics.qa_probe curve --out <结果目录> --chain "R11-SFT=base_7b:0,sft_formal_e1:154,..."
$PY -m eval.diagnostics.failure_breakdown --eval <评测目录> --out <前缀>
```

## 6. 提交（分支 `transition-qa`，从旧到新）

- 2026-10-17 至 18：
  - `081465f2` 第 1 步文档；
  - `4f3862af`、`d7447b46`、`1a19d536`、`e6b0bc6a` 第 2 步；
  - `3322b8c2`、`2416c3be`、`f7c39f26` 第 3 步；
  - `c7a549f9`、`263ee8b7`、`c9e886be`、`39d5d941` 第 4 步；
  - `2db60377`、`a842323a`、`6dcd8a1f` 第 6 步；
  - `28849c0f`、`fea558ed` 第 7 步；
  - `0f02b448` 第一版报告。
- 2026-10-19：
  - `bd29d26b` 合入 main，`54a69d7e` 合并记录；`f3456c80`、`adc041e9` 矩阵按第十二轮更新；
  - 第二批：
    - `1359c557` 题库规则和探针集；
    - `84a9adba` 探针的宽松读法和格式不合规率；
    - `548a7c06` E 组的指标和回答上限；
    - `eeed7669` C0、标准名变量、数据契约；
    - `a1d09600` 合入 main `71372b7e`；
    - 以及结果和文档的提交。

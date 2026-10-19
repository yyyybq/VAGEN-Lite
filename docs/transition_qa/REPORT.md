# 转移题、混合 SFT、输出格式开关、诊断、实验矩阵：总报告

2026-10-19。任务说明：用户 2026-10-17 的提示词（第 0–8 节）。代码和文档都在分支 `transition-qa`
（工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data`，从 main `a59f9583` 开出），17 个提交，只在本地，没有推送，没有合并。
逐步记录见 `PROGRESS.md`，改动清单见 `CHANGES.md`。

## 结论

1. **转移题题库（P1）生成完了，除人工核对外的验收都通过。**
   - 规模：训练集 35,039 道，只来自 train 场景；测试集 4,073 道、探针集 500 道，只来自 val 场景。
   - 比例：训练集里状态题、前向题、逆向题占 49.9% / 34.9% / 15.3%，目标 50 / 35 / 15。
   - 平衡：每种题的正确字母和均匀分布最多差 2.4 个百分点，验收线 5。
   - 划分：train 和 val 场景没有交叉。
   - 一致性：答案全部来自环境和生成器自己的判定函数，在全部 11,095 条轨迹上和环境的成功判定一致，没有发现判定函数之间的矛盾。
   - **30 道人工核对还没有结果**：核对页已发布（https://claude.ai/artifact/NszETEpoarL6CXzbAZKdfQ ），截至写这份报告时没有提交任何选择。
2. **混合配方构建器能用了。**
   - 一个配置文件给出三个池（P0 仿 benchmark 的 QA、P1 转移题、P2 SFT 动作轨迹）的来源和每组的条数。
   - 230 道 P0 样题、500 道 P1、300 条 P2 在 SCO 单卡上被 LLaMA-Factory 全部加载并训练 5 步。
   - 三个池的验证损失分别算出：P0 0.494、P1 0.340、P2 0.200。
3. **诊断基线：现有模型的主要失败是"差最后一步"，而现有动作训练几乎没有改变模型回答转移题的能力。**
   - 失败拆解：所有 SFT、RL 模型都有 28%–37% 的题用完 12 个动作时，最好的那个回合只差一个成功条件；提前停下的不超过 0.4%。
   - P1 探针：36 个检查点都在 43.4%–48.0%，原始 7B 45.4%，随机猜 36.4%。
   - 只有"相机做了什么动作"随 SFT 上升（49% → 59%–78%）。
   - "当前是否已满足要求"所有模型都在 51%，等于随机猜。
4. **输出格式开关：只改了一个现有文件。**
   - 改动：`training_prep.py rl-config` 加了 `--output-format think_action`（15 行），供 E 组 RL 用 `free_think` 提示；其余几处开关原来就有。
   - 验收：在 278 道和 279 道评测题的历史轨迹上（47 个模型，86,067 个回答），两种写法读出的动作 99.98% 相同。
5. **矩阵 A–E 的配置写好了，没有启动。**
   - 启动前要先把分支合并到 main（等第十一轮训练链结束，由你决定）。
   - 训练链还要加 7 个字段（每个都写了不改训练链时的手工办法）。
   - 用卡估计约 600 卡时，RL 占大头。
6. **本任务用了约 3.4 个 H800 卡时**，都是 SCO 单卡任务：渲染 7,528 + 14 张图、LLaMA-Factory 干跑、36 个检查点的探针。
   跑完的任务都已删除，28 GB 的干跑权重也已删除。

## 术语

- **状态 / 帧对**：状态是 SFT 轨迹上的一个相机位姿和它的图；帧对是同一条轨迹里相邻的两张图，中间是一个回合的动作。
- **门**：任务成功判据里的一项条件，例如"目标在画面里的部分不少于 95%"；全部通过才算成功。
- **状态题 / 前向题 / 逆向题**：
  - 状态题问当前这张图，例如"现在 A 在 B 的左边还是右边"；
  - 前向题问某个动作之后，例如"左转 40 度后哪个更近"，不另外渲染；
  - 逆向题给前后两张图，问相机做了什么动作。
- **P0 / P1 / P2**：三个 SFT 数据池。P0 是仿 8 个 benchmark 的 QA（现在只有 230 道样题），P1 是转移题，P2 是 SFT 动作轨迹。
- **正式题 / 后备题**：生成器给每个场景出的两批题。SFT 只用正式题，RL 只用后备题（第十一轮第 7 条决定）。

## 1. 每一步

| 步 | 做了什么 | 结果 | 详见 |
| --- | --- | --- | --- |
| 1 | 读代码，写"题型 → 复用函数"对照表，估渲染预算 | 几何判定全部复用现有函数；推荐的方案 0 张渲染；5 件事请你定（你都选了推荐项） | `PROGRESS.md` 第 1 节 |
| 2 | 出状态题、前向题、逆向题（逆向题用现有帧对，并入本步） | P1 题库，见结论第 1 条；按预授权渲染 7,528 个目标区域内部的成功状态 | 第 2 节、`P1_STATS.md` |
| 3 | 混合配方构建器 `data_gen/sft_mixture/` | 干跑通过；A–E 的配方表 | 第 3 节、`RECIPES.md` |
| 4 | 诊断 1（失败拆解）和诊断 3（P1 探针）的基线；诊断 2、4 只写了代码 | 见结论第 3 条 | 第 4 节、`DIAG_BASELINE.md` |
| 5 | 逆向题渲染 | 并入第 2 步，用现有帧对，不渲染 | 第 2 节 |
| 6 | 输出格式开关 | 见结论第 4 条 | 第 6 节、`CHANGES.md` |
| 7 | 矩阵配置 `configs/matrix_v2/` | 五组配置、启动前要做的 | 第 7 节、`configs/matrix_v2/README.md` |

**P1 题库的组成（训练集）。**

| 大类 | 题类 | 题数 |
| --- | --- | --- |
| 状态题 | 当前是否已满足要求 / 左右 / 距离几米 / 是否都在画面里 / 哪个更近 / 高度占几成 / 被挡的程度 | 8,719 / 3,231 / 2,123 / 1,320 / 1,151 / 716 / 207 |
| 前向题 | 动作后是否满足要求 / 距离 / 高度 / 是否都在画面里 / 左右 / 哪个更近 / 被挡的程度 | 6,104 / 2,522 / 958 / 860 / 856 / 754 / 162 |
| 逆向题 | 两帧之间的动作 | 5,356 |

**矩阵五组（训练条数；一条 = 一段对话，P2 一条是一整条多回合轨迹）。**

| 组 | P0 | P1 | P2 | 输出格式 | 之后 |
| --- | --- | --- | --- | --- | --- |
| A | 0 | 0 | 0 | 只输出动作 | 原始 7B 直接 RL |
| B | 177（230 道样题） | 0 | 0 | 只输出动作 | RL |
| C | 177 | 0 | 9,802 | 只输出动作 | RL |
| D | 177 | 9,802 | 9,802 | 只输出动作 | RL |
| E | 同 D（共用 D 的 SFT 模型） | | | RL 时 `free_think` | RL（探索性，前 50 次更新解析失败率超过 20% 就放弃） |

所有组 SFT 训 5 个 epoch，按 val400 挑最好的 epoch。RL 只用后备题，配修正包 `p1` 和稀疏奖励 S0，训 300 次更新。
评测集是 val400 和 test（279 道）。

## 2. 等你决定的事（按影响排序，推荐项放第一个）

1. **合并 `transition-qa` 到 main：** 矩阵要用的代码都在这个分支，训练链只按 main 的提交运行。
   - (a) **推荐**：第十一轮训练链结束后合并。分支只改了 `training_prep.py` 一个现有文件，默认行为不变，其余 39 个是新文件。
   - (b) 先不合并。
2. **训练链的 7 个字段**（`CHANGES.md` 第 5 节）：
   - (a) **推荐**：在 `train_chain.py` 里加上，矩阵五组就能无人值守地跑；
   - (b) 按各组 `manual_steps` 手工准备 progress.json（第十一轮修正包 RL 用过这个办法）。
     A、B、C、D 可以这样跑；E 的评测和输出格式要多手工两步。
3. **E 组的两件事**（你说先按待定写进配置）：
   - 要不要记录"带思考文字的回答比例"。在 `free_think` 提示下只回动作也不扣分，模型可以完全不写思考文字；
   - 一条序列模式的回答上限要不要从 4,096 放宽到约 6,144。
   两件都要改现有文件（vagen 的指标、`training_prep.py` 的常量）。
4. **30 道人工核对：** 在核对页上点完即可，我读回后写通过率。
5. **P0 全量题库什么时候生成：** 由你通知。现在 B 组只有 177 条记录。全量可用时改混合配置的 P0 路径、重建，并给 P0 一份 val 场景的验证集。
6. **（可选）诊断 2，历史打乱：** 代码已有。对现有模型跑三档（正常 / 打乱 / 删掉历史），每档每个模型一个单卡评测，约 1 小时。

## 3. 风险和局限

- **P0 很小：** 现在只有 230 道样题。B 组几乎等于不做 SFT；P0 的验证集来自 train 场景（样题导出时每 10 道留 1 道）。
- **"任务要求满足了吗"类题的上限：** 这类题在状态题、前向题里各最多占一半（你决定保留），所以训练集停在 35,039 道。
  去掉上限能到约 6.8 万道，其中这类题约占 59%。
- **左右前向题的答案几乎从不改变：** 转身不改变两个物体的左右顺序，所以 `left_right_after` 的答案基本等于现在的答案。这类题信息量低，只占训练集 2.4%。
- **逆向题的动作分布：** 动作来自最短路径，后退较多（可用帧对里约三分之一是后退）。正确选项的字母是均匀的，但动作种类不均匀。
- **探针的局限：**
  - 每种题约 37 道，只适合看总体和三大类；
  - D 组练过 P1，探针题和 P1 同一种格式，D 在探针上会直接占优；所以探针主要看 RL 过程中 QA 能力保持还是丢失，迁移要看评测成功率。
- **A 组的风险：** 从原始 7B 起步（验证集成功 5.8%），修正包只训"同一道题 8 条轨迹里有成有败"的题，这种题会很少，每次更新可能凑不满。
- **没有在 GPU 上跑过的代码：** 诊断 2（历史打乱）、诊断 4（迁移表，等矩阵结果）、`rl-config --output-format think_action` 生成的配置（只有单元测试）。

## 4. 产物

| 什么 | 在哪（工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data` 下） |
| --- | --- |
| 转移题生成器、测试 | `data_gen/transition_qa/` |
| 题库、统计、原始候选、内部状态的图、核对题 | `exps/transition_qa_20261018/`（`p1_train/test/probe.jsonl`、`stats.json`、`raw/`、`interior/`、`review/`） |
| 混合配方构建器、配置、测试 | `data_gen/sft_mixture/` |
| 干跑、矩阵的混合数据 | `exps/sft_mixture_dryrun_20261018/`、`exps/sft_mixture_matrix_v2_20261018/{B,C,D}` |
| 诊断代码、测试 | `eval/diagnostics/` |
| 诊断结果 | `exps/diagnostics_20261018/`（失败拆解、`probe_all/` 探针和曲线、`format_check`） |
| 矩阵配置 | `configs/matrix_v2/` |
| 唯一改动的现有文件和它的测试 | `data_gen/interiorgs_v10/training_prep.py`、`tests/test_output_format.py` |
| 文档 | `docs/transition_qa/`：本报告、`PROGRESS.md`、`P1_STATS.md`、`RECIPES.md`、`DIAG_BASELINE.md`、`CHANGES.md`、`probe_curve_baseline.png` |

测试合计 48 个，全部通过：`transition_qa` 17、`sft_mixture` 11、`eval/diagnostics` 9、输出格式 5，以及原有的 `test_training_prep.py` 6。

## 5. 复现

在工作树下执行。`PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python`；
`SFT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft`；`META=/mnt/umm/users/yinbaiqiao/InteriorGS_meta`。

```bash
# P1：候选题（CPU，12 进程约 7 分钟）、内部状态（约 5 分钟）、渲染（单卡约 13 分钟）、构建、统计、核对题
$PY -m data_gen.transition_qa.generate raw --sft-dir $SFT --scenes-root $META --out exps/transition_qa_20261018 --workers 12
$PY -m data_gen.transition_qa.interior plan --sft-dir $SFT --scenes-root $META --out exps/transition_qa_20261018 --workers 12
$PY data_gen/interiorgs_v10/pilot_sco.py prepare --pilot exps/transition_qa_20261018/interior --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS_ply
$PY data_gen/interiorgs_v10/pilot_sco.py submit --pilot exps/transition_qa_20261018/interior
$PY -m data_gen.transition_qa.interior questions --sft-dir $SFT --out exps/transition_qa_20261018
$PY -m data_gen.transition_qa.generate build --out exps/transition_qa_20261018 --split-manifest $SFT/split_manifest.json
$PY -m data_gen.transition_qa.report --out exps/transition_qa_20261018 --md docs/transition_qa/P1_STATS.md
$PY -m data_gen.transition_qa.review sample --sft-dir $SFT --out exps/transition_qa_20261018
# 混合配方
$PY -m data_gen.sft_mixture.build --config data_gen/sft_mixture/configs/mixture_matrix_v2_20261018.yaml
# 诊断
$PY -m eval.diagnostics.failure_breakdown --eval <评测目录> --out <前缀>
$PY -m eval.diagnostics.sco_probe --run <目录> --probe exps/transition_qa_20261018/p1_probe.jsonl --from-eval <评测目录> --names <结果目录名> --submit
$PY -m eval.diagnostics.format_check --eval <评测目录> --out <前缀>
```

## 6. 提交（分支 `transition-qa`，从旧到新）

`081465f2` 第 1 步文档；`4f3862af`、`d7447b46`、`1a19d536`、`e6b0bc6a` 第 2 步；`3322b8c2`、`2416c3be`、`f7c39f26` 第 3 步；
`c7a549f9`、`263ee8b7`、`c9e886be`、`39d5d941` 第 4 步；`2db60377`、`a842323a`、`6dcd8a1f` 第 6 步；`28849c0f`、`fea558ed` 第 7 步；
以及本报告。

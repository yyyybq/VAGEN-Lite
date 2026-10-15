# InteriorGS 空间问答：进度

本文件是这项工作的交接记录。每完成一个部分就更新，并提交、推送到 GitHub main。
任务说明：用户 2026-10-14 的提示词（构造式出题、样题审核网页），2026-10-15 的第二轮提示词（对报告第九节的决定；A1–A4；交回标注后的 B1–B5）。

## 1. 做完了什么

**第一轮（2026-10-14）**，详见 [`REPORT_20261014.md`](REPORT_20261014.md)：
- 出题代码 `data_gen/interiorgs_qa/`（单图 S1–S7、多图 M1–M11、视频 V1–V6）；样题 240 道（单图 70、多图 110、视频 60）；
  CPU 构造式普查、指代普查、统计扩量运行；盲答基线发现的 5 处答案偏差已修正并重跑。
- COVERAGE.md 第二版（92 个子任务编号全部有着落）。

**第二轮（2026-10-15）**，详见 [`REPORT_20261015.md`](REPORT_20261015.md)：
- 标注网页发布到 claude.ai：https://claude.ai/artifact/5RY1E9BeCYwSvaaesoAgKJ （240 道样题，标注存在页面的数据库里，
  集合 `labels`，每题一个文档；用 ArtifactData 读取）。生成：`site_artifact.py` + `site_artifact_template.html`。
- 用户决定落实：试验 F、G 和颜色试做不做，13 个子任务、BLINK 朝向类关系、MMSI 颜色改为"不做（用户决定）"
  （COVERAGE.md、REPORT_20261014.md 已改）；每张图记录俯仰角（`extra.pitch_deg`）。
- A1 训练格式：`export_lf.py`（Act-SFT 同一 LLaMA-Factory 格式，视频按多轮对话），`lf_tokens.py`（真实预处理计数）；
  单卡训练试跑（7B 前向反向、3B 全套配置）通过。产物 `exps/interiorgs_qa_lf_20261015/`。
- A2 评测：`evaluate.py`（vLLM 推理、scoring.py 打分、按组 / benchmark / 措辞组汇总，带题数和 95% 区间）；
  基础模型在 240 道样题上跑完：`exps/interiorgs_qa_eval_20261015/base_sample/`。
- A3 全量流水线：渲染前的同名物体 CPU 预筛（`views.same_name_visible`）、题库模式（`pipeline bank`，分片、断点续跑，
  不做答案平衡，物体上限）、`pipeline bank-merge`、抽样脚本 `sample.py`（预算、组间配比、答案上限、物体上限，
  训练集和评测集一起抽）；配置 `configs/bank_plan_20261015.json`、`configs/sampling_plan_20261015.json`。
  5 个 train 场景试跑：生成 → 抽样 → 导出 → 真实预处理全部通过（`exps/interiorgs_qa_bank_trial_20261015/`；
  加快预筛后的计时重跑 `exps/interiorgs_qa_bank_trial_20261015b/`，平均每场景 265 秒）。
- A4 评测题数估计：`estimate_eval.py`，结果在报告第五节。

## 2. 正在运行什么

没有。全量生成按要求没有跑。

## 3. 下一步

**现在：等用户交回标注。** 标注在网页里自动保存，不需要导出文件；用户说"标完了"后，我用 ArtifactData 读集合 `labels`。

**用户交回标注以后（B1–B5，计划）：**

- **B1 读标注**：每组按五项（能否认出物体、能否认出同一物体、标准答案是否正确、有没有歧义、256 下能否认出）分别算合格率，
  附题数和 95% 区间；列出整组性的问题（某组多数题同一项不合格）和集中出错的物体类别。
- **B2 校准和修正**：
  - M8（Around）的遮挡阈值：按"标准答案是否正确"和"有没有歧义"两项，比较远处物体可见比例和被挡比例不同的题，定阈值；
  - "能认出"的各项阈值（可见部分至少占画面 0.4%、露出至少一半、投影至少 60–70% 在画面内）：按"能否认出"一项，
    看哪些阈值附近的题被判不合格，调整；
  - 两条放宽规则（"只出现在一张图里"允许 0.2% 边角、M3 共享 1 个物体）：按"能否认出同一物体"一项定去留；
  - 标错的物体类别：加进不出题清单或改名；
  - 分辨率：按"256 下能否认出"一项，定训练用 256 还是 512。
- **B3 合格率低的组**：改代码，只重新渲染这些组的样题，发布到同一个网页给用户复核。
- **B4 用户确认后全量生成**：800 个 train、100 个 val、100 个 test 场景，`pipeline bank` 分片到多张卡（每片一个
  SCO 单卡任务），完成后 `bank-merge`。
- **B5 抽样、导出、汇报**：按第一节的决定（`sampling_plan`）抽出训练集和评测集，`export_lf.py` 导出，汇报每组题数、
  答案分布、图片输入次数和 token 数（`lf_tokens.py`）。

## 4. 约定

- 代码：`data_gen/interiorgs_qa/`（进 Git）。文档：`docs/interiorgs_qa/`（进 Git）。
- 运行产物（图片、深度、网页、题库、导出数据）：`exps/interiorgs_qa_*`（git 忽略）。
- 同一目录里另一条工作线（v10 训练）也在提交；本任务只用 `git commit -- <自己的路径>` 提交自己的文件，不碰别人的未提交改动；
  不改 Act-SFT 的代码和数据（只调用 `job_sco.py` 和 `sco_interiorgs_v10_train.sh` 的单卡试跑模式）。
- 共享缓存 `~/.cache/huggingface/datasets` 里有 root 拥有的文件：本任务的 CPU 预处理用 `HF_DATASETS_CACHE=/tmp/psv/hf_datasets_cache`。

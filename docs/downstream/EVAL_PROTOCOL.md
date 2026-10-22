# 下游线闭环评测协议

最后更新：2026-10-22（运行方式改为 1 卡分片；噪声测量和统计按你 10-22 的指令改）。适用于本线所有闭环评测：阶段 0 的 16 个检查点、阶段 1–3 的检查点，以及之后新增起点的评测。

依据：
- 决定 D6（2026-10-20）；
- 你对第二步报告的决定 1、3、4、6（2026-10-21）；
- 你 10-22 的分阶段指令（统计、噪声下限）；
- 第二步报告 [reports/STEP2_SMOKE_20261020.md](reports/STEP2_SMOKE_20261020.md) 3.2 节。

## 1. 固定评测计划

计划写在 [downstream/eval/eval_d6.json](../../downstream/eval/eval_d6.json)（计划名 `seen34_spatial50_v2`），每次评测都按它执行。

| 项目 | 规定 |
| --- | --- |
| 任务 | 34 个已见 target 任务：RoboCasa 1.0.1 注册表里的 18 个单步任务（atomic_seen）和 16 个多步任务（composite_seen） |
| 回合数 | 9 个空间关系任务各 50 回合，其余 25 个任务各 20 回合，共 950 回合 |
| 评测种子 | 空间关系任务 1000–1049，其余任务 1000–1019。评测种子和训练种子（1、2、3）分开 |
| 每个回合 | `gym.make("robocasa/<任务>", split="target", seed=s)` 后 `env.reset(seed=s)`；步数上限用 RoboCasa 给每个任务定的时限；策略每输出 16 步动作调用一次；成功就停 |
| 台面区域顺序 | 固定（第 2 节） |
| 分组汇总 | 全部 / 单步 / 多步 / 空间组 9 个 / 边界 3 个。边界 3 个不算入空间组，单独列出 |

- **空间组**（[downstream/demos/spatial_tasks.json](../../downstream/demos/spatial_tasks.json)）：CoffeeSetupMug、NavigateKitchen、OpenDrawer、SlideDishwasherRack、TurnOffStove、DeliverStraw、PrepareCoffee、SearingMeat、StackBowlsCabinet。
- **边界**：KettleBoiling、PackIdenticalLunches、SetUpCuttingStation。

**运行方式**（`downstream/eval/run_eval_ckpt.sh CKPT OUT 1 12 I/8`，由看板 `downstream/eval/watch.py` 提交；10-22 起）：
- 每个检查点 8 个 1 卡 sco 任务（分片），第 I 片跑计划里第 I、I+8、I+16… 个回合，约 119 回合。评测的瓶颈在 CPU 仿真，GPU 利用率不到 10%；每张卡配 14 个 vCPU，所以每片 12 个仿真进程、1 个策略服务，OSMesa 软件渲染。
- 只投到已有别的任务、还有空卡的机器上，不占空闲整机（STATUS.md 第 1.2 节）。
- 8 片都有结果后，报告把它们合成一次 950 回合的评测；缺任何一片都不算完成。
- 回合按任务时限从长到短的顺序跑，避免长回合拖在最后。
- 10-21 以前的方式（每个检查点一个 8 卡任务、24 个进程）没有产生过任何结果，已不用。

**每次评测记录**（`EXP/evals/<检查点>/shards/s<I>of8/out/`；阶段 0 在 `EXP/evals_run_stopped_20261022/`）：
- `plan.json`：检查点、策略服务、进程数、台面区域顺序开关 `stable_counter_regions`、计划名、全部回合；
- `episodes/episodes.jsonl`：每个回合一行，含任务、种子、成败、步数、所在进程，以及初始状态指纹（布局号、风格号、指令、关节位置哈希、首帧图像哈希）和全部关节位置 `init_qpos`；
- `episodes/summary.json`：总数、成功数、每个任务的成功率、平均每回合时间，以及开关和计划名；
- `episodes/frames/<任务>_s<种子>.jpg`：每个回合等间隔 8 帧的画面条（10-21 起），报告的范例用它。
- `gpu_util.csv`：策略服务的 GPU 利用率，每 5 秒一次。

## 2. 对 RoboCasa 行为的唯一改动：固定台面区域的顺序

### 2.1 为什么要改

- RoboCasa 1.0.1 的 `Counter.get_reset_regions` 用 `list(set(valid_geoms))` 给候选的台面几何体去重（`robocasa/models/fixtures/counter.py:813`）。这些几何体是 XML 元素对象，按内存地址算哈希，所以去重后的顺序每个进程都可能不同。
- 随后 `Fixture.sample_reset_region` 用 `self.rng.choice(valid_regions)` 按位置抽一个区域（`fixture.py:383`）。结果是：同一个种子在不同进程里会把设备或物体放到不同的台面上，站在设备前面的机器人也跟着换位置；有时还会因为摆放失败而重建场景，整个厨房布局都变了（`kitchen.py:604`、`826-851`）。
- 第二步的评测里，同一检查点、同一种子在不同运行中出现了不同的初始状态，最大的一个机器人起点差 1.65 米。这违反了"同种子两次重置的初始状态相同"（补充要求 2），也会给起点之间的配对比较加噪声。

### 2.2 改了什么

- `downstream/eval/closed_loop.py` 的 `stable_counter_regions()`：每个仿真进程启动时，把 `Counter.get_reset_regions` 换成一个副本。副本的代码、默认参数完全相同，只有一处不同：函数里用到的 `set` 换成按首次出现顺序去重（`with_ordered_set`）。所以 `list(set(...))` 得到的是 RoboCasa 自己生成候选列表时的先后顺序，不再随内存地址变化。
- 只在我们的评测进程里生效，不写任何文件。
- 默认开启。关闭方法：`closed_loop.py --no-stable-regions`，或在计划文件里写 `"stable_counter_regions": false`。每次评测的 `plan.json` 和 `summary.json` 都记录了开关状态。
- 单元测试：`downstream/tests/test_eval_and_experiment.py` 的 `test_with_ordered_set_keeps_first_seen_order_whatever_the_hashes`。

### 2.3 没改什么

- RoboCasa 的源码文件（`probe_spatial/robocasa365/robocasa/` 下没有任何改动）。
- 任务定义、物体和厨房资源、随机数生成器、任务时限、成功判定、动作空间、控制器、相机和渲染设置。
- 候选区域本身：哪些台面、每个区域的位置和大小都不变，变的只是它们在列表里的先后。原版下这个先后随进程变化，任何一种都可能出现；固定后只是始终取其中一种。所以评测场景仍在原版会出现的场景之内。
- 训练示教：示教是 RoboCasa 预先采好的数据，不经过这段代码。

### 2.4 扫描结果

做法：
- 34 个评测任务各取种子 1000–1002，共 102 对（任务, 种子）；
- 每对在 3 个新进程里各重置一次，原版和固定顺序各跑一遍；
- sco 任务 `pt-cb0kvvf0`（1 卡，12 个进程，2026-10-21 07:46–09:01 UTC）；
- 结果文件：`downstream_runs/checks/reset_sweep_20261021/out/sweep_{plain,stable}.json`；工具：`downstream/eval/reset_repeat.py`、`run_reset_sweep.sh`。

| | 原版 | 固定顺序 |
| --- | ---: | ---: |
| 3 次重置逐位相同的对数 | 89 / 102 | **102 / 102** |
| 布局、风格或指令变了的对数 | 2 | 0 |
| 有关节或物体位置差超过 1e-4 的对数 | 11 | 0 |
| 最大位置差：自由关节（物体）/ 其他关节（含机器人底座） | 2.01 米 / 2.05 米 | 0 / 0 |

原版不能复现的 13 对：
- 布局都变了：SteamInMicrowave / 1000、WashLettuce / 1000；
- 机器人起点变了：PreSoakPan / 1000、ScrubCuttingBoard / 1000 和 1002、TurnOnSinkFaucet / 1000；
- 物体换了位置：KettleBoiling / 1002、PickPlaceSinkToCounter / 1000 和 1002、PreSoakPan / 1002、RinseSinkBasin / 1002、SteamInMicrowave / 1002、TurnOnSinkFaucet / 1002。

此外，第二步的 NavigateKitchen / 1006 和 OpenStandMixerHead / 1002 在固定顺序后，各 4 个新进程的结果也全部逐位相同。

**局限**：每对只重复了 3 次。固定顺序后没有再发现差异，但这不能证明不存在别的跨进程差异来源。之后每次评测都记录全部关节位置，同一检查点评两遍的结果（第 3 节）可以再核对一次。

## 3. 评测噪声下限（10-22 起）

- 做法：阶段 1 的四个最终检查点，每个用同一批评测种子评两遍（第二遍在 `evals/<检查点>_repeat/`，同样 8 个 1 卡分片）。权重和种子都相同，两遍的差全部来自评测本身（仿真进程的调度、浮点运算顺序、所在机器）。
- 报告内容（`downstream/report/stats.py` 的 `noise_floor`）：
  - 合并差：第二遍减第一遍的总成功率差；
  - 逐任务差：34 个任务各一个成功率差，报平均绝对值和最大绝对值；
  - 逐任务平均差及对任务重抽样的 95% 区间；
  - 结果不同的回合数 / 比较的回合数。
- 用法：起点之间的逐任务平均差落在噪声下限的范围里时，判为分不出来。
- 10-21 定的 P = 48 / 96 并行度测试不再做：1 卡分片每片只有 14 个 vCPU，进程数固定为 12。

## 4. 统计（你 10-22 的指令）

- **主检验：逐任务配对**。配对单位是 34 个任务：同一任务、同一批评测种子（因而同一批初始场景，第 2 节）下，两个起点各有一个成功率，相减得一个差。报：
  - 34 个差的平均（单位百分点）和 95% 区间（对任务重抽样 2000 次）；
  - 差为正 / 为负的任务个数，和只看正负个数的符号检验（双侧精确二项）；
  - Wilcoxon 符号秩检验（差为 0 的任务不计）。
  - 分组（单步、多步、空间组 9 个、边界 3 个、其余 22 个）各算一遍，配对单位是组内的任务。
- **噪声下限**：第 3 节。
- **参考**：每个成功率的 95% Wilson 区间；逐回合配对的 McNemar 精确检验及 Holm 校正（每个任务一个检验）。它们把回合当成独立的，只作参考，不作结论。
- 代码：`downstream/report/stats.py`（`task_level`、`noise_floor`、`compare`），单元测试 `downstream/tests/test_report.py`。

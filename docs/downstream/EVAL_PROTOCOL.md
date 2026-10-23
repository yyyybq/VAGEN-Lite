# 下游线闭环评测协议

最后更新：2026-10-22 19:55 UTC（1 卡分片和整机 8 卡两种运行方式、速度和内存实测、坏进程处理与续跑、噪声下限只做 B）。适用于本线所有闭环评测：阶段 0 的 16 个检查点、阶段 1–3 的检查点，以及之后新增起点的评测。

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
- 投到哪里、何时让路，见 [REFERENCE.md](REFERENCE.md) 第 2 节。
- 8 片都有结果后，报告把它们合成一次 950 回合的评测；缺任何一片都不算完成。
- 回合按任务时限从长到短的顺序跑，避免长回合拖在最后。
- 10-21 以前的方式（每个检查点一个 8 卡任务、24 个进程）没有产生过任何结果，已不用。
- **10-22 16:30 起的另一种方式**（你要求用 h800 的空闲机器）：每个检查点一个 8 卡任务，8 个策略服务 × 96 个仿真进程（`run_eval_ckpt.sh CKPT OUT 8 96`，和阶段 1 的任务内评测相同），占满一台空机器；计划、种子、台面区域顺序都相同，结果在 `EVALDIR/out/episodes/`。
- **速度实测（10-22）**：一台机器的仿真总吞吐量约每秒 120 个控制步，进程多了每个进程就慢：24 个进程（8 个服务）每控制步 0.209 秒、每次重置 26.3 秒（第四步导出测试，24 回合）；96 个进程每控制步 0.77–0.81 秒、重置 60–64 秒（阶段 0 的 8 卡评测，前 70–98 回合）；1 卡分片（12 个进程）在忙机器上 0.64–0.72 秒、在较空的机器上 0.33 秒。一个检查点（950 回合，成功率低时大多跑满时限）约 3 小时，用 8 个分片或一个 8 卡任务都差不多。所以每回合时间不能用来比较两种进程数的效率（你 10-22 第二条指令第 1 条）。
- **仿真进程死了或卡住**（10-22 起）：`closed_loop.py` 记下每个进程正在跑的回合（`OUT/episodes/inflight/<pid>.json`），进程死了就把那个回合重跑（最多 2 次，仍不行记为错误）；一个回合跑了超过 600 秒 + 每控制步 3 秒（约最慢实测的 4 倍）的进程先被杀掉再重跑。在这之前，一个死掉的进程会让整个评测一直等（10-22 E0 B 10 万步的 3 个分片）。
- **重跑只补缺的回合**：同一个输出目录里已有相同计划（检查点、分片、台面区域顺序、回合列表）时，只跑还没完成的回合，`summary.json` 覆盖全部回合，`resume_<时间>.json` 记下续了多少。看板把 90 分钟没有新回合的评测删掉重提，就是用这个续跑。
- **内存**（你 10-22 第二条指令第 3 条）：每次评测每 30 秒记一次内存（`OUT/episodes/memory.csv`；`summary.json["memory"]`：任务容器的上限、峰值、峰值占比、内存不足被杀的次数、每个仿真进程的峰值常驻内存）。机器内存：整机 8 卡任务 1,920 GiB，1 卡分片 240 GiB。规则：整机评测的峰值超过机器内存的 80%（1,536 GiB）时，把进程数降到安全值。第一次实测（整机评测，96 个进程）的结果：**未测到**，下一个整机评测开始记录，结果出来后写在这里。

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

- 做法（你 10-22 第二条指令第 1、7 条）：同一检查点、同一批评测种子评两遍，权重和种子都相同，两遍的差全部来自评测本身（仿真进程的调度、浮点运算顺序、所在机器）。
  - 初步估计（E0）：B 的第 10 万步检查点，第一遍是 8 个 1 卡分片，第二遍是一个整机 8 卡任务（`evals_run_stopped_20261022/steps_100000_repeat/`），写进阶段 0 报告。
  - 正式的噪声下限（E1）：只做 B 一个起点。最终检查点第一遍是训练任务内的评测，第二遍是一个整机 8 卡任务（`evals/<检查点>_repeat/`），有空机时再做。
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

## 5. 剔除清单（你 10-23 的决定 1）

- **规则**：同一个（任务, 种子）在至少两个不同检查点的评测里让仿真进程死掉或挂住，就剔除。所有起点、E0 / E1 / E3 用同一份；**E1 的评测开始后冻结**，再加要单独报告。
- **清单**（[downstream/eval/excluded_episodes.json](../../downstream/eval/excluded_episodes.json)）：SearingMeat 1044、StoreLeftoversInBowl 1003、PreSoakPan 1016、DeliverStraw 1006、DeliverStraw 1014。每次评测 950 → **945 回合**；空间组 450 → 447（SearingMeat 和 DeliverStraw 是空间任务）。
- **怎么用**：
  - `closed_loop.py` 在分片之后去掉清单里的回合，所以一个分片的回合不变；清单和它的 sha256 记在每次评测的 `plan.json`、`summary.json` 和 `resume_*.json` 里。
  - 报告（`downstream/report/collect.py`）和看板（`watch.merged`）不计清单里的回合，不管评测有没有记下它们。清单之前完成的评测里，这 5 个回合大多也跑完了（全部失败），一律不计。
- **证据**（10-23 00:05 UTC 为止的 9 个评测）：
  - 每个评测第一次跑完时缺的回合，以及同一次运行里进程死掉或卡住、重跑过的回合（`attempt` ≥ 1）；
  - 来源：各评测的 `episodes.jsonl` 和 `resume_*.json`；
  - 明细见 [reports/E0_PARTIAL_20261023.md](reports/E0_PARTIAL_20261023.md) 第 5 节。
  - 出事的评测个数：SearingMeat 1044 是 7 个，StoreLeftoversInBowl 1003 是 6 个，DeliverStraw 1006 是 5 个，DeliverStraw 1014 是 4 个，PreSoakPan 1016 是 4 个。
  - 按同一规则，到这时还有 5 个也满足：LoadDishwasher 1002（4 个）、PrepareCoffee 1003（3 个）、OpenDrawer 1012、StackBowlsCabinet 1012、StirVegetables 1007（各 2 个）。它们是否加入，**等你定**；没定之前按上面 5 个执行。
- **原因不是内存不足**：
  - 整机评测（96 个进程，机器 1,920 GiB）内存峰值 554–631 GiB（29%–33%），单个仿真进程最多 6.3 GiB；
  - 有内存记录的评测里，内存不足被杀 0 次；
  - 这些回合在同时只跑几个回合的重提里都正常完成；在满载的整机评测里，第一次的进程死掉后重跑一次就完成。
  - 进程死掉的原因未核实，最可能是仿真或渲染本身崩溃。

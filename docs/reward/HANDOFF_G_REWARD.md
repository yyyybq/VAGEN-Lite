# G 组的指认奖励怎么给：奖励线的教训和通道说明（2026-11-06）

给矩阵线，经用户转。奖励线分支 reward-design。文中代码行号注明了分支：reward-stmt 的 aab6fadd，或 matrix-v3 的 fba7057a。

**一句话：** 指认得分不要加进整局回报，只给坐标 token 一个单独的优势；以动作合法为前提；按 token 数归一；守护加三条只报不停的 1% 预警。

**先要用户定的一件事：这份改法和 G 的预登记冲突。**
- G 的预登记第四版（transition-qa 49b54c54 的 `docs/transition_qa/PREREG_G_vs_D.md`，11-06 定稿）写的是：
  - 指认得分"乘 0.1 加到原有奖励上"，S0、V4 两侧都加；
  - V4 补题时扣掉每局的指认奖励；
  - 旧格式（整段只有一个动作段）"也认、动作照常执行，不算失败"，也拿不到指认奖励；
  - "定稿后不改指标、题集、检查点和判定规则……另加的分析一律标'探索性'"。
- 下面第 1、2 条改的正是前三条。照做，要在 G 的 RL 开跑前由用户批准并改预登记；不改预登记的话，只能另开一臂，不能算作 G。
- G 现在的写法（matrix-v3 fba7057a）就是预登记那一套：每回合把 0.1 × 指认得分加进环境奖励（`env.py` 第 1202 行）。

用语：
- **指认得分：** 一回合 `<targets>` 段里，任务的每个目标"点在框内"算对；看不清的目标写 none 算对。"看不清"是按成功判据的可见规则不算清楚可见，露出一部分但不够的也算。看不清的目标给了点记 0。取平均（`pointing.judge_turn`）。
- **坐标 token：** 一回合回答里 `<targets>…</targets>` 那一段的 token。
- **E：** 一局的整局优势，即同题 8 局回报比较的 verl GRPO 优势，S0 / V4 原来就有的那个。
- **z：** 一回合的指认得分减去本批全部回合的均值，再除以标准差。

## 1. 不加进整局回报，只加在坐标 token 上（B 的结论）

- **B 的结论：** 把进度分换成真实剩余步数（O）、加进整局回报以后，在被训练的有成有败题上，O 的组内优势和 S0 方向几乎一致（夹角余弦平方的平均 0.98–0.99），独有信号只占 1.1%–1.8%。成功率也没涨：O 对 V4 +1.0 个点（只有种子 1，逐题 79 / 72，p 0.63；不是证明两者相等）。原因是成败的回报差约 5 分，压住了每回合零点几分的项。
- **放到 G 上：** 指认项一局最多加 0.1 × 回合数，每局约 6–7 回合，也就是最多约 0.7 分，同样被 5 分压住。
  - S0 一侧的 mixed_outcome 只留有成有败的组：这一项（最多约 0.7 分）改不了成败之间的先后，只在成败相同的几局之间分先后，在组内优势里占得很少（照 B：O 项同样加在回报里，独有信号只有 1.1%–1.8%）。
  - 它真正起作用的只剩成败相同的组，也就是 V4 按回报分散度补进来的全失败题。而在那些组里，它推的是整局所有 token：动作 token 也按"指得准不准"被推。
- **改法：**
  - 环境的 `pointing_reward` 设 0。格式示例是 `targets_*` 时，环境照样判、照样记（`_pointing_on()`），得分在 `info["pointing"]`。
  - 坐标 token 的优势 = E + w × z′（z′ 见第 3 条）；其余 token 只有 E，和 D 组的 S0 / V4 一样。这就是方向 3 rw_st1 的做法：`turn_credit` 的 statement 通道，`episode_norm: none`。
  - V4 一侧有一处不同：keep 补进来的组，整行优势按回报分散度缩小，坐标项也跟着一起缩（matrix-v3 fba7057a 的 `custom_filter/filter.py` 第 279–290 行；打完下面的补丁以后是第 313–324 行）。
  - V4 的 `keep_std_exclude: pointing_return` 随之不需要了，回报里已经没有指认项。
- **w 要另定：** rw_st1 用 0.5，也就是陈述 token 上加 0.5 × z。G.yaml 的 0.1 是加在回报里的分值，不是同一个量。

## 2. 以动作标签存在且合法为前提，否则记 0（方向 3 的"写完就停"）

- **rw_st1 的教训：** 陈述得分不看有没有动作。第 115–120 次更新，读不出率（日志口径：每局读不出回合占比的平均）从 0.26% 涨到 13.6%，约每次翻一倍；第 120 次按回合数是 227 / 1,138 = 19.9%。这 227 个读不出的回合全是 `</observe>` 后直接结束、没有 `<action>`，其中 219 个陈述四项写全，8 个缺项。第 130 次被 10% 线停下。
  - 可能的原因（没查证）：陈述项只加在陈述 token 上（它们同时有整局优势），结束符只拿整局优势，所以这种回答的陈述部分照样被往上推。
- **G 的环境已经这样判：** matrix-v3 fba7057a 的 `env.py` 第 1199 行把 `action_executable and strict_format_correct` 传进去，第 1527 行调 `pointing.judge_turn(..., readable=...)`。没有能执行的动作，或者回答不是"targets 段 + action 段"的短结构，得分就是 0。
- **通道里接着守：**
  - 判为不合格的回合（`info["pointing"]["ok"]` 为假：没有合法动作，或不是短结构），单独优势加在整个回答上，结束符也算在内，不只加在坐标段上。这样"写完坐标就停"的结束符也被压。陈述通道对读不出的陈述就是这样做的：`turn_credit.turn_record` 里没有陈述范围时，取整个回答。
  - 只写动作、不写坐标的旧格式回答，也按这一条整段压下去。这一条改的是预登记"旧格式不算失败"那一句，要用户定（见开头）。
  - 合格但点错的回合，范围仍是坐标段，值是它的得分。

## 3. 按 token 数归一（方向 1 的"塞动作"）

- **rw_tc1 的教训：** 逐回合项每回合给一个值，却加在这一回合的每个 token 上。
  - 回合的值随动作数涨，token 数也随动作数涨（7 + 3 × 动作数：1 个动作 10 个 token，5 个 22 个）。一回合的总拉力是两者相乘：5 个动作 +7.9，1 个动作 −1.1。
  - 再加上整局项也在压 1 个动作的回合，每回合的动作数从 1.7 涨到 3.1，成功率比 V4 低 11.8 个点。
- **坐标段的长度也会变：** Qwen3-VL-8B 的分词器把数字逐位切开，"(120,88)" 是 8 个 token，"none" 是 1 个。整段算（用分词器实测）：
  - 一个目标：写坐标 9 + 坐标位数个 token，0–1000 的坐标一般 3 + 3 位，就是 15 个；写 none 8 个。
  - 两个目标：都写坐标 14 + 坐标位数个，一般 26 个；都写 none 12 个。
  - 不归一，写坐标的回合拉力大，正负都放大。
- **改法：** 坐标段每个 token 的值 = w × z × L̄ ÷ L。
  - L 是这一回合加单独优势的 token 数，L̄ 是本批全部回合的平均。
  - 这样每回合的总拉力只看得分、不看长度，平均每个 token 的大小不变。
  - G 的预登记第 3.6 节已经记着"指认奖励按回合发，多分几个回合拿得多"，要报回合数。挪到坐标 token 上、按 token 数归一以后，每回合的拉力还是按回合给，这一点不变。

## 4. 读不出 10 次均值过 1% 就预警，加进 G 的守护

- **依据：** 被 10% 线停下的运行里，1% 预警比过 10% 早了多少次更新。口径是训练日志里的 `transition/invalid_action_rate/mean`，也就是守护和停止线读的那个：
  - C0 SFT e8 起点的 5 个：早 18–142 次（q3s0 18、rw_s0 67、qw3dcks3 77、qw3s0s4 87、qw3nks2 142）；
  - c3、weak 起点的 2 个：各早 23 次；
  - 方向 3：rw_st0 第 109 次过 1%、第 122 次过 10%；rw_st1 第 119 次、第 129 次，只早 13、10 次。
  - 矩阵 Cr 起点的 2 个（`sft/Cr` checkpoint-2107，守护和 G 相同）：m3crs1s0 第 245 次过 1%、第 270 次过 10%，早 25 次；m3crs1d 第 221 次过 1%，第 231–264 次回到 1% 以下，第 265 次再过，第 286 次过 10%，按第一次算早 65 次，按再过的那次算早 21 次（预警会响了又消）。
  - 坏得快的时候预警只早一二十次，所以响了就要马上看回答。
  - DRIFT_S0 写的"早 45–150 次"是它按回答逐个重判的口径，比日志口径过 1% 早。
  - 熵不能作预警：到 11-01 的 34 个 Qwen3 运行全都到过前 50 次均值的 1.5 倍，包括不走样的。
- **G 现在的守护只有停止线**（vars.yaml 第 23 行：invalid_max 0.10、kl_max 1.0、entropy_max 0.3、entropy_window 5、vision_token_crashes 2，由 matrix_plan.py 第 341 行写进计划；G.yaml 第 41 行只是照抄，计划不读）。建议加三条只报、不停的，都是最近 10 次平均 > 1%：
  - `transition/invalid_action_rate/mean`：读不出动作；
  - `transition/targets_bad_rate/mean`：judge_turn 判为 bad 的回合，即坐标段坏了、有自由文字、或没有能读的动作（matrix-v3 fba7057a 的 `pointing.py` 第 226–228 行）；
  - `transition/targets_old_rate/mean`：只写动作、不写坐标的旧格式回答。bad 不含旧格式，前两条都抓不到"不再写坐标段"这种走样。bad 和旧格式的回合得分都是 0。
- **写法：** 照 `train_chain.guard_reason` 里读不出那一段（matrix-v3 fba7057a 的 train_chain.py 第 380–385 行），用同一个 10 次窗口，阈值 0.01，返回的是报警而不是停止原因。本线的 `reward/stmt_watch.py`、`reward/tc_watch.py` 是另一种写法：单独读 RL 日志。
- **开关放在哪：** 不要加进 G.yaml 或 vars.yaml 的 guards。G.yaml 的 guards 计划不读，而且 test_matrix_g.py 第 31–32 行要 G 的 guards 和 D 的相同；vars.yaml 的 guards 有 test_matrix_v3.py 第 78 行要它等于登记的。放在 train_chain 的默认里，或由 matrix_plan 给 G 的计划另加一个键。
- **熵那条线另有一件事：** G 的预登记第 9 节（transition-qa 49b54c54 第 110、115 行）定了 RL 的熵停止线只按动作段算（`actor/entropy_action`），没加之前不起 G 的 RL。fba7057a 上还没有：train_chain.py 第 404 行读的是整段的 `actor/entropy`（ray_trainer.py 第 2617 行算）。要动的 gym_agent_loop.py、ray_trainer.py、train_chain.py 也是第 5 节的补丁改的文件，两件事一起做时注意先后。

## 5. turn_credit 的 statement 通道改成坐标 token

**先搬代码。** matrix-v3 上没有 turn_credit。
- reward-stmt 上的 4 个提交 4d525b00、541f9a74、1cecf88e、c51a2c66，在 fba7057a 的临时克隆里依次搬过：
  - 只有 `gym_agent_loop.py` 有冲突：matrix-v3 在这两处先后加了 E 组的 think_turns（548a7c06，10-19）和 G 组的指认计数，补丁也在同一处加行，两边都留即可（AgentData 的计数字段之后、每回合 `end_turn` 之后）；
  - `docs/reward/`、`reward/start_chain_b.sh`、`vagen/statement_truth.py` 在 matrix-v3 上没有，不搬。statement_truth 只有陈述项用；陈述专用的 `reward/tests/test_statement_credit.py`、`reward/statement_eval.py` 也去掉了。
  - 0002 的提交说明列了新文件 `data_gen/interiorgs_v10/distance_table_near.py`，补丁里没有：matrix-v3 上已经有，和 541f9a74 的逐字节相同。
- 搬完以后跑了 `data_gen/interiorgs_v10/tests`、`reward/tests`、`tests/active_spatial`、`data_gen/sft_mixture/tests`：578 个通过、3 个跳过、0 个失败（conda 环境 vagen-lite 的 Python）。一直失败的 starVLA 测试 `tests/test_starvla_transfer.py` 不在这四个目录里。
- 现成的补丁在 `reward_runs/handoff_g/turn_credit_onto_fba7057a/`，共 4 个，附说明。用 `git am` 打到 fba7057a 上，得到的代码和测过的逐字节相同（树 d828a093）。
- **G 的链在跑时不要在 VAGEN-Lite-matrix 里打补丁：** m3_g_sft 的计划提交是 fba7057a，评测任务读这个 worktree 的代码。在另开的 worktree 或分支上打，或者等链停了再打。

**要改的行**（照 statement 的写法加一个 "targets"）：

| 文件 | 行 | 现在 | targets 照着写 |
| --- | --- | --- | --- |
| vagen/turn_credit.py（aab6fadd） | 53，`TURN_TERMS` | ("shaping", "statement") | 加 "targets" |
| | 75–76，`settings` | statement 不收 turn_reward、stall_penalty | targets 同样 |
| | 105–128，`turn_record` | 读 `info["statement"]` 的 accuracy；可读时 statement_span 是陈述段，否则是整个回答 | 读 `info["pointing"]`：targets_score = score；ok 时 targets_span 是坐标段，否则是整个回答（第 2 条） |
| | 140–142，`episode_record` | 按回答长度截 statement_span | 同样截 targets_span |
| | 161–162，`turn_values` | 值 = statement_accuracy | 值 = targets_score |
| | 315，`token_advantages` | 用 statement_span | 用 targets_span |
| | 324–327 | `adv[i, s:e] = turn_weight × a` | 再乘 L̄ ÷ (e − s)（第 3 条） |
| vagen/statement_truth.py（aab6fadd） | 185–198，`statement_token_count` | 二分找最少几个 token 能解码出 "</observe>" | 同样的二分，找 "</targets>"；要放进 pointing.py，因为 statement_truth.py 没搬 |
| vagen/agent_loop/gym_agent_loop.py（aab6fadd） | 477–490（打完补丁 513–526） | 陈述可读才算陈述段 | `info["pointing"]["ok"]` 才算坐标段，传给 turn_record |
| data_gen/interiorgs_v10/training_prep.py（aab6fadd） | 803–804（打完补丁 827–828） | 只要给了 `--turn-credit`，有密集奖励预设就拒绝 | 改成只对 turn_term "shaping" 拒绝。G 的 V4 一侧是密集奖励，不改这里 V4 跑不起来 |
| | 805–809（打完补丁 829–833） | statement 要 `--statement`；其余的项都要距离表（808–809；打完补丁 832–833） | 加一个 targets 分支：要 `targets_*` 格式示例、pointing_reward 为 0，不走查距离表那一条 |
| configs/matrix_v3/G.yaml（fba7057a） | rl 块 | `pointing_reward: 0.1`、`keep_std_exclude: pointing_return` | 去掉这两项，加 `turn_credit: {turn_term: targets, episode_norm: none, turn_weight: W}` |
| data_gen/sft_mixture/matrix_plan.py（fba7057a） | 337–356 | G 的 RL 计划从 G.yaml 只取顶层的 format_example（第 340 行）、rl 块的 pointing_reward（第 342–343 行）和 keep_std_exclude（第 351–353 行），另从 eval 块取 Pg 探针（第 356 行）；其余设置来自 vars.yaml | 再取 rl 块的 turn_credit，两侧都给；搬过来的 train_chain 会把计划 rl 里的 turn_credit 变成 `--turn-credit` |
| data_gen/sft_mixture/tests/test_matrix_g.py（fba7057a） | 29、68、73、79、82–85 | 断言 G.yaml 和两侧计划的 pointing_reward 是 0.1、rl-config 带 `--pointing-reward 0.1`、V4 一侧带 `--keep-std-exclude pointing_return` | 改成两侧都带 turn_credit（turn_term targets），不带 `--pointing-reward`（或为 0），也不带 `--keep-std-exclude`；不改这组测试会失败 |

- 环境不用改。rl-config 给 `--turn-credit '{"turn_term":"targets","episode_norm":"none","turn_weight":W}'`，`--pointing-reward` 不给或给 0。
- **注意现在的顺序：** ray_trainer 先算优势（matrix-v3 fba7057a 第 2842 行），后按组筛（第 2870 行，打完补丁是第 2871 行）。
  - z 按筛前全部候选题的回合算；筛掉的组，坐标项也跟着没了。
  - S0 一侧的 mixed_outcome 只留有成有败的组。rw_st1 也是这样（mixed_outcome，target_groups 12）。

**怎么核对**（照 `reward/statement_dump_check.py`，reward-stmt aab6fadd）：
- 冒烟时，turn_credit 的设置加 `dump_dir` 和 `dump_updates: 3`。估计器会把前 3 次更新每个样本的记录、token 和优势写出来（`vagen/custom_advantage/turn_credit.py`）。
- 检查脚本不调 turn_credit，只用 numpy 重算，逐回合查三件事：
  - 坐标段从回答开头起，解码出来是 `<targets>…</targets>` 再加合并 token 带出的 `<`：`</targets><action>` 里的 `><` 是一个 token，归坐标段；其余部分有 `action>`、没有 `<targets>`（statement_dump_check.py 第 111 行查的也是 `"action>"`）；不合格的回合，范围是整个回答；
    - 这个 `><` 也是 G 预登记第 9 节动作段的第一个 token，坐标项会落在它上面。要不要把它让给动作段，由矩阵线定；
  - 坐标段每个 token = E + w × z × L̄ ÷ L，其余 token = E（verl GRPO：组内均值、n − 1 标准差、eps 1e−6），差超过 1e−4 记一处不符；
  - 这样每回合坐标段（不合格的回合是整个回答）上 (优势 − E) 的和就是 w × z × L̄，上一条逐 token 的核对已经含了。再按坐标段的 token 数、按动作数分组，报 z 的均值和每回合的总拉力作参考，不要求持平：随动作数变，可能只是得分和动作数相关，不是代码错（rw_sts1 冒烟的陈述项，1 个动作 −1.9、5 个动作 +7.9，0 处不符）。
- 脚本要改的地方：
  - `assert` 里的 turn_term 改成 "targets"（第 63 行）；
  - 记录的键 statement_accuracy / statement_span 改成 targets_score / targets_span（第 78、82、89、93、97、99、130 行）；
  - "<observe>" / "</observe>" 改成 "<targets>" / "</targets>"（第 108–112 行）；
  - 期望值乘 L̄ ÷ L，L̄ 每次更新算一次；
  - 加按坐标段 token 数的分组。现在只按动作数分组（第 128–132 行）。
- **参照：** rw_sts1 冒烟（rw_st1 的设置，前 3 次更新）576 局、3,718 个回合，0 处不符，和离线重算最大差 4 × 10⁻⁶。`reward/turn_credit_dump_check.py` 在同一份转储上只报 span_adv_not_constant，这是一个回答里有两段不同优势的正常结果。
- **那次冒烟没走到"整个回答"那条路径：** 3,718 个回合的陈述全部读得出，只有没搬的 `reward/tests/test_statement_credit.py` 测过它。G 的不合格回合和旧格式回答（第 2 条）全走这条路径，所以 G 的冒烟转储里要有 ok 为假的回合，并核它们的范围；最好补一个单元测试。

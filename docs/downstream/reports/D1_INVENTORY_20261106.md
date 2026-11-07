# D1 第一阶段盘点报告：VLAKit 能否挂冻结的 Qwen3-VL-8B

> **已归档（2026-11-07）。** 你 11-07 的第 2 条："D1 暂停，留到论文之后；盘点报告归档。"
> - D1 不再往下做。本文"要你定的事"里的 6 件不再等你定，留到论文之后重开时再看。
> - 正文照原样保留。到那时要先复核：文中引的 VLAKit 代码和集群文件，到时可能已经变了。

写于 2026-11-06 约 22:00 UTC（核查一轮后定稿）。这一步只读了代码，以及集群上的配置和统计文件。没有训练，没有占卡，没往集群拷任何东西。VLAKit 的东西一样都没运行、导入或安装。

标注说明：
- 〔码〕：读代码核实过，但没运行。
- 〔档〕：只在文档里写着。
- 〔测〕：来自我们集群上的实测或文件。
- 〔推断〕：从代码或实测推出来的。
- 〔未核实〕：没找到依据。

VLAKit 的路径都相对 `ext/NEO/VLAKit/`，提交 57e21d6。本地克隆在 `C:\Users\user\Desktop\code\ext\NEO\VLAKit`，是按你批的"只拉代码"做的，只读过。

正文只放结论、要你定的事、两张任务表、任务书问题的一行一答和前五条风险。只和丙有关的细节、各项的代码依据和来源，都在最后的附录 A1–A8。

---

## 结论先说

- **停在这里，等你定。** VLAKit 挂不上 Qwen3-VL-8B，整套适配约 12 个工作日，超过任务书第 5 条的两天线。第一阶段到此为止：没训练，没占卡，没往集群拷任何东西。你在"要你定的事"第 1 项选定路线之前，我不动 D1。如果选了甲，适配做下来超过两天，我会再停下来报你。
- **挂不上，要写代码。**
  - VLAKit 只支持 NEO 自己的 2B 主干〔码〕。
  - Qwen3-VL-8B 的文件夹在读配置这一步就报错〔码〕。
  - 只有"专家接主干哪几层"（layer_map）改配置就行。主干、视觉塔、位置编码和掩码、权重加载、图像 token、专家的旋转编码都是 NEO 专用的代码〔码〕。
- **适配工作量。**
  - 整套走 VLAKit（丙）约 12 个工作日（9–15）〔推断〕。
  - 光主干一侧（A1–A6）约 5 天（4–6）〔推断〕，已经超过两天线。
- **推荐的备选：StarVLA 里现成的 QwenPI_v3（甲）。选它就是改第二阶段写定的设计，要你明确同意。**
  - 任务书第二阶段写的是"VLAKit 的流匹配动作专家，读主干所有层的键值"。甲有两处不同：一是专家换成 StarVLA 自己的 36 层流匹配 DiT；二是读的是每层输出的隐状态，不是每层注意力的键值。隐状态每层先过一个 LayerNorm+Linear，从 4096 维压到 1024 维〔码〕。
  - 读法也不同。甲默认每个 DiT 块只有一个交叉注意力，查询是 32 个可学 token 加 16 个动作 token，动作 token 之间互不注意（`QwenPI_v3.py:126-133`，`cross_attention_dit.py:240-243, 298-318`）。打开交替自注意力，奇数块就改成自注意力，只读 18 层。VLAKit 的代码则把主干每层的键值和专家自己的键值放进同一个 softmax〔码〕。论文本身没读。
  - 所以任务书立论用的 NEO-VLA 证据（冻结时差 11.4、全微调后差 4.5）搬不到甲上。甲回答的问题是：StarVLA 的逐层流匹配头在冻结主干上学不学得动，四个起点差多少。
  - 要和原设计一致，只能选乙（约 3.5–4.5 天）或丙（约 12 天），两条都超过两天线。
- **还没有证据表明，冻结主干上的任何头能过 15% 闸门，甲也一样。**
  - X1 的回归头训到 5 万步，四个起点都不到 2%〔测〕。NEO-VLA 的证据搬不到甲上。200 步的试跑也回答不了这个问题。
  - 阶段 1（底座种子 1，约 210 卡时）就是这个答案的价钱。闸门不过就停，花掉的就是这约 210〔推断〕。
  - 想早点看出苗头，可以在训练中途加评一次，约 30 卡时，但这也改了固定设计，要你另定〔推断〕。
- **一次训练的卡时。** 设定为：主干冻结、只训头、输入同 StarVLA（三相机 224，不用历史帧和本体状态）、20 万步 × 批 64、8 卡。
  - 乙、丙（只训专家）：训练约 175 卡时（160–190）〔推断〕；VLAKit 引擎的开销没量过，另有约 ±30% 的不确定。
  - 甲：看训练时每个样本复制几份去加噪。设 2 约 160–210，用代码缺省的 16 可能要 200–490〔推断〕。
  - 评测 943 回合约 30 卡时（26–33）〔推断〕。
  - 每个起点合计约 205〔推断〕。
- **300 卡时线。** 按推荐设置，单次训练约 175 卡时，加评测约 205，都不到任务书"超过 300 卡时的训练"那条线〔推断〕。阶段 1（一次训练加试跑）约 210；阶段 2、3 各有两次训练，每个阶段合计约 410。有两种设置会让单次训练过线或可能过线：丙用 VLAKit 自己的布局（训练约 380–450），一定过；甲的复制数用缺省值 16（约 200–490），可能过。
- **分阶段的总卡时**〔推断〕：
  - 先做试跑，约 3–6。
  - 底座没过 15% 就停：约 210。
  - 做到 V4 加底座第二个种子：约 620。
  - 四个起点全做：约 1,030（930–1,120；如果引擎开销按 ±30% 算，范围放宽到约 750–1,300）。
- **能并行。**
  - 推荐设置下，一次训练放得进一张 80 GB 卡：只训专家时每卡 8 个样本约 43 GB，甲复制数设 2 时约 30 GB〔推断〕。
  - 四个起点加底座第二个种子，共 5 次训练，全部并行要 40 张卡，约 1 天〔推断〕。
  - 按分阶段做，最多同时跑两次训练，16 张卡〔推断〕。

---

## 要你定的事

### 1. 走哪条路（任务书第 5 条的停点）

| 选项 | 适配 | 每起点卡时（训练 + 评测） | 要你另批的 |
|---|---|---|---|
| **甲（推荐）StarVLA QwenPI_v3** | 约 1.5–2 天〔推断〕 | 约 205（复制数设 2；设 16 约 230–520）〔推断〕 | 改第二阶段设计：专家换成 QwenPI_v3（读每层隐状态，只用交叉注意力），不用 VLAKit 的专家 |
| 乙 把 VLAKit 的专家照着写进 StarVLA（我们自己的代码） | 约 3.5–4.5 天〔推断，未核实〕 | 约 205〔推断〕 | 无（只照着重写；如果要整段照搬 VLAKit 源码，要你批） |
| 丙 整套 VLAKit | 约 12 天（9–15）〔推断〕 | 约 205（按丙的推荐设置）〔推断〕 | 把 VLAKit 拷进集群；装 hydra-core，装进新建的环境，不动评测正在读的 StarVLA 环境；下载 HF 上的 manifest 样例，这一件超出你批的"只拉代码"，还要拷进集群 |
| 丁 任务书点名的备选：StarVLA 冻结主干 + 回归头（X1 换成 Qwen3-VL-8B） | 约 0.5 天〔推断〕 | 约 175（按 X1 每步 0.32 秒推）〔推断〕 | 无 |
| 戊 D1 停在盘点 | 0 | 0 | 无 |

- **甲**
  - 这个头就在我们钉住的 StarVLA（4507931）里，我们的补丁没动它〔码〕。
  - 数据管线、16×12 动作、min-max 归一化、三相机、闭环服务都照旧，和 E2、T2、X1 在同一套栈里比〔推断〕。
  - 和原设计的差别见结论。要定的设置和适配拆分见第 2 项。
- **乙**
  - 机制和 VLAKit 代码一样：专家每层把主干的旋转后键值并进同一个 softmax。
  - 用 HF 自带的 Qwen3-VL 前向取键值，所以位置编码和 DeepStack 不用重写〔推断〕。前向时要显式传 `use_cache=True`（附录 A3）。
  - 数据、归一化、评测协议都留在 StarVLA，不用转数据，也不用新协议〔推断〕。
  - 代价：要手写专家层，专家的旋转编码约定得自己对齐；流匹配损失和采样器可以复用 StarVLA 现成的〔推断〕。
  - 约 3.5–4.5 天，也超过两天线。照 VLAKit 的代码重写进我们自己的框架。如果有整段照搬，按 Apache-2.0 保留版权声明，并且先问你。
  - 天数是我综合各项的估计。没人逐行查过 StarVLA 框架接口〔未核实〕。
- **丙**
  - 最贴近任务书原文。
  - 代价分五块：主干和视觉约 5 天，数据转换约 3 天，动作侧补丁约 0.5 天，新策略服务约 2 天，环境、作业包装和试跑约 1.5 天，合计约 12 天。
  - 要你批的三件见表。运行前还要对拷进集群的副本做代码审查，审查时间没估。
- **丁**
  - X1 在四个 Qwen2.5 起点上都不到 2%〔测〕，不过 X1 只训了 5 万步，第二阶段是 20 万步。换成 Qwen3-VL、训到 20 万步，底座过 15% 闸门的可能仍然很小〔推断〕，不推荐。
- **戊**
  - 问题仍然没测。好处是不占卡、不推迟收口，论文里写明冻结主干的流匹配探针没做。

**推荐：甲。** 先做试跑（8 卡 200 步加 1 卡约 50 步，含评测约 3–6 卡时）。如果你要和任务书原设计同样的机制，也就是 VLAKit 代码里的联合 softmax 专家，选乙，不选丙。

### 2. 选甲要定的设置（四个起点相同，开跑前定死）

- **训练时每个样本复制几份去加噪**（`trainer.repeated_diffusion_steps`）。
  - 配方里不写，就取代码缺省值 16（`QwenPI_v3.py:315-317`）〔码〕。我们的 `recipe.yaml` 没写这一项。StarVLA 自带的 QwenPI_v3 例子用 2。
  - 复制的是动作和 36 层投影后的隐状态。DiT 在 float32 autocast 下算（`QwenPI_v3.py:308`），StarVLA 没开 TF32〔码〕。
  - 设 2：一次训练约 160–210 卡时，每卡 8 个样本约 30 GB。设 16：DiT 的计算量和冻结主干前向相当，一次训练可能要 200–490 卡时，有可能单次就超过 300 卡时线，显存约 55 GB〔推断〕。
  - 推荐设 2，试跑时把 2 和 16 的每步耗时都量一下。
- **推理去噪步数。** 缺省是 4 步 Euler（`QwenPI_v3.py:111`，`LayerwiseFM_ActionHeader.py:365-407`），VLAKit 用 10 步。推荐沿用 4，试跑时报 4 步和 10 步的时延。4 步够不够还没核实。
- **注意力结构。** 保持缺省的全交叉注意力（`interleave_self_attention=false`），这样 36 层都读。
- **学习率。** 36 个投影层不在 `action_model` 里面（`QwenPI_v3.py:227-239`），按我们的配方会拿 base 学习率 2.5e-5，DiT 拿 1e-4（`trainer_tools.py:127-146`，`recipe.yaml:48-51`）。推荐投影层和动作头用同一个学习率。
- 甲可训练的参数约 0.69B：DiT 和动作编解码约 0.54B，36 个投影层约 0.15B〔推断，按代码里的形状算〕。这是 X1 回归头的 5.3 倍，和 VLAKit 接 36 层的专家（约 717M）差不多。
- **看代码已经查清两件。**
  - ①冻结主干的前向不建计算图：虽然没包 no_grad，但主干参数全部 requires_grad=False，输入是 token 和像素，训练入口也不开梯度检查点，所以计算图从投影层开始（`QwenPI_v3.py:256-273`，`trainer_tools.py:222-224`）。X1 走同一条调用路径，只前向时显存峰值 25.4 GB，T2 反传时约 61 GiB（gpu.csv 62,520 MiB），可以作旁证〔码 + 测〕。
  - ②去噪步数缺省 4 步，可以配置。
  - 还没核实的有三件：8B 下的显存峰值和每步耗时（都取决于复制数）；4 步够不够；DeepSpeed ZeRO-2 下不用的参数（`proj_out_1/2`；state_dim 不置空时还有 state_encoder）会不会出问题。这三件都在试跑里查。
- **甲的适配拆分：约 1.5–2 天〔推断〕。**
  - ①新 recipe（QwenPI_v3、LayerwiseFM、去掉 state_dim、复制数、去噪步数、投影层学习率）和 make_experiment 的开关，约 0.5 天。
  - ②train_entry 的初始化哈希加上 project_layers（现在只哈希 `model.action_model`，`train_entry.py:238,398`），约 0.1 天。
  - ③合同分支（`starvla_contract.py:33-34`），约 0.1 天。
  - ④按调用设噪声种子（见第 5 项），约 0.25–0.5 天。
  - ⑤冒烟：训练、服务、跑几个回合，约 0.5 天。
  - QwenPI_v3 在我们环境里从没跑过，集群上没有一个实验配置用过它，第一次用可能更久。
  - ③④要改评测正在读的文件（`vagen/envs/robocasa/starvla_contract.py`、`downstream/eval/closed_loop.py` 和 StarVLA 补丁）。按你 11-06 第 6 条，要先确认没有评测在跑，或者在新工作区里改。冻结期不推主线。

### 3. 只有选丙才要定的三项设置

选丙时再定，选项、代价和依据见附录 A2.1。推荐放前面：
- **冻结到什么程度。** 推荐 `frozen_vlm_tokenizers` 加 A6（干净前缀），一次约 175 卡时；另一种是 `frozen_vlm`，一次约 190–225 卡时，显存约 60–80 GB〔推断〕。
- **输入布局。** 推荐 StarVLA 式；VLAKit 自己的布局每起点约 430 卡时，单次训练就过 300 卡时线〔推断〕。
- **拷贝和环境。** 推荐拷一份 VLAKit 进集群，先审查再改；hydra-core 装进新建的环境。不装 hydra 就要改 VLAKit 的训练引擎，天数没估。

### 4. 第二阶段的卡时怎么批

- 推荐逐阶段批。规则本身不强制这样做，因为推荐设置下没有哪一次训练超过 300 卡时：阶段 1 约 210（试跑加底座种子 1），随路线一起批；阶段 2 约 410（V4 加底座种子 2），阶段 3 约 410（SFT 加 S0），每个阶段开跑前我再问你。如果试跑量出来单次训练超过 300 卡时，那一次照规则单独问你。
- 也可以一次批全程，约 1,030〔推断〕。
- **底座第二个种子什么时候训。**
  - 推荐闸门过了以后，和 V4 一起训。闸门不过就省下约 205。
  - 也可以和种子 1 同时训。噪声底能早约 1 天出来，但闸门不过也已经花掉约 205〔推断〕。

### 5. 确定性评测的噪声（甲、乙、丙都要定）

- 流匹配的输出是一次采样，不是均值〔码〕。
- VLAKit 不给种子时用不带种子的 `torch.randn`（`neo_vla/model/decoder.py:861-863`）〔码〕。我们闭环的请求里没有种子（`downstream/eval/closed_loop.py:391`）〔码〕。
- 推荐：每次调用的噪声种子由"任务、回合种子、调用序号"哈希出来，客户端随请求发给服务端（照 VLAKit 的 `bench/client.py:83-87`），服务端每次预测前用它设种子。现在的请求只有图像、指令和 do_sample（`closed_loop.py:391`），所以客户端也要改，这部分算进各路线服务那一块的工作量。四个起点用同一个方案，同一个种子跑两遍，看能不能复现。
- 甲也是这样。StarVLA 采样用的 `torch.randn` 不带生成器（`LayerwiseFM_ActionHeader.py:359-363`），服务只在启动时设一次种子（`server_policy.py:22-26`）。12 个客户端共用一个服务时，每次调用拿到的噪声取决于请求到达的先后。websocket 服务在一个 asyncio 循环里逐个做预测，所以只要在 `policy_wrapper.predict_action` 之前调一次 `torch.manual_seed` 就够了〔码〕。
- 另一种：从零噪声起步。约改 5 行〔推断〕，但输出就不再是模型的采样分布了。

### 6. 两件排期（甲、乙、丙都要定）

- **每次训练用几张卡。** 推荐 8 卡：每次约 1 天，但要等一台整机空出来。另一种是 1 卡，和下游线 11-05 那批一样（都是 zoetrope 的 1 卡任务），每次约 7–8 天。两种的卡时一样。
- **第二阶段排在收口之前还是之后。** 推荐排在 D3 之后、收口汇总之前，收口汇总里 D1 那一行等结果出来再补。全程约 1,030 卡时，是 11-05 那批（约 15 卡时）的约 70 倍；全程墙钟约 5 天（适配 1.5–2 天，加三个阶段约 3.2 天），另加排队。另一种是排在收口之后，但 11-05 任务书说，收口汇总之后这条线转入写作、不再新开实验；STATUS 里也记着 D1 和预算、和这条规则有冲突〔档〕。

---

## 任务表（计划，全部未开始）

两张表的卡时都是〔推断〕，按推荐设置估：只训头、StarVLA 式输入、20 万步 × 批 64。乙、丙按只训专家估，同一量级；甲要看复制数，以试跑实测为准。VLAKit 引擎的开销没量过，另有约 ±30% 的不确定。四份起点权重就用 Qwen3 闸门检查时那几份，第二阶段开跑前从 DS-Q3* 的 experiment.json 核对路径〔未核实〕。训练排在 (g)、第 7 条、D3 之后。

### 训练任务

| 编号 | 任务 | 起点 | 卡 | 预计卡时 | 前置条件 | 状态 | 时间 |
|---|---|---|---|---|---|---|---|
| D1-T0 | 试跑：8 卡 200 步，量每步耗时和显存峰值（甲还要量复制数 2 和 16）；再用 1 卡跑约 50 步，给 D1-E0 用 | 底座 | 8，另 1 | 约 2–3（8 卡部分约 2） | 你定路线和排期；(g)、第 7 条、D3 之后 | 未开始 | — |
| D1-T1 | 底座 种子 1 | Qwen3-VL-8B 底座 | 8 | 约 175（160–190） | T0 正常 | 未开始 | — |
| D1-T2 | V4 | 密集 RL（V4） | 8 | 约 175 | D1-E1 ≥ 15% | 未开始 | — |
| D1-T3 | 底座 种子 2（噪声底） | Qwen3-VL-8B 底座 | 8 | 约 175 | D1-E1 ≥ 15% | 未开始 | — |
| D1-T4 | SFT | 主线 SFT | 8 | 约 175 | V4 − 底座 ≥ 5 个百分点 | 未开始 | — |
| D1-T5 | S0 | 稀疏 RL（S0） | 8 | 约 175 | V4 − 底座 ≥ 5 个百分点 | 未开始 | — |

- 试跑用 8 卡，是因为 1 卡量不到 8 卡的每步时间（梯度同步约 30 ms 一步）。先例：T2 的 200 步短任务 8 卡跑了 12 分钟，量出的显存峰值和正式训练一样〔测〕。8 卡要等整机，排队可能更久。

### 评测任务

| 编号 | 评什么 | 回合 | 卡 | 预计卡时 | 前置条件 | 状态 | 时间 |
|---|---|---|---|---|---|---|---|
| D1-E0 | 试跑评测：每次调用时延、同种子复现、12 维动作映射 | 约 20–50 | 1 | 约 1–3 | D1-T0 完成 | 未开始 | — |
| D1-E1 | D1-T1 终点 | 943 | 8 | 约 30（26–33） | D1-T1 完成 | 未开始 | — |
| D1-E2 | D1-T2 终点 | 943 | 8 | 约 30 | D1-T2 完成 | 未开始 | — |
| D1-E3 | D1-T3 终点 | 943 | 8 | 约 30 | D1-T3 完成 | 未开始 | — |
| D1-E4 | D1-T4 终点 | 943 | 8 | 约 30 | D1-T4 完成 | 未开始 | — |
| D1-E5 | D1-T5 终点 | 943 | 8 | 约 30 | D1-T5 完成 | 未开始 | — |

评测按 EVAL_PROTOCOL 的固定评测单：
- 34 个任务、943 回合；
- 9 个空间任务单列；
- 逐任务配对；
- 评完自动扫条图；
- 坏回合用 4 进程重跑。

---

## 任务书的问题，一行一答

| 问 | 答 | 甲和丙的差别 | 依据 |
|---|---|---|---|
| 能不能挂 Qwen3-VL-8B，skip map 是不是改配置就行 | 挂不上。只有 layer_map 改配置就行，主干一侧要写约 5 天代码〔码 + 推断〕 | 甲原生支持 Qwen3-VL，读的是隐状态不是键值 | A1 |
| 数据格式要写多少适配 | 丙约 3 天、600–700 行新代码〔推断〕 | 甲、乙直接用 StarVLA 的数据管线，0 天 | A4 |
| 视觉处理器和分词器能不能全走本地 | 能。VLAKit 只本地加载分词器，自己不加载视觉处理器，全树没有联网下载〔码〕 | 甲照旧；丙要从本地 `preprocessor_config.json` 接 HF 处理器 | A5.1 |
| 三相机 | 支持，默认就是 3 个视图〔码〕 | 甲照旧；丙按左、右、手腕放进槽 0/1/2 | A5.2 |
| 16 步 × 12 维和 min-max 能否和 StarVLA 一致 | 丙：16 步改配置；12 维能映射，control_mode 要打补丁；恒等映射能和 min-max 一致，但要关掉拟合检查〔码 + 测〕 | 甲完全沿用 StarVLA | A5.3 |
| 本体状态要不要 | 不要。VLAKit 可以不给状态〔码〕 | 甲不用；丙推荐用 A6 把状态列从前缀里拿掉 | A5.4 |
| 评测策略服务和每回合耗时 | 每次调用约 0.08–0.2 秒〔推断〕；两次调用实测间隔 14.0 秒〔测〕；每回合耗时和 StarVLA 差不多〔推断〕 | 甲、乙只加合同分支和按调用的种子，约 0.5 天；丙要新服务，约 2 天〔推断〕 | A5.5 |
| 克隆 | 按你批的"只拉代码"克隆到本机，只读过〔档〕 | 丙还要拷进集群、装 hydra-core、下载 manifest 样例，要你批 | A5.6 |
| 卡时、显存、能否并行 | 见结论 | 甲看复制数；丙看冻结模式和输入布局 | A6 |

---

## 风险（前五条，其余见附录 A7）

1. **超过两天线。** 丙约 12 天，乙约 3.5–4.5 天，都超过。任务书要求停下等你定。甲约 1.5–2 天，但 QwenPI_v3 在我们环境里从没跑过，可能更久。
2. **甲是改设计。** 它读每层隐状态，只用交叉注意力，动作 token 之间互不注意。"读全部 36 层"和"动作 token 互相可见"在甲里只能取一个，VLAKit 代码里的联合 softmax 两者都有。所以 NEO-VLA 的灵敏度证据（冻结时差 11.4、全微调后差 4.5）搬不到甲上。还没核实的三件（8B 下的显存和每步耗时、4 步去噪够不够、ZeRO-2 下不用的参数）靠试跑查。
3. **卡时和显存都是缩放估计**，误差约 ±30%。甲的复制数用缺省 16，单次训练可能过 300 卡时线。丙的 EMA 和编译设置会让每步时间相差超过 ±30%（附录 A6.3）。第一次正式训练前必须先试跑。
4. **闸门风险。** X1 在冻结主干上训到 5 万步，不到 2%。还没有证据表明冻结主干上的头能过 15%。如果底座过不了，在阶段 1 停下时已经花掉约 210 卡时〔推断〕。
5. **流匹配的输出是采样。** 成功率和回归头不同，可能跟主干无关。噪声种子方案要在开训前定死，四个起点一样，客户端也要改。和 E2、T2、X1 比，是不同的头，而且那三档是 Qwen2.5。

---

# 附录（细节和依据）

## A1 能不能挂 Qwen3-VL-8B（VLAKit 一侧）

### A1.1 VLAKit 只有一种主干

- 模型注册表里只有 `neo_vla` 一个工厂（`neo_vla/model/build.py:496`），全树就这一处注册〔码〕。
- 只有一个模型配置 `config/model/neo2b.yaml`（NEOpp-2B-SFT，第 1-11 行），一个结构 `config/arch/skip28.yaml`（在 NEO-2B 的 40 层里接 28 层，第 1-6 行）〔码〕。
- 主干是拷进仓库的 NEO 建模代码，不是 transformers 的类（`neo_vla/model/backbone/neo_llm.py:1-13`）〔码〕。
- 视觉"塔"只有两层卷积，没有 transformer（`neo_vla/model/backbone/vision.py:1-14`）〔码〕。
- 树里没有 Qwen-VL、InternVL、SigLIP 主干，只在注释里提到（`neo_llm.py:3`，`neo_vla/data/understanding.py:136-142`）〔码〕。
- 任务书说论文比了两种主干，另一种不在代码里〔码〕。

### A1.2 Qwen3-VL-8B 的文件夹在哪一步报错

1. **读配置。** 代码要 `llm_config` 和 `vision_config`。Qwen3-VL 的是 `text_config`，直接 KeyError（`neo_llm.py:171-174`）〔码〕。
2. **改了键名也不行。** 任何非空的 `rope_scaling` 都会被拒（`neo_llm.py:186-192`）〔码〕。Qwen3-VL 用的是交错 MRoPE，`mrope_section` 为 [24,20,20]〔测，集群 config.json〕。
3. **读权重。**
   - 只打开 `<路径>/model.safetensors` 这一个文件（`neo_llm.py:911-912`），键名前缀要 `language_model.model.` 和 `vision_model.`（`neo_llm.py:43-47`）〔码〕。
   - 我们的 checkpoint-1232 是 7 个分片、750 个张量，前缀是 `model.language_model.*` 和 `model.visual.*`〔测〕。底座是 4 个分片〔档，REFERENCE.md:13〕。
   - 加载两头都严格，对不上就报错（`neo_llm.py:942-971`）〔码〕。
4. **图像 token。** 代码用 NEO 的 151669/151670/151671（`neo_llm.py:53-55`）〔码〕。Qwen3-VL 用 `<|image_pad|>` 151655、`<|vision_start|>` 151652、`<|vision_end|>` 151653〔测〕。

把分片合成一个文件就是拷贝。任务书第二阶段规定"按路径读，不拷贝不移动"，所以这条路不行。

### A1.3 几何对照

| 项 | Qwen3-VL-8B（checkpoint-1232/config.json）〔测〕 | VLAKit 的 NEO 代码〔码〕 |
|---|---|---|
| 文本层数 | 36 | NEO-2B 40 |
| 隐层维度 | 4096 | 树里没写〔未核实〕 |
| 注意力头 | 32 个查询头、8 个键值头，每头 128 维 | 16 / 8 / 128（`neo_llm.py:71-72` 注释） |
| q/k norm | 每层一个，覆盖整头 128 维 | 每头拆成 t 64 + h 32 + w 32，四个 64 维的半尺寸 norm（`neo_llm.py:524-581`） |
| 旋转编码 | 交错 MRoPE，theta 5e6 | t 那半用 rope_theta，h/w 那半用 rope_theta_hw = 1e4（`neo_llm.py:722-737`） |
| 掩码 | 因果 | 块因果：t 相同的 token 互相双向可见（`neo_vla/model/encoder.py:832-846`） |
| 视觉 | 27 层 ViT，宽 1152，patch 16，merge 2，时间 patch 2；DeepStack 取 ViT 第 8/16/24 层，加到 LLM 第 0/1/2 层之后 | 两层卷积（`vision.py:1-7,196-226`） |
| 像素归一化 | 均值和方差都是 0.5，即输入 [-1,1] | 在塔入口做 ImageNet 均值方差（`neo_vla/data/images.py:24-41`） |
| 每 token 覆盖 | 32×32 像素 | 32×32 像素 |
| 每层键值形状 | [B, 8, S, 128] | [B, 8, S, 128] |

对得上的只有最后两行和 GQA。权重加载会因为形状被拒：k_norm 是 128 对 64，q_norm_hw 和 k_norm_hw 也没人写入（`neo_llm.py:932-937`）〔码〕。

### A1.4 skip map 怎么接

- **接法**〔码〕：
  - 专家第 j 层拿主干第 `layer_map[j]` 层旋转后的键和值，形状 [B, 8, Lp, 128]。
  - 它把这些键值和专家自己后缀的键值拼起来，做一次 SDPA softmax，缩放 head_dim^-0.5。GQA 重复 4 倍。掩码只遮补齐位。
  - 出处：`neo_vla/model/expert.py:119-135, 249-267`，`neo_vla/model/decoder.py:387-419`。
- **专家是串联的。** 第 j 层的输入是专家第 j-1 层的输出，不是主干的隐状态（`expert.py:256-268`）〔码〕。
- **主干截断。** 不联训时，主干必须正好结束在最高的接入层，更深的层会被丢掉（`build.py:525-532`）〔码〕。
- **跟主干绑定的。** 头数、键值头数、head_dim 都从主干配置取，所以键值形状自动对上（`build.py:278-293`）〔码〕。
- **跟主干无关的。** 专家残差宽 1024、FFN 宽 3072，都是自由配置（`neo2b.yaml:42-46`）〔码〕。
- **对 Qwen3-VL 只改配置的部分。** 新写一个结构 yaml：`layer_map` 设为 [0..35]，主干层数 36〔推断〕。
- **要写代码的部分**〔码 + 推断〕：
  - 专家的 q/k norm 改成整头 128 维。旋转改成整头旋转：NEO 只转 t 那半（`expert.py:73-87`）。Qwen3 文本位置上 t=h=w，等价于 theta 5e6 的普通一维旋转〔推断〕。
  - 校准时从主干拷 2 个增益，不是 4 个（`expert.py:212-235`）。
  - 编码器里后缀的 cos/sin 表，从 head_dim//2 改到 head_dim（`encoder.py:1277-1283`）。
  - 后缀的起始位置改成 Qwen3-VL 前缀的最大位置 + 1。
- 这个约定错了不会报错，只会让查询和键的相对相位悄悄对不上〔推断〕。

### A1.5 专家大小

- 宽 1024 时，按代码里的投影形状算〔推断〕：
  - Qwen3-VL-8B 每层约 19.93M 参数。
  - 接全部 36 层约 717M，接 28 层约 558M，接 18 层约 359M。
  - 流匹配头另有约 3.2M。
- NEO-2B skip28 约 440M〔推断〕。
- X1 的回归头是 129M〔测，X1_RESULTS_20261029.md:30〕。甲约 0.69B（见"要你定的事"第 2 项）。
- 任务书要"读主干所有层"，所以是 36 层、约 717M。专家大小跟着主干头数走，和 VLAKit 自带 NEO-2B 配置的专家（skip28 约 440M）不一样大。

### A1.6 冻结、梯度检查点、数值精度

- **冻结。**
  - `model.trainable` 有四种取值：full_ft、frozen_vlm、frozen_tokenizers、frozen_vlm_tokenizers（`build.py:39-56`）〔码〕。
  - `frozen_vlm` 对 `encoder.backbone` 和 `encoder.vision` 设 `requires_grad_(False)`，覆盖全部解码层、嵌入表和最后的 norm（`build.py:383-441`）〔码〕。
  - VLAKit 发布的第一阶段脚本用的是 `frozen_vlm`（`scripts/stage1.sh:2,21`）〔码〕；论文的冻结消融用的是哪种模式，没核实。
- **`frozen_vlm` 的问题。**
  - 前缀里的状态、内参、外参 token 还在训练，它们的输出写进主干输入（`encoder.py:1104-1139`）〔码〕。
  - 代码故意不用 `no_grad`（`build.py:8-11`）〔码〕，所以整个 8B 主干要建计算图、做反传。没有权重梯度，但多了激活显存和计算〔推断：T2 实测每步时间是只前向的 1.2 倍〕。
- **`frozen_vlm_tokenizers`。**
  - 把 `encoder.tokenizers` 和 `decoder.tokenizers` 也冻住（`build.py:56`），只训专家和流匹配头，主干前向不建计算图〔码 + 推断〕。
  - 仓库没提供这个 yaml，要靠覆盖参数选它。
  - 这个模式本来是给条件 token 预训练过之后的真机微调用的（`build.py:398-402`）〔码〕。我们没有 NEO 预训练，这个模式下条件 token 恒定不变：
    - 数值投影的输出层是零初始化（`tokenizers.py:449-450`），填进去的数对 token 没有影响〔码〕。
    - "未知"行、角色行、臂行都是固定的随机向量（`tokenizers.py:454, 562-564`）〔码〕。
    - 进 Qwen3 前缀的 6 个状态列和每视图 12 个几何列，都是没训过的 token〔码〕。
    - `decoder.tokenizers` 只放块身份嵌入（`tokenizers.py:809-860`），冻在随机值上不影响区分各块〔码〕。
  - 所以推荐配 A6，把这些列从前缀里拿掉（A2.1）。
- **梯度检查点。** `trainer.gradient_checkpointing` 默认关，只作用在主干上（`config/train.yaml:31-34`，`neo_vla/engine/runner.py:1199-1202`）〔码〕。只训专家时不需要〔推断〕。
- **精度。**
  - 冻结的子树保持 bf16，可训练参数提升到 fp32，计算在 bf16 autocast 下跑（`build.py:444-493`，`neo_vla/engine/context.py:38-41`）〔码〕。
  - 默认用 DDP，也能用 FSDP（`neo_vla/engine/strategy.py`）〔码〕。
- **存盘。** 保存的是完整 DDP state_dict，冻结主干也在里面（`strategy.py:138-139`）〔码〕。每次约 25–30 GB〔推断〕。
- **锚点行。** 构建时会用固定种子重抽保留嵌入行 151682 及以后（`build.py:172-174`，`tokenizers.py:746-786`）〔码〕。Qwen3-VL 新增 token 最大到 151668〔测，added_tokens.json〕，所以这些行没用到，四个起点也一样〔推断〕。

## A2 丙路线：要定的设置和适配清单

### A2.1 丙的三项设置

- **冻结到什么程度。**
  - 推荐 `frozen_vlm_tokenizers` 加 A6：只训专家和流匹配头，主干只做前向，一次约 175 卡时〔推断〕。
    - 这个模式本来是给条件 token 预训练过之后的真机微调用的（`build.py:398-402`）。我们没有 NEO 预训练，前缀里的状态、内参、外参 token 会一直停在初始化状态。
    - 数值投影的输出层是零初始化（`tokenizers.py:449-450`），填进去的数对 token 没有影响，剩下的只是固定的随机行（`tokenizers.py:454, 562-564`）。
    - A6 把这些列从前缀里拿掉，前缀就只剩图像和指令，和 StarVLA 的输入一致。
  - 另一种是 `frozen_vlm`，VLAKit 发布的第一阶段脚本用的就是它（`scripts/stage1.sh:2,21`）；论文的冻结消融用的是哪种模式，没核实。
    - 这些 token 会训练，梯度要回传穿过整个 8B 主干，但不算主干的权重梯度。
    - 按 T2 实测推：T2 反传穿过冻结的 7B，每步 0.39 秒，只前向的 X1 是 0.32 秒。照这个算，一次约 190–225 卡时，不到 300 卡时线；显存约 60–80 GB，可能要开梯度检查点〔推断〕。
- **输入布局。**
  - 推荐 StarVLA 式：三相机当前帧，不用历史帧，前缀约 300 token。
  - VLAKit 自己的布局是历史帧加当前帧共 6 个视图，每个 256×352，前缀约 700–908 token。每起点约 430 卡时（含评测），每次训练都超过 300 卡时线〔推断〕。
- **拷贝和环境。**
  - 推荐：拷一份 VLAKit 进集群，先审查再改；hydra-core 装进新建的环境。
  - 另一种是自己写训练入口。但 `neo_vla/engine/runner.py:24` 在顶层导入了 hydra，第 147 行用它建数据集，配置组也靠 Hydra 组合（`config/train.yaml:5-11`）〔码〕。所以不装 hydra，就得改 runner.py 这两处，再自己用 OmegaConf 把各组配置合并起来。天数没估，不会只有 0.25 天。

### A2.2 丙路线的适配清单

| 项 | 内容 | 工作日 |
|---|---|---|
| A1 主干包装 | 用 HF transformers 4.57.0 的 Qwen3-VL（和 StarVLA 环境同版本），换掉 PrefixEncoder 的内部，子模块名保持 encoder.backbone / encoder.vision；读分片；逐层取旋转后的键值；位置用 get_rope_index；换成因果掩码；接 DeepStack | 2 |
| A2 视觉 | 调 HF 的 Qwen3VLVisionModel；每个视图裁到真实图块再编码；换图像 token id；分词器接线约 0.25 天，含在 A1/A2 里 | 1 |
| A3 脚手架 | 去掉 CHAT_CLOSE 里的 think 块（`encoder.py:94-96`）；前缀长度常数 636/16/908 要重算（`constants.py:182-184`） | 含在 A1 |
| A4 专家 | 整头旋转和 q/k norm，2 个增益的校准 | 0.5 |
| A5 测试 | 包装出的键值对照普通 HF 前向；图像 token 数；冻结检查 | 1 |
| A6 干净前缀 | 把状态列和几何列从前缀里拿掉，不让 Qwen3 读到没训过的固定 token，这样前缀和 StarVLA 一致。推荐模式下主干本来就只做前向 | 0.5 |
| B 数据 | RoboCasa365 转 VLAKit 语料（见 A4） | 3（2.5–3.5） |
| C 动作侧 | control_mode 补丁、损失权重、关联训（历史帧在 B 的 manifest 和 D 的服务端里处理）；回合末窗口可选 | 0.5（0.25–1） |
| D 策略服务 | 讲我们 websocket 协议的新服务、合同分支、按调用的噪声种子（客户端也要改）、启动脚本、1 卡冒烟（见 A5.5） | 2 |
| E 环境和作业 | 装 hydra-core（0.25 天，不含审批；自写入口另算）；sco 作业包装（0.5–1）；1 卡试跑（0.5） | 1.5 |
| **合计** | | **约 12（9–15）** |

- 全部是〔推断〕。服务读者列的分词器 0.25 天、视觉 0.5 天和 A1/A2 重复，已去掉；"关历史帧"只在 B、D 里算，C 里不再算。
- 不含两项：对拷进集群的 VLAKit 副本做代码审查；选 `frozen_vlm` 时的额外工作。

## A3 甲、乙、丁的细节

- **甲：QwenPI_v3**（集群 `third_party/starVLA`，git 4507931）
  - 每层一个 LayerNorm+Linear，把 4096 压到 `action_dit_hidden_dim`，默认 1024（`starVLA/model/framework/VLM4A/QwenPI_v3.py:227-239`）〔码〕。
  - DiT 深度等于主干层数 36，取所有层隐状态（`QwenPI_v3.py:189-223, 249-273`，`share_tools.py:261-295`）〔码〕。读的是 `hidden_states[-36:]`（`QwenPI_v3.py:271`），不是键值〔码〕。
  - 支持 Qwen3-VL（`QwenPI_v3.py:189-196`，`modules/vlm/QWen3.py:73`，以及我们补丁里按 model_type 分派）〔码〕。
  - 默认（`interleave_self_attention=false`）每个 DiT 块只有一个交叉注意力：查询是 32 个可学 token 加 16 个动作 token，键值是对应那一层投影后的隐状态。动作 token 之间互相不做注意力（`cross_attention_dit.py:240-243, 298-318`）〔码〕。打开交替自注意力后，奇数块改成自注意力，只读 18 层，18 个投影层用不上〔码〕。
  - 流匹配：Beta(1.5,1) 时间采样、速度损失、Euler 采样器（`LayerwiseFM_ActionHeader.py:277-408`）〔码〕。动作头模块文件：`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`。
  - 复制数缺省 16，复制动作和每层隐状态（`QwenPI_v3.py:315-321`）；头在 float32 autocast 下算（`QwenPI_v3.py:308`）。StarVLA 环境的 torch 在 cuda 分支只检查 bf16 支不支持（`torch/amp/autocast_mode.py:314-322`），float32 生效；starVLA/ 和我们的 downstream/ 里没有打开 TF32 的地方（只有 PI0 和 openpi 的文件里有，而且是关掉）。所以 DiT 按真 fp32 算〔码〕。
  - 冻结用 `trainer.freeze_modules='qwen_vl_interface'`，和 X1 一样（`starVLA/training/trainer_utils/trainer_tools.py:191-233`）〔码〕。
  - 不带 v3 的 QwenPI，DiT 宽 4096，放 8B 太大〔码 + 推断〕。
  - 我们的评测合同只接受 QwenOFT（`vagen/envs/robocasa/starvla_contract.py:33-34`），要加一个分支〔码〕。我们的 `recipe.yaml` 和 `make_experiment.py` 写死 QwenOFT〔码〕。
  - 集群上 downstream_runs 和 models_runs 下没有一个实验配置用过 QwenPI 或 LayerwiseFM〔测〕。
- **乙：把 VLAKit 的专家照着写进 StarVLA**
  - HF 的 Qwen3-VL 注意力把 k_norm 和旋转之后的 [B,8,S,128] 键值写进缓存（StarVLA 环境 `transformers/models/qwen3_vl/modeling_qwen3_vl.py:427-438`）〔码〕。前向时显式传 `use_cache=True`，并且不开梯度检查点，就能从返回的 past_key_values 逐层取到〔推断〕。不能靠配置的默认值：checkpoint-1232 的 config.json 里 use_cache 是 false〔测〕，底座的没读到，四个起点要统一显式传。
  - 要手写的是专家层（照 `expert.py`）。时间采样（Beta(1.5,1)）、速度损失和 Euler 采样器可以复用 StarVLA 的 LayerwiseFlowmatchingActionHead（`LayerwiseFM_ActionHeader.py:277-408`），只换掉里面的 DiT。若要逐项照搬 VLAKit `decoder.py` 的损失和 10 步 Euler 采样，写明理由。
  - 写成我们自己的 StarVLA 框架。不运行 VLAKit；VLAKit 是 Apache-2.0 许可（`pyproject.toml:8`）〔码〕。
  - 天数约 3.5–4.5〔推断，未核实〕：
    - 取键值，约 0.5 天；
    - 专家，约 1 天；
    - 流匹配头和采样（复用 StarVLA），约 0.25–0.5 天；
    - 框架接线和合同分支，约 0.5–1 天；
    - 对照测试和冒烟，约 1 天。
    - 各项加起来约 3.25–4 天，留些余量。
- **丁：回归头**
  - X1 在 Qwen2.5-7B 冻结主干上，训了 5 万步，四个起点都不到 2%〔测〕。
  - X1-B-030-s1 训练 5 万步约 36 卡时，评测 31.7 卡时〔测〕。

## A4 数据格式要写多少适配（只丙需要）

甲和乙直接用 StarVLA 现有的数据管线，0 天。

### A4.1 VLAKit 只读自己的语料

- 它由三部分组成〔码〕：
  - 一份 schema 3 的 manifest JSON（旧版本会被拒，`neo_vla/data/dataset.py:563-573`）；
  - 一份 `neo_vla_embodiments_v2` 归一化文件 `embodiments.json`；
  - 处理好的缓存：每个目录一张 Feather/Arrow 数值表、一份帧索引 JSON、相机参数数组、渲染好的提示 JSONL。
  - 出处：`dataset.py:546-617`，`neo_vla/data/reader.py:1-14, 642-731`，`neo_vla/constants.py:17-41`。
- 它不直接读 RLDS 或 LeRobot 的 parquet〔码〕。帧索引可以直接指向 LeRobot 的 mp4，不用重编码（`reader.py:350-374`）〔码〕。
- **构建工具没发布。**
  - 文档说源数据转换、manifest 和缓存构建、归一化统计都不在仓库里（`docs/preprocessing.md:7-9`，`docs/training.md:185-199`）〔档〕。
  - 代码里引用了 `tools/prepare_data.py`、`build_manifest.py`、`build_cache.py`、`build_stats.py`、`build_embodiment.py` 等，但 `tools/` 里只有 10 个 filter/qc/validate 脚本〔码〕。
- 完整的 manifest 样例只在发布到 HF 的语料里，克隆里没有。去拿就是"从集群外拷"，而且超出"只拉代码"，要你批。
- 加载器约有 100 处拒收检查，只能照着它反推 JSON 结构〔推断〕。

### A4.2 RoboCasa365 源数据（集群只读）〔测〕

- 路径：`/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0/target/`。
- 格式：LeRobot v2.1，robot_type 是 PandaOmron，20 Hz。
- 相机：三路 256×256 h264，分别是 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。
- 状态 16 维，动作 12 维。
- 手臂是 OSC_POSE 增量，参照底座坐标系，output_max 为 0.05 m 和 0.5 rad。底座是 JOINT_VELOCITY。
- 相机配置在抽查的 150 个回合（覆盖 50 个任务）里完全一致。
- 30% 子集是每个任务 150 个训练回合（`downstream/demos/manifests/target50_030.json:4`），共 7,500 回合、4,442,662 帧（REFERENCE.md:182）〔档〕。

### A4.3 字段怎么填

| VLAKit 字段 | 怎么填 | 依据 |
|---|---|---|
| 目录 | 每个任务一个条目，共 50 个 | 〔测〕 |
| source id | 新加 robocasa365 = 18，改副本里 `registry.SOURCE_IDS` 一行 | `neo_vla/data/registry.py:19-26`〔码〕 |
| weight | 1/行数，让每个任务等概率被抽到 | `neo_vla/data/mixture.py:3-14`〔码〕；StarVLA 按任务等概率〔推断〕 |
| frame_ref | robot，动作留在底座坐标系，和 StarVLA 一样 | 〔推断〕 |
| control_hz | 20，提示里写"Frequency: 20 Hz" | info.json fps、embodiment.json〔测〕 |
| embodiment | 新 key，例如 `panda_omron_robocasa`，仿 `libero_franka_sim`；机身 base: command，torso 和 head: none | `config/embodiments.yaml:58-65,312-330`〔码〕，取值〔推断〕 |
| 指令 | episodes.jsonl 的 `tasks[0]`；用 VLAKit 自己的渲染器，Qwen3-VL 分词器数 token，预算 256 | `config/prompt_template/policy.yaml:17-56`〔码〕 |
| 相机 | 槽 0/1/2 = 左、右、手腕，和 StarVLA 同序；frames 类型 mp4，直接指向现有文件 | `dataset.py:2996-3046`〔码〕 |
| 内参 | 推荐模式下内参 token 不随输入变化，填不填都一样。建议和外参一样置 none（加了 A6 就直接拿掉），省掉换算。真要填：constant，来源记 vendor；fx = fy = 128/tan(fovy/2)，cx = cy = 128；左右相机 fovy 60 → 221.70，手腕 fovy 75 → 166.81 | cam_configs、model.xml〔测〕，换算〔推断〕；token 不变见 A1.6〔码〕 |
| 外参 | 推荐全部置 none，省约 0.5 天；也可以左右用 constant，算法照 `bench/libero.py:34-41` 的 MuJoCo→OpenCV 那一步 | 〔推断，没用仿真渲染核对〕 |
| 本体状态 | 手臂 state layout 置 null，编码为 UNKNOWN，对应 StarVLA 的 include_state false；加了 A6 就从前缀里拿掉 | `dataset.py:1860-1884`〔码〕 |
| 手臂动作 | flat_dim 12，parquet 列序是底座 4、模式 1、末端位置 3、末端旋转 3、夹爪 1；dpos = 列 5–7，每单位 5 cm；drot = 列 8–10，每单位 0.5 rad，参照系 spatial；夹爪 = 列 11，开 -1、合 +1，method 记 declared | 〔测〕 |
| 机身 | `action.body` = [action[0:3], NaN×5]，即底座速度命令；`observation.body` 全 NaN；torso = action[3] 在样本里恒为 0，声明为缺省 | 〔测〕 |
| control_mode | 没有对应的列，见 A5.3 | 〔码〕 |
| 历史帧 | `history.blocked` 列出每个回合的全部位置，这样永远不带历史帧 | `neo_vla/data/dataset.py:1953-1962`〔码〕 |
| 归一化 | 每列的 min/max 写进 low/high，见 A5.3 | 〔推断〕 |

### A4.4 工作量

约 600–700 行新代码，约 3 个工作日（2.5–3.5），不用 GPU〔推断〕：
- 语料构建器约 350–400 行，约 1.5 天；
- embodiments.json 构建约 150 行，约 1 天；
- 副本里的配置和注册表改动约 40 行；
- 往返检查约 80 行，约 0.5 天。

转换本身只做读写，一个 sco 作业不到 1 小时〔推断〕。

### A4.5 缺口

- **control_mode 没有列**，见 A5.3。
- **图像尺寸。** StarVLA 先把图缩到 224（`recipe.yaml:37`），但它的 Qwen3 接口用的是 AutoProcessor 的默认设置（`QWen3.py:65`）：shortest_edge 是 65536 = 256²，smart_resize 会把 224 放大到 256（`image_processing_qwen2_vl.py:76-79`），所以每视图是 8×8 = 64 个 token〔码；放大结果按代码算过，刚好 256〕。要和它一致，VLAKit 这边先照样缩到 224 再放大到 256，`model.prefix.max_grid` 设 [8,8]，前缀长度常数跟着重算。VLAKit 默认画布是 256×352（[8,11]），256×256 的帧本来也是 64 个 token〔码〕。
- **历史帧的位置。** 关了历史帧，Earlier 那 3 个视图块仍占着前缀长度，只是被遮住〔码〕。
- **帧数。** 7,500 个回合的 mp4 帧数是否都等于 parquet 行数，没逐个查〔未核实〕。不一致时加载器到训练时才会报错。

## A5 任务书的六样细节

### A5.1 视觉处理器和分词器能不能全走本地路径

- **分词器：能。**
  - VLAKit 只有一处加载分词器：`AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)`（`neo_vla/data/understanding.py:320-322`）〔码〕。
  - 分词器里没有 `<IMG_CONTEXT>` 时，NEO 的探测会跳过（`understanding.py:323-324`）〔码〕。Qwen3-VL 的分词器没有这个 token，所以能过。
  - Qwen3-VL 的分词器文件都在 checkpoint-1232 里〔测〕。
- **视觉处理器：VLAKit 根本不加载。**
  - 缩放和归一化都是它自己的代码（`neo_vla/data/images.py:189-218`）〔码〕。
  - 换成 Qwen3-VL 后，要从本地 `preprocessor_config.json` 加载 HF 的处理器。这个文件在 checkpoint-1232 里：均值和方差 0.5，patch 16，merge 2〔测〕。
- **权重：读的是本地文件，但加载器只认 NEO 格式**，见 A1.2〔码〕。
- **没有任何联网下载或遥测。**
  - 全树没有 hf_hub_download、snapshot_download、load_dataset、torch.hub、wandb、tensorboard〔码，grep〕。
  - 启动脚本默认设 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`（`scripts/stage1.sh:17`，`stage2.sh:19`，`posttrain_real.sh:39`）〔码〕。
- **版本。**
  - VLAKit 要 transformers ≥ 4.50、torch ≥ 2.6、hydra-core ≥ 1.3（`pyproject.toml:13-25`）〔码〕。
  - StarVLA 环境是 torch 2.6.0、transformers 4.57.0，带 modeling_qwen3_vl.py，但没有 hydra（`docs/downstream/reports/step1/env_starVLA.json:5,7,40`）〔测〕。
  - VLAKit 的 `train.py` 是 Hydra 入口（第 13-19 行），训练引擎 `runner.py` 第 24 行也在顶层导入 hydra，第 147 行用它建数据集〔码〕。服务端不导入 hydra〔码〕。
- **底座权重。** Qwen3-VL-8B 底座文件只有 root 可读，只能在 sco 作业里读（REFERENCE.md:13）〔档〕。

### A5.2 三相机

- **支持，而且是默认配置。** `model.prefix.n_views` 是 3（`config/model/neo2b.yaml:19-20`），最多 8 个（`neo_vla/model/tokenizers.py:32-34`）〔码〕。
- 缺的槽位会被遮住，不会丢掉〔码〕。
- 角色设置：左右固定相机设 exterior 或 side，手腕设 wrist、arm_slot 0。每个槽只能放一个视图（`registry.py:109-148`，`dataset.py:3000-3016`）〔码〕。
- 我们 StarVLA 的顺序是左、右、手腕，224×224（`starvla_contract.py:15-19`）〔码〕。

### A5.3 动作块 16 步 × 12 维和 min-max 归一化

- **甲、乙没有下面这些问题**，16×12 和 min-max 都沿用 StarVLA。
- **16 步：改配置就行。**
  - 设 `model.action.chunk=16`（`neo2b.yaml:57`，`build.py:197-211`）〔码〕。后缀是 3×16 = 48 个 token。
  - 假定 30 步的 DCT 项默认关着〔码〕。
- **12 维：不能配置，但能映射。**
  - 手臂行固定 26 维：关节 7、位移 3、旋转 3、夹爪 1、手 12。机身行固定 8 维：底座 3、躯干 3、头 2。别的宽度会报错（`constants.py:63-80, 98-110`，`build.py:225-251`）〔码〕。
  - 末端位置填到位移，末端旋转填到旋转。参照系选 spatial，这样不做任何转换，也不需要状态位姿〔码〕。
  - 夹爪填夹爪列。底座 x、y、yaw 填机身底座，类型 command。
  - 躯干恒为 0，关节和手遮住。遮住的列不加噪声，也不解码（`decoder.py:850-899`）〔码〕。
- **control_mode（第 11 维）没有列。** 三个办法：
  - (b) 改 `neo_vla/data/body.py:384-405` 约 10 行，让姿态目标能读命令列。丙推荐这个。反正要改副本，这样最干净〔推断〕。
  - (a) 塞进一个空闲的姿态列，往后错一行，再遮住对应的状态维。只动数据，但很别扭，容易错位〔推断〕。
  - (c) 评测端按底座有没有动推出模式。在 61,366 行的样本里，模式为 +1 恰好等于底座在动，只有 0.32% 的行例外〔测〕。
- **VLAKit 的归一化公式写死在代码里。**
  - 手臂列：y = 2(x−lo)/(hi−lo) − 1。夹爪：2g − 1。底座：x/s。
  - 文件头必须写 q01q99_linear，写别的会被拒（`neo_vla/data/embodiment.py:83-86, 769-799`）〔码〕。
  - lo、hi、s 的值来自我们自己写的 embodiments.json。
- **StarVLA 的归一化。**
  - 每维 min_max；min = max 的维置 0；解码不截断（集群 `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:164-184, 216-219`）〔码〕。
  - DS-B/S/RS/RD-030-s1 四份统计完全相同：除躯干（第 10 维，0/0）外，每维 min = −1、max = +1〔测〕。
- **能做到一致，但要关一道检查。**
  - 写 lo = −1、hi = 1、s = 1、夹爪开 −1 合 +1，VLAKit 的映射就是恒等，正好等于 StarVLA 这里的 min-max，不用改代码〔推断〕。
  - 两个小问题。一是文件头写着 q01q99，实际填的是 min/max，要在文档里记清楚。二是启动时的拟合检查（R14：每个有效范围列的 p99 |y| 必须在 [0.5, 2] 内，`embodiment.py:186`）按恒等映射过不了。核查时在集群上每个任务抽 20 个回合（共 1,000 回合、593,827 行），末端旋转三列的 p99 |a| 只有 0.46–0.47〔测〕。新 key 没有豁免（`runner.py:796-805`），启动时会被直接拒掉。要保持恒等映射，就得设 `data.fit_check=false` 关掉这道检查（`runner.py:741`；`robot_adaptation.yaml:4` 默认是开的），并在计划里写明。如果改用 q01/q99 作 lo/hi，就和 StarVLA 的 min-max 不一致了。
- **其他和 StarVLA 不同的地方。**
  - 回合末窗口：VLAKit 会补齐并遮住，每回合多约 15–16 个窗口，约 2.7%〔推断〕。StarVLA 直接丢掉不完整的块。要完全一致，得改 `mixture.py:245-250` 约 10 行〔码〕。
  - 解码截断：VLAKit 把范围列截断到 ±(1+2α)，夹爪截断到 ±1（`decoder.py:898-905`），StarVLA 不截断。在恒等映射下影响很小〔码 + 推断〕。
  - 损失权重：VLAKit 末端 ×2、机身 ×1，StarVLA 是 12 维等权 L1。这个能配置，但必须大于 0（`neo2b.yaml:67-77`）〔码〕。
  - 输出是一次采样，不是回归均值〔码〕。
  - 提示多了频率行、机器人描述句和坐标系句（`policy.yaml:18-44`）〔码〕。
  - VLAKit 的 robot_adaptation 默认是批 128、3 万步、学习率 5e-5，还带对话联训，都要改成 E2 的批 64、20 万步，并关掉联训（`scripts/posttrain_real.sh:14-16,54`；`config/data/robot_adaptation.yaml:17,36`）〔码〕。

### A5.4 本体状态（以及历史帧）

- **VLAKit 不要求本体状态。**
  - 没有状态布局，或者请求里 state 为 null，就用"未知"状态 token。四个起点完全一样（`neo_vla/data/conditioning.py:221-245, 449-462`，`neo_vla/serve/predictor.py:247-270`）〔码〕。
  - 推荐用 A6 把状态列从前缀里拿掉。如果保留，在 frozen_vlm_tokenizers 下，填零和"未知"得到的都是固定不变的向量：填零是角色行加臂行，"未知"再多加一行 unknown（`tokenizers.py:567-576`）。训练时加的噪声也不起作用。四个起点用同一个种子初始化，所以仍然一致〔码 + 推断〕。
  - 只有在 frozen_vlm 下两者才有区别：填零会被当成实测值，训练时以 0.8 的概率往上加噪声（`history.py:10-11, 284-332`）〔码〕，那时推荐"未知"。
- **VLAKit 多读一帧历史，StarVLA 没有。**
  - `history_dropout` 必须在 [0,1) 里，设 1.0 会被拒（`history.py:296-302`）〔码〕。
  - 关历史帧的办法：manifest 的 `history.blocked` 列出所有回合，服务时每次都传 history null（`dataset.py:1953-1962`，`predictor.py:340-372`）〔码〕。

### A5.5 评测端策略服务和每回合耗时

- **VLAKit 的服务。**
  - 普通 HTTP + JSON，ThreadingHTTPServer，有 /health、/metadata、/predict 三个接口（`neo_vla/serve/server.py:1-5, 25-34, 60-121`）〔码〕。
  - 没有 websocket。服务器能接受多个连接，但预测加了锁，一次只做一个；自带的评测客户端一个回合接一个回合地跑〔码〕。
  - LIBERO 客户端每块执行 16 行；RoboTwin 执行 30 行（`neo_vla/bench/libero.py:202-226`，`robotwin.py:178-197`）〔码〕。
  - 每次调用的噪声种子由客户端按 blake2b 算出（`neo_vla/bench/client.py:83-87`）〔码〕。
- **我们的闭环。**
  - StarVLA 的 websocket + msgpack。连上后核对 ckpt_path、动作块 16、动作键顺序、图像尺寸〔码〕。
  - 请求是三张 224×224 uint8 图（PIL bicubic）加指令，`do_sample` 为 False。返回 (1,16,12)，已经反归一化〔码〕。
  - 每次执行 16 行。不传状态、历史帧和种子（`downstream/eval/closed_loop.py:173-185, 386-402`，`starvla_contract.py:29-53`）〔码〕。
- **丙要新加一个服务类型，约 2 天**〔推断〕：
  - 新服务讲我们现有的 websocket 协议，复用 StarVLA 的 WebsocketPolicyServer，策略对象是我们自己写的；
  - 加合同分支；
  - 加每次调用的噪声种子，客户端也要改；
  - 不带历史帧；
  - 在 `downstream/eval/run_eval_ckpt.sh` 里加启动分支；
  - 1 卡、12 进程冒烟。
  - 改用 VLAKit 的 HTTP 协议，客户端那边要改得更多，不推荐。
- **甲、乙**：只要加合同分支和按调用的噪声种子，约 0.5 天，已含在各自的天数里〔推断〕。
- **每次调用的耗时**：VLAKit 文档里没有时延数据〔档：查无〕。
  - 前缀每次调用只算一次，去噪每步只重跑专家（`decoder.py:807-808, 877-892`）〔码〕。
  - 估计 8B 前缀约 25–60 ms，每步去噪约 7–10 ms。10 步不编译时约 0.12–0.16 秒（范围 0.08–0.2）〔推断〕。
  - 参照：StarVLA Qwen2.5-7B 低负载时每次 54–67 ms（STEP2_SMOKE_20261020.md:188-192）〔测〕。满负载评测时，客户端看到每次约 0.97 秒（Qwen2.5）和 1.04–1.11 秒（Qwen3 StarVLA），主要是 CPU 争抢，GPU 平均利用率只有 3.5%〔测〕。
- **每回合的耗时。**
  - DS-B-030-s1 平均每回合约 81 次调用（944 回合共 76,200 次）〔测〕。X1 约 92 次，因为失败回合跑满时限〔测〕。
  - 两次调用平均间隔 14.0 秒，其中推理约 1.1 秒、仿真约 11.7 秒〔测〕。
  - 每回合 GPU 时间约 10–15 秒〔推断〕。
  - 每回合墙钟平均 1,134–1,263 秒，主要花在仿真上。96 进程时每步约 0.79–0.82 秒〔测〕。
  - 所以每回合耗时应该和 StarVLA 差不多〔推断〕。
- **服务端显存**约 19–21 GB〔推断〕。StarVLA Qwen2.5-7B 服务是 16,837 MiB〔测〕。

### A5.6 克隆

- 本地 Windows 上有一份浅层稀疏克隆，在 `C:\Users\user\Desktop\code\ext\NEO\VLAKit`，提交 57e21d6。这是按你批的"只拉代码"做的（STATUS.md 第 3 行"批了克隆 VLAKit 的代码"）〔档〕。里面只有代码、配置、文档和几张文档配图，没有权重和数据。
- 这份克隆只读过，没运行、导入或安装任何东西。
- 下面三件还要你另批：
  - 把 VLAKit 拷进集群；
  - 装 hydra-core（装进新建的环境，不动评测正在读的 StarVLA 环境）；
  - 下载 HF 上发布的 manifest 样例（超出"只拉代码"，还要拷进集群）。
- 只有丙需要这三件，甲和乙都不需要。

## A6 卡时和显存

### A6.1 VLAKit 自带的配方不能直接用

- 它的配方是按 128 张卡设计的〔码〕：
  - 第一阶段（冻结主干）：全局批 8,192，113,400 步，学习率 2e-4，不开梯度检查点，EMA 0.99（`scripts/stage1.sh:13-15, 21, 28-35`，`config/schedule/pretrain.yaml:1-21`）。
  - 真机微调：批 128，3 万步（`scripts/posttrain_real.sh:14-16, 54-62`）。
- 批 8,192 的话，我们 4.44M 帧只够跑约 542 步〔推断〕。

### A6.2 实测参照〔测〕

| 运行 | 设置 | 秒/步 | 卡时 | 来源 |
|---|---|---|---|---|
| E2 30% | Qwen2.5-7B 全微调，批 64，8 卡 | 0.75 | 20 万步约 335 | C1_RESULTS_20261104.md:70-72；T_DESIGNS_20261026.md:3（metrics.jsonl 的 timing/model） |
| DS-Q3B/S/RS/RD-100-s1 | Qwen3-VL-8B 全微调，同一配方 | 0.785–0.80 | 20 万步约 349–356 | stage_times.tsv（156,921–160,059 秒） |
| T2-B-030-s1 | Qwen2.5-7B LoRA（反传穿过整个语言模型，不开检查点），每卡 8 | 0.39 | 10 万步约 88 | T2_RESULTS_20261030.md:47；stage_times.tsv 39,450 秒 |
| T2 显存峰值 | 同上 | — | 62,520 MiB（约 61 GiB） | gpu.csv |
| X1-B-030-s1 | Qwen2.5-7B 冻结 + 回归头，每卡 8 | 0.32 | 5 万步约 36 | X1_RESULTS_20261029.md:30-34 |
| X1 显存峰值 | 同上 | — | 25.4 GB | X1_RESULTS_20261029.md:32 |

DS-Q3* 用的是 100% 数据，但每步耗时和数据量无关〔推断〕。

| 评测 | 回合 | 卡时 | 来源 |
|---|---|---|---|
| DS-B-030-s1，第 20 万步 | 944 | 26.7（墙钟 × 8） | summary.json |
| DS-Q3V4S3 / DS-Q3DCKS2，各个检查点 | 943 | 25.0–26.5 | summary.json |
| DS-Q3* 训练作业内评测 | 944 | 28.0–29.0（12,603–13,037 秒） | stage_times.tsv |
| T2-B-030-s1（成功率 12.6%） | 944 | 30.5（墙钟 13,705 秒 × 8） | 核查时读的评测记录 |
| X1-B-030-s1（944 回合只成功 1 个，失败回合跑满时限） | 944 | 31.7 | summary.json |

### A6.3 一次训练的估计

条件：8 卡，20 万步 × 批 64，共 1,280 万个样本。全部〔推断〕，按 StarVLA 实测缩放。

推法〔推断〕：X1 每步 0.320 秒，乘 Qwen3 对 Qwen2.5 全微调的实测比 1.05–1.07（0.785–0.80 对 0.75 秒），再加专家、优化器和梯度同步约 0.03–0.06 秒，得 0.37–0.41 秒/步，即 165–182 卡时。
- 8B 对 7B：语言部分非嵌入参数 6.95B 对 6.53B（1.06 倍），视觉 0.41B 对 0.63B，前向比约 1.0–1.06，和实测比一致。
- 717M 专家对 129M 回归头：专家前向加反传每卡每步约 1.9 TFLOP，只有主干前向（约 33 TFLOP）的约 6%；多出来的主要是 0.72B 参数的优化器和梯度同步。
- VLAKit 引擎的开销没量过，另有约 ±30% 的不确定。

| 设置 | 秒/步 | 墙钟 | 训练卡时 |
|---|---|---|---|
| **只训专家，StarVLA 式输入（推荐）** | 0.37–0.41 | 21–23 h | 约 175（160–190） |
| frozen_vlm，StarVLA 式输入（按 T2 实测推） | 0.43–0.50 | 24–28 h | 190–225 |
| 只训专家，VLAKit 自己的布局 | 0.85–1.0 | 47–56 h | 380–450 |
| frozen_vlm，VLAKit 布局，开检查点 | 2.0–2.4 | 111–133 h | 900–1,050（按旧倍数估，没按 T2 重算，可能偏高） |

- frozen_vlm 那行按 T2 实测推：反传穿过冻结的 7B，每步 0.39 秒，是只前向（X1 0.32 秒）的 1.2 倍。VLAKit 自己的引擎不开编译时可能更高〔推断〕。
- 甲（QwenPI_v3）：复制数设 2 时一次训练约 160–210 卡时；设缺省 16 时约 200–490 卡时〔推断〕。见"要你定的事"第 2 项。
- **"至少 1 个 epoch"的口径。** 1,280 万个样本是 4,442,662 帧的 2.88 遍；StarVLA 自己的计数是 1.18 个 epoch〔测，E2_RESULTS〕。两种口径都满足"至少 1 个 epoch"。计划里要写明用哪种口径，四个起点用同一个样本量。
- **不能把 skip map 离线缓存。** 每帧约 44 MB，30% 子集要约 195 TB〔推断〕。
- 第一次正式训练前，先做 8 卡 200 步试跑，量每步耗时和显存峰值（D1-T0）。
- 丙的每步时间还取决于三项 VLAKit 设置，上面的估计没算进去〔码〕：
  - ①EMA：pretrain 配方的启动脚本默认开 EMA 0.999（`scripts/train.sh:220-231`）；第二阶段配方把它放在 CPU 上，用 16 个线程更新（`loop.py:89, 295-299`，`stage2.sh:42-43`），而 1 卡作业只有 14 个 vCPU。推荐关掉 EMA（`trainer.ema_decay=null`），或者放在 GPU 上。
  - ②编译：VLAKit 自己记录的是 eager 每步 4,048 ms，编译后 2,402 ms（`compiled_layers.py:6`），差 1.7 倍，比 ±30% 还大，试跑时两种都要量。
  - ③`keep_skipped` 默认是 3（`config/train.yaml:77`），遇到非有限梯度时最多另存 3 份，不受 keep_checkpoints 管，建议设 0（`stage2.sh:45` 也是 0）。

### A6.4 显存（每张 H800 80 GB）〔推断〕

固定部分：
- 冻结的 bf16 权重约 16–17.5 GB。
- 只训专家时，可训练参数约 0.72B，每个约 24 字节（fp32 权重、梯度、AdamW 两个矩、DDP 桶、EMA），约 17.3 GB。

| 输入 | 模式 | 每卡批 | 显存 |
|---|---|---|---|
| StarVLA 式（约 300 token） | 只训专家 | 8 | 约 43 GB |
| StarVLA 式 | 只训专家 | 32 | 约 55 GB |
| StarVLA 式 | 只训专家 | 64（单卡全批） | 约 68 GB |
| VLAKit 布局（约 700 token） | 只训专家 | 8 | 约 47 GB |
| StarVLA 式 | frozen_vlm | 8 | 约 60–80 GB（T2 在 7B 上实测峰值约 61 GiB，那里还没有约 17 GB 的专家优化器状态），可能要开检查点 |
| VLAKit 布局 | frozen_vlm，不开检查点 | 8 | 约 81 GB，很可能爆显存（没按 T2 重算） |
| VLAKit 布局 | frozen_vlm，开检查点 | 8 | 约 51 GB（没按 T2 重算） |
| 甲 QwenPI_v3 | 复制数 2 | 8 | 约 30 GB |
| 甲 QwenPI_v3 | 复制数 16 | 8 | 约 55 GB |

- 只训专家的各种设置、甲的两种复制数都放得进一张卡；frozen_vlm 可能要开检查点。4 卡或 8 卡只是缩短墙钟，卡时基本不变〔推断〕。
- 8 卡时 DDP 每步要同步约 2.9 GB 梯度，约 30 ms，不到每步的 10%〔推断〕。所以 1 卡试跑量不到 8 卡的每步时间。
- 甲在 8B 下的显存还没实测，在试跑里量。

### A6.5 评测

- 评测卡在 CPU 仿真上，策略每执行 16 步才调用一次〔测〕。
- 每回合墙钟：中位数 639 秒，90 分位 3,244 秒，最长 4,285 秒〔测〕。
- 换成流匹配策略后，每次调用约 0.08–0.2 秒。DS-B-030-s1 第 20 万步评测里，两次调用平均间隔 14.0 秒，其中推理约 1.1 秒，仿真约 11.7 秒（944 回合、76,200 次调用）〔测〕。每个服务带 12 个仿真进程时，服务利用率不超过约 20%〔推断〕。
- 每次约 30 卡时（26–33），成功率越低越贵：X1（0.1%）31.7，T2（12.6%）30.5，成功率 28–36% 的 25.0–26.7〔测〕。底座在 15% 闸门附近时约 30–31〔推断〕。5 次合计约 150（130–165）〔推断〕。
- 坏回合重跑和扫条图，每次不到 1 卡时〔推断〕。

### A6.6 分阶段的总量〔推断〕

| 阶段 | 内容 | 新增卡时 | 累计 | 在这里停的条件 |
|---|---|---|---|---|
| 0 | 试跑（训练 + 评测） | 约 3–6 | 约 5 | 试跑不正常 |
| 1 | 底座 种子 1 | 约 205 | 约 210 | 底座 < 15% |
| 2 | V4 + 底座 种子 2 | 约 410 | 约 620 | V4 − 底座 < 5 个百分点 |
| 3 | SFT + S0 | 约 410 | 约 1,030 | — |

- 全程约 1,030（930–1,120；如果引擎开销按 ±30% 算，范围放宽到约 750–1,300）。
- 换成别的设置：
  - 甲用缺省复制数 16：每起点约 230–520。
  - 丙用 frozen_vlm：每起点约 220–255，全程约 1,100–1,300。
  - 丙用 VLAKit 布局：每起点约 430，全程约 2,150。
- 墙钟：用 8 卡跑，每个阶段约 22 小时训练加 3.5–4 小时评测。三个阶段约 3.2 天，再加排队。用 1 卡跑约 3–4 周。

### A6.7 并行和排队

- 四个起点加底座第二个种子，共 5 次训练，互相独立，技术上能并行〔推断〕：
  - 每次 8 卡：全部并行要 40 张卡，约 1 天。
  - 每次 4 卡：要 20 张卡，约 2 天。
  - 每次 1 卡：要 5 张卡，每次约 7–8 天。
- 分阶段的设计最多同时两次训练，要 16 张卡。
- 只投 zoetrope，不碰 bq_tp_mmxbf16 和 bq-ready-*（任务书规则）。
- 8 卡作业要等一台整机空出来。11-03 那次，zoetrope 上 1 卡作业能插到排队的 8 卡作业前面〔档，集群记录〕。
- 1 卡 sco 作业有约 14 个 vCPU。VLAKit 默认 8 个加载进程，能不能喂上约每秒 22 个样本，要在试跑里查〔推断〕。

### A6.8 存盘

- VLAKit 每次存盘都带冻结主干，约 25–30 GB〔推断〕。
- 默认每 1,000 步存一次，20 万步要写 200 次，总写入约 6 TB。
- 保留 2 份时，每次训练常驻约 50–60 GB〔推断〕。`keep_skipped` 3 另见 A6.3。
- 建议改成每 1–2.5 万步存一次。
- 配额现在约 17–19 / 22 TB〔档，盘线记录〕。
- 甲（StarVLA）存盘带冻结主干〔测，X1 走同一条存盘路径〕：X1-B 的 final_model 是 16.8 GB，续训全状态每份 35.0 GB。换成 8B，估计只含权重的存盘约 19 GB，全状态约 45 GB（模型状态约 38 GB，0.69B 可训练参数的优化器约 8 GB）〔推断〕。我们的入口只留最新一份全状态。

## A7 其余风险（接正文 1–5）

6. **VLAKit 发布脚本的冻结模式会反传穿过主干**（丙的 frozen_vlm）。每步时间约 1.2–1.5 倍（T2 实测 1.2 倍），冻结主干还会读到新训出来的 token〔码 + 推断〕。
7. **主干包装写错了也不报错。** 丙要自己写位置编码、掩码和 DeepStack，任何一处写错都会悄悄改掉四个起点的冻结表征。必须拿包装出的键值和普通 HF 前向对照〔推断〕。乙用 HF 自己的前向，这个风险小一些。
8. **专家的旋转编码约定。** 主干的键是旋转过的，所以专家必须按 Qwen3 整头旋转。照搬 NEO 的半头旋转不会报错，但结果是错的（`expert.py:73-87`）〔码 + 推断〕。乙和丙都有这个风险。
9. **丙的数据侧。**
   - 没有 manifest 样例，只能照约 100 处拒收检查反推；
   - control_mode 要打补丁；
   - 拟合检查要关掉；
   - 回合末窗口有约 2.7% 的差异；
   - 7,500 个回合的帧数没逐个查；
   - 外参约定没用仿真核对。
10. **信任和环境（丙）。**
    - VLAKit 是外部代码，拷进集群前要审查；
    - StarVLA 环境缺 hydra-core，装它要联网，要装进新建的环境；
    - manifest 样例要从 HF 下载；
    - 这三件都要你批。
    - 底座权重只能在 sco 作业里读。
11. **专家大小随主干变。** VLAKit 接 36 层约 717M，是 X1 回归头的 5.5 倍，也和 VLAKit 自带 NEO-2B 配置的专家（约 440M）不一样大；甲约 0.69B，是 5.3 倍〔推断〕。
12. **排队。** 只能投 zoetrope。8 卡作业要等整机。1 卡跑每次约 7–8 天，还可能被加载速度拖住。
13. **存盘占配额。** VLAKit 每次存盘约 25–30 GB，存盘频率不改会写很多；`keep_skipped` 不设 0，最坏每次训练再多约 75–90 GB〔推断〕。
14. **改在用的文件。** 甲、乙要改评测合同、闭环客户端和 StarVLA 补丁。改之前要确认没有评测在读它们〔未核实：现在有没有评测在读〕。

## A8 来源

**VLAKit**：github.com/EvolvingLMMs-Lab/NEO 的提交 57e21d6，本地浅层稀疏克隆在 `C:\Users\user\Desktop\code\ext\NEO\VLAKit`。只读过，没运行、导入或安装。
- 模型：`neo_vla/model/build.py`、`backbone/neo_llm.py`、`backbone/vision.py`、`expert.py`、`encoder.py`、`decoder.py`、`tokenizers.py`、`policy.py`、`prefix.py`、`compiled_layers.py`。
- 数据：`neo_vla/data/dataset.py`、`reader.py`、`embodiment.py`、`body.py`、`actions.py`、`history.py`、`conditioning.py`、`mixture.py`、`images.py`、`understanding.py`、`registry.py`、`rays.py`、`prompt/render.py`、`prompt/template.py`、`prompt/embodiments.py`。
- 服务和评测：`neo_vla/serve/server.py`、`predictor.py`、`protocol.py`、`loader.py`、`neo_backend.py`、`backend.py`、`client.py`；`neo_vla/bench/client.py`、`libero.py`、`robotwin.py`、`robotwin_policy/deploy_policy.py`。
- 引擎：`neo_vla/engine/runner.py`、`context.py`、`strategy.py`、`loop.py`、`checkpoint.py`、`logging.py`；`neo_vla/constants.py`。
- 配置：`config/model/neo2b.yaml`、`config/arch/skip28.yaml`、`config/mode/frozen_vlm.yaml`、`full_ft.yaml`、`cotrain.yaml`、`config/train.yaml`、`config/data/robot_adaptation.yaml`、`config/schedule/pretrain.yaml`、`finetune.yaml`、`config/embodiments.yaml`、`config/prompt_template/policy.yaml`。
- 脚本：`scripts/stage1.sh`、`stage2.sh`、`posttrain_real.sh`、`train.sh`、`lib.sh`；`train.py`、`pyproject.toml`、`README.md`、`THIRD_PARTY_NOTICES.md`。
- 文档：`docs/training.md`、`preprocessing.md`、`inference.md`、`benchmarks.md`；`data_release/README.md`。

**我们的仓库**：`C:\Users\user\Desktop\code\VAGEN-Lite-downstream`，只读。
- 任务书：`docs/downstream/tasks/TASKS_20261106_D1_D3.md`。
- 参考和协议：`docs/downstream/REFERENCE.md`、`EVAL_PROTOCOL.md`。
- 训练和数据：`downstream/train/recipe.yaml`、`train_entry.py`、`make_experiment.py`、`downstream/demos/manifests/target50_030.json`、`downstream/patches/starvla_4507931_working.patch`。
- 评测：`downstream/eval/closed_loop.py`、`run_eval_ckpt.sh`、`eval_d6.json`；`vagen/envs/robocasa/starvla_contract.py`、`vagen/envs/robocasa/utils/actions.py`；`downstream/rlinf/e3_check.py`。
- 报告：`docs/downstream/reports/` 下的 X1_RESULTS_20261029、E2_RESULTS_20261026、T2_RESULTS_20261030、T_DESIGNS_20261026、C1_DESIGN_20261028、C1_RESULTS_20261104、V4_RECHECK_20261101、X_DESIGNS_20261026、STEP1_INVENTORY_20261020、STEP2_SMOKE_20261020、Q0_G0_DESIGN_20261028、OPENDRAWER_PROBE_20261031、E3_CHECK_20261030、`step1/env_starVLA.json`。
- 进度：`docs/downstream/STATUS.md`（第 3 行克隆批准，第 295 行 D1 和预算的冲突）。这份文件读的时候正在被改，行号会变。

**集群**（只读，在登录节点上做的轻量读取，都加了 nice -n 19），根目录 `/mnt/umm/users/yinbaiqiao/`：
- Qwen3-VL-8B 配置：`models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232/` 下的 `config.json`、`preprocessor_config.json`、`model.safetensors.index.json`、`added_tokens.json` 和分词器文件。
- StarVLA（git 4507931）：`VAGEN-Lite-downstream/third_party/starVLA/` 下的 `starVLA/model/framework/VLM4A/QwenPI.py`、`QwenPI_v3.py`、`share_tools.py`、`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`、`cross_attention_dit.py`、`starVLA/model/modules/vlm/QWen3.py`、`starVLA/training/trainer_utils/trainer_tools.py`、`train_starvla.py`、`starVLA/dataloader/gr00t_lerobot/transform/state_action.py`、`starVLA/dataloader/lerobot_datasets.py`、`examples/simBenchmarks/Robocasa_365/train_files/data_registry/data_config.py`、`examples/simBenchmarks/RoboDojo/train_files/starvla_robodojo_v21_qwenpi_v3_h50_q99.yaml`、`deployment/model_server/tools/websocket_policy_server.py`、`server_policy.py`、`policy_wrapper.py`。
- StarVLA 环境里的代码：`.conda/envs/starVLA/` 下的 `transformers/models/qwen3_vl/modeling_qwen3_vl.py`、`transformers/models/qwen2_vl/image_processing_qwen2_vl.py`、`torch/amp/autocast_mode.py`。
- 实测：`downstream_runs/experiments/` 和 `models_runs/downstream/experiments/` 下 DS-B-030-s1、DS-Q3B/S/RS/RD-100-s1、DS-Q3V4S3-100-s1、DS-Q3DCKS2-100-s1、X1-B-030-s1、T2-B-030-s1、T2-B-030-short 的 `summary.json`、`episodes.jsonl`、`stage_times.tsv`、`run/dataset_statistics.json`、`run/metrics.jsonl`、`gpu_util.csv` / `gpu.csv`、`config.yaml`、`experiment.json`；盘线 b8、b31 的删除记录（X1 存盘大小）。
- RoboCasa365 数据：`probe_spatial/robocasa365/datasets/v1.0/target/`，读了 `meta/info.json`、`modality.json`、`embodiment.json`，`extras/dataset_meta.json`，`episode_*/ep_meta.json`，`model.xml.gz`，再加一个 61,366 行的动作样本；核查时另抽 1,000 回合、593,827 行量拟合检查的 p99（单线程，临时文件已删）。

**论文**：NEO-VLA，arXiv 2611.06918。11.4 和 4.5 个百分点这两个数取自任务书原文，论文本身没读。

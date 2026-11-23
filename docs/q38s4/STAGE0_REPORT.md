# C 线阶段 0 报告：只读盘点（2026-11-22）

分支 `qwen38-s4`，工作树 `/mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38-s4`。阶段 0 只动了两样东西：本线自己的目录 `q38/s4/`，以及两次 2 机 NCCL 测试任务（每次几分钟，已删）。没有碰任何在跑的任务，也没读任何评测数。

## 结论

1. **HF 通路通了。** 经 hf-mirror、不走代理，单连接约 4.9 MB/s。README 和 `vsi_590k.jsonl` 已下好，sha256 和 LFS 记录一致。236 GB 的视频包**还没下**，等你同意（第 7 节问题 2）。
2. **多机可行，但要换镜像。**
   - sco 的 PyTorch 多机作业（`--worker-nodes=2`）一次提交就跑通了，两台机器 16 张卡组成一个通信组。
   - 本线原定的镜像 `wc-dev:260617` 里没有 IB 用户态库 `libibverbs`。NCCL 因此退回到以太网，跨机 all-reduce 的总线带宽只有 **2.5 GB/s**，训 27B 不可用。
   - 换成集群上别人跑 4 机任务用的 `verl:cu129_…_0709` 镜像后，走上了 IB + GPUDirect RDMA，带宽 **169 GB/s**，是单机 NVLink（213 GB/s）的 79%。
   - Python 环境仍用共享盘上的 q38 env，换镜像只换系统库。这一条要你点头（第 7 节问题 4）。
   - H800 池的 2 机测试排了 18 分钟没凑齐两台机器，怕挡住主线的任务就删了，没测到带宽。见第 4 节。
3. **官方配方已逐行核对**，有三处是原文没给、必须由我们定的：
   - NFP 子集是哪 29 万条：公开 jsonl 里**没有** `nfp` 标记，也没有名单。
   - 头的初始化：官方代码里 `trunc_normal_` 未定义，原样跑会报错。
   - NFP 样本保留了约 6–7 个模板 token 的文本损失。

   第 2 节是逐项清单，第 6 节是预登记要用的差异表。
4. **27B 侧按 D4 定的设置**：
   - 2 fps 抽帧，相邻 2 帧合成 1 个时间单元（= 1 秒）。
   - 每单元约 256×256 像素，即 64 个视觉 token；最多 128 个单元，共 8,192 个视觉 token。
   - 同一段 128 秒视频，用 27B 的默认视频配置会得到 12,288 个视觉 token。
   - 27B 没有把编码器中间层注入语言模型的机制（`deepstack_visual_indexes` 为空），冻结编码器即可。
5. **磁盘**：配额 22 TB，已用 15 TB，**剩 7 TB**。文档写的是腾出 10 TB，实际少 3 TB。本线计划用不到 3 TB，够用。
6. **还要你定的**：第 7 节，共 4 个问题。另有几项我按"照搬官方 / 最小改动"先定了，列在第 7 节末尾，你可以推翻。

## 1. HF 通路（阶段 0 第 0 项）

| 通路 | 32 MB 区间下载速度 | 结论 |
|---|---|---|
| 已配代理 → huggingface.co | 0.23 MB/s | 能通，太慢 |
| 已配代理 → hf-mirror.com | 0.21 MB/s | 同上 |
| **不走代理 → hf-mirror.com** | **4.9 MB/s（单连接）** | 用这条 |

- 做法：`env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY` 加上 `HF_ENDPOINT=https://hf-mirror.com`。大文件会被重定向到 xet CDN（cas-bridge.xethub.hf.co）。
- 记忆里 10-18 测过 30–36 MB/s，今天单连接慢得多，236 GB 要靠多连接并行。估计 2–14 小时，取决于并发数和镜像站状态。
- 已下：
  - `q38/s4/data/VSI-590K/README.md`
  - `vsi_590k.jsonl`：276,557,539 字节，sha256 `025355ea…d231`，与 LFS oid 一致。

**待下清单**：目标目录 `/mnt/umm/users/yinbaiqiao/q38/s4/data/VSI-590K/`。HF 给的校验值是 sha256（LFS oid），没有 md5。

| 文件 | 字节 | GB | sha256（LFS oid） |
|---|---:|---:|---|
| arkitscenes.tar.gz | 90,154,741,781 | 90.15 | 4c54a042dd9c3ca808a61d05536598878d93a1b395b668551daec19895da9728 |
| ytb_roomtour.tar.gz | 55,261,730,714 | 55.26 | 98699f63d65223e04034b372d56ff53919819ab90885e2a3f0c7ce317d3e0946 |
| scannet.tar.gz | 27,632,592,849 | 27.63 | 8a8a07b37ad3e52c286bdc60118c6445a6543f19b9423007d60a9485912f9ab2 |
| adt.tar.gz | 23,895,961,889 | 23.90 | bf9fa6b79ad9e529210eb57df114372de2ea452fe0871d569036132e05f87c1b |
| scannetppv2.tar.gz | 21,438,045,428 | 21.44 | 3ef0b2f7aa0cc30ad2508a19a52af7239e0160be809d7341ee57198777d7ea3b |
| hypersim.tar.gz | 6,775,393,985 | 6.78 | 8376d276022beac3ab2a8af8a4ef23890a784c9c7da126cba6e1eca1a763a847 |
| procthor.tar.gz | 6,523,652,708 | 6.52 | 881d7be20b7d67b4a4bd76f23082c48623c2e68cf02ebb661eeaa1f1bffb7bc6 |
| s3dis.tar.gz | 2,865,916,732 | 2.87 | c97c0dfd60430bef6365f4adfc376c1a94ff0e0ce50909ca84df0e2396ff1b53 |
| robotics.tar.gz | 1,104,873,829 | 1.10 | 9bae84fac90fcfa7f86d4a7dfb75f3798da241d7211ec10a512d2e3c01ba5816 |
| **合计 9 个** | 235,652,909,915 | **235.65** | |

**jsonl 的构成**（全量统计，单进程）：

- 共 590,667 行，与论文一致。
  - 视频问答 374,148 行，来自 5,963 段不同的视频。
  - 图像问答 216,519 行，来自 44,857 张不同的图。论文写 44,858 张，差 1。
- 字段有 `conversations`、`question_type`、`video` 或 `image`，另有 20,092 行带 `id`。**没有 `nfp` 字段。**

| 来源 | 视频问答行 | 不同视频数 | 图像问答行 |
|---|---:|---:|---:|
| scannetppv2 | 138,701 | 856 | |
| scannet | 92,145 | 1,201 | |
| adt | 60,207 | 183 | |
| arkitscenes | 57,816 | 2,899 | |
| procthor | 20,092 | 625 | |
| s3dis | 5,187 | 199 | |
| hypersim | | | 176,774 |
| ytb_roomtour | | | 20,100 |
| robotics | | | 19,645 |

注意两点：

- **ytb_roomtour 在 jsonl 里只有图像行**，所以 NFP 和 E2 的"按来源分开"里没有"房间漫游"这一类，只有上表 6 个视频来源。
- 问题类型共 12 种，最多的是 relative_direction_object（176,991 行）。

## 2. 官方配方清单（阶段 0 第 1 项）

- 依据：集群上的拷贝 `/mnt/umm/users/yinbaiqiao/cambrian-s`，关键文件的字节数与 GitHub main 逐个一致。下文 file:line 相对这个目录。
- 标签含义："照搬"指照搬可行；"改动"指 27B 上必须改；"没写"指原文没写，要我们定。
- NFP：next frame prediction，即预测头；论文叫 LFP。

| 项 | 官方做法 | 证据 | 标签 |
|---|---|---|---|
| 头结构 | `Linear(H,H) → GELU → Linear(H,1152)`，带 bias，与 lm_head 并列；输入是最后一层**经过 final norm** 后的 hidden | cambrian_arch.py:263-268；cambrian_qwen2.py:449-457 | 照搬（H 改成 5120） |
| 头初始化 | 意图是 trunc_normal std 0.02、bias 0，但 `trunc_normal_` **未定义也未 import**，原样跑会 NameError | cambrian_arch.py:269-272 | 改动（用 `nn.init.trunc_normal_(std=0.02, a=-0.04, b=0.04)`，即截断在 ±2σ） |
| 逐 token 还是池化 | **逐 token**：第 t 帧 (h,w) 位置的 hidden 预测第 t+1 帧 (h,w) 的目标；沿帧维 roll(-1) | train_fsdp.py:1691-1698 | 照搬 |
| 掩码 | 去掉最后一帧和每行末尾的换行列；帧之间没有分隔符和时间戳 | train_fsdp.py:1660-1698 | 改动（27B 每单元前有时间戳文本和 vision_start/end，只在视觉 token 上算，见第 3 节） |
| 非方形视频 | 先补成方形，再去掉 padding 对应的 token；16:9 每帧只剩 32 个视觉 token | train_fsdp.py:1633-1688 | 改动（27B 保长宽比缩放，见第 3 节） |
| 目标特征 | SigLIP2 删掉最后一层后的输出（= 倒数第二层），**投影前**，不过 post-LN；27×27 双线性插值到 8×8；不归一化；实际无梯度 | builder.py:107-109；encoder:578-597；cambrian_arch.py:419-475 | 改动（27B 编码器倒数第二个 block 的输出，2×2 合并窗口内取平均；显式 detach） |
| MSE | 1152 维取均值，再按有效 token 做 masked mean | cambrian_qwen2.py:323-331 | 照搬 |
| cos | 逐 token 1−cos，再 masked mean | cambrian_qwen2.py:333-335 | 照搬 |
| 合成 | `CE + 0.1·MSE + 0.1·cos` | cambrian_qwen2.py:455-482；lfp_s4.sh:62-64 | 照搬 |
| 哪些样本算 NFP | 只算 `"nfp": true` 的视频样本；QA 视频、图像不算 | train_fsdp.py:1446、1691-1698 | 照搬 |
| NFP 样本的文本损失 | 视频之后全部 IGNORE；但视频之前约 6–7 个模板 token（`\n`、`<\|im_start\|>`、`<\|im_end\|>`）仍算 CE | train_fsdp.py:1664-1672、878-945 | 改动（我们 NFP 样本完全不算文本损失，见第 7 节） |
| NFP 取帧 | 1 fps；超过 128 帧时取**随机连续 128 帧窗口**；短视频有几帧用几帧 | utils.py:73-88 | 照搬（换算成 27B 的单元，见第 3 节） |
| QA 视频取帧 | 不论长短，linspace 均匀取**恰好 128 帧** | utils.py:50-71 | 照搬（取 256 帧 = 128 单元） |
| NFP 子集 | jsonl 用 `"nfp": true` 标；论文说约 29 万条视频样本；**公开 jsonl 没有标记，也没有名单** | train_fsdp.py:1278 等；论文 4.1 | 没写（见第 7 节问题 3） |
| 超长 | 没有截断规则，超长会让进程退出；方形视频 9,216 token < 11,264 | train_fsdp.py:1443-1446、1890-1894 | 改动（我们 8,192 视觉 token + 约 1k 时间戳文本，序列上限 16,384，不会超） |
| 优化 | lr 1e-5、warmup 3%、cosine（min_lr 0）、wd 0、1 epoch、AdamW；头、projector、LLM 同一个 lr | lfp_s4.sh:31-42；cambrian_trainer.py:286-330 | 照搬 |
| batch | 每卡 8、累积 1；全局批取决于芯片数，脚本没写；论文写 256 | lfp_s4.sh:29-34 | 照搬论文的 256 |
| 序列上限 | lfp_s4 是 11,264，普通 s4 是 16,384 | lfp_s4.sh:44；s4.sh:43 | 用 16,384（文档 S2 要求；我们的序列 < 11,264，取哪个都不截断） |
| 梯度裁剪 | 脚本没设，即 transformers 默认 1.0；在 XLA FSDPv2 上是否生效没核实 | pyproject.toml:16 | 没写（我定 1.0，见第 7 节） |
| 训练范围 | 视觉塔冻结；LLM 全参、projector、image_newline、nfp_head 都训练 | train_fsdp.py:2340-2380 | 照搬（27B：LLM + merger + 头训练，视觉 block 冻结） |
| `connector_only True` | 只表示"不在 LLM 层里插 SVA 交叉注意力"，**不是**只训连接器 | cambrian_qwen2.py:186；cambrian_arch.py:219-223 | 照搬（27B 无此结构，忽略） |
| 存盘 | 每 250 步、只留 1 个 | lfp_s4.sh:36-38 | 改动（按文档：每 500 步存 bf16 HF 权重 + 1 个完整检查点） |
| 硬件 | 训练路径只支持 TPU（XLA SPMD），目标拼装用 XLA 手动 sharding 内核 | cambrian_arch.py:405-480 | 改动（GPU 上重写） |
| 推理时的惊讶度 | 1 − 逐 token cos(预测, 实际下一帧 8×8 特征) 的平均；第 0 帧固定为 1 | cambrians_vsr.py:385-509 | 照搬（E2 的惊讶度曲线用这个定义） |

另外两点：

- 官方 `nfp_head` 开关**不会**顺带打开 MIV。MIV 是所有视频共用的 64 token 数据路径，NFP 依赖它，不是它的开关。
- 集群拷贝里的 `cambrian_qwen2.py` 在 2026-09-13 被改过 mtime，但字节数和 nfp 相关行都与 main 的 blob 9dee0ef 一致。

## 3. 27B 侧参数和 D4 设置（阶段 0 第 2 项）

**Qwen3.8-27B**（model_type `qwen3_5`）

- **语言模型**：hidden 5,120；64 层，其中 48 层 linear attention、16 层 full attention；24 个注意力头，4 个 KV 头；词表 248,320；位置上限 262,144。
- **视觉编码器**：27 个 block，宽 1,152，16 头，FFN 4,304，激活 gelu_pytorch_tanh。patch 16，空间合并 2（2×2 个 patch 合成 1 个 token）。**有时间合并**：temporal_patch_size 2，相邻 2 帧合成 1 个时间单元。merger 把 1,152 维投影到 5,120。`deepstack_visual_indexes` 为空，即没有中间层注入。
- **视频默认配置**：Qwen3VLVideoProcessor，默认 2 fps。像素上限按**所有帧合计**算：`longest_edge` 25,165,824，`shortest_edge` 4,096。每个视觉 token 覆盖 32×32 像素 × 2 帧，所以默认最多 25,165,824 ÷ 2,048 = **12,288 个视觉 token**。
- **视频的 prompt 布局**：每个单元写成 `<x.x seconds><|vision_start|>（若干视频 token）<|vision_end|>`。时间戳是文本 token。

**D4 的具体设置**

| 项 | 设置 | 说明 |
|---|---|---|
| 抽帧 | 2 fps | 2 帧合 1 个单元，1 单元 = 1 秒，对应官方"1 fps、预测下一帧" |
| 每单元分辨率 | 面积约 256×256 = 65,536 像素，保长宽比，每边取 32 的倍数 | 每单元约 64 个视觉 token。4:3 的 ScanNet 帧是 288×224 → 9×7 = 63 个；方形是 8×8 = 64 个 |
| 单元数上限 | 128 | 视觉 token ≤ 128 × 64 ≈ 8,192 |
| NFP 样本 | 2 fps 下取**随机连续 256 帧窗口**（128 单元）；不足 128 秒的视频全取，帧数取偶数 | 对应官方随机连续 128 帧 |
| QA 视频 | 整段视频 linspace 均匀取 256 帧（128 单元） | 对应官方均匀取恰好 128 帧 |
| 时间戳文本 | 照常保留，每单元约 7–8 个 token，128 单元约 1,000 个 | 不进预测目标 |
| 预测目标 | 编码器倒数第二个 block（blocks[25]）的输出，每单元 4×64 = 256 个 patch × 1,152，在每个 2×2 合并窗口内取平均 → 64 × 1,152，detach | 编码器的 patch 顺序本来就按 2×2 合并窗口排列，平均后与语言模型看到的 64 个视频 token 逐个对齐 |
| 对齐 | 第 t 个单元第 k 个视频 token 的 hidden 预测第 t+1 个单元第 k 个 token 的目标；最后一个单元不算 | 各单元 token 数相同（同一视频分辨率固定），逐位对应 |
| 头 | `Linear(5120,5120) → GELU → Linear(5120,1152)`，约 3,212 万参数 | |

**默认配置和本线的对照**（同一段视频）：

| 视频长度 | 默认配置 | 本线（D4） |
|---|---|---|
| 128 秒 | 256 帧 → 128 单元 × 约 96 token = 12,288 | 128 单元 × 64 = 8,192 |
| 6 分钟 | 720 帧 → 360 单元 × 约 34 token = 12,288 | QA：128 单元 × 64 = 8,192（均匀抽）；NFP：随机 128 秒窗口，8,192 |
| 30 秒（640×480 源） | 60 帧 → 30 单元 × 300 token（总像素没到上限，保留原分辨率）= 9,000 | QA：128 单元 × 63 ≈ 8,064（短视频重复帧，同官方）；NFP：30 单元 × 63 ≈ 1,890 |

默认配置的单帧分辨率随视频长度变化，从每单元约 34 个 token 到 300 个不等。D4 固定每单元约 64 个 token，单元数和每单元 token 数都和 Cambrian-S 对齐。

## 4. 训练器和集群（阶段 0 第 3、4 项）

**SFT 训练器（27B 线在用的那套）**

- **框架**：LLaMA-Factory b4e4d3fc + accelerate FSDP1（`vagen/q38/configs/sft/accelerate_fsdp_qwen35.yaml`，`num_machines 1`），模板 qwen3_8，`enable_thinking false`（4 个 token 的空思考块），开 `sync_each_batch` 和 `expandable_segments`。
- **视频**：LF 支持 qwen3_5 的视频输入，`check_lf_qwen35` 已验证过视频记录和时间戳。
- **多机**：需要让 accelerate 的 `num_machines` / `machine_rank` / 主节点地址取 sco 给的变量（`RANK`、`WORLD_SIZE`、`MASTER_ADDR`、`MASTER_PORT`；见下）。这是阶段 2 第 6 项，只加新文件，不改现有 yaml。
- **NFP 损失**：要在 LF 的模型 forward 和损失上加钩子。阶段 2 用新文件做包装，不改 LF 源码，diff 先给你看。
- **16,384 序列能否单机放下**：C0 SFT 的冒烟记录（峰值约 62 GiB 已分配 / 73 GiB 预留）是图像轨迹序列，不能直接推到 16k 视频序列。由 S2 实测。

**sco 多机作业**

- 用 `--training-framework=pytorch --worker-nodes=N --worker-spec=N4lS.Iq.I80.8`。
- 每台机器拿到 `RANK`（机器序号 0..N−1）、`WORLD_SIZE`（机器数）、`MASTER_ADDR`（master 主机名）和 `MASTER_PORT`（23456）。
- 集群上别人在跑 2–6 机的任务，例如 zoetrope 上一个 4 机任务，H800 上一个 6 机任务。

**2 机 all-reduce 测试**

- 脚本：`vagen/q38s4/tools/allreduce_bench.py` + `run_allreduce.sh`，提交 2e60320d。
- 测法：bf16，每个消息大小先预热 5 次，再取 20 次的中位数。
- busbw 是总线带宽，即 nccl-tests 的口径：算法带宽 × 2(n−1)/n，可直接和链路带宽比。
- all-reduce 是 FSDP 每步都要做的跨卡求和。

| 池 / 镜像 | 跨机传输 | 2 机 16 卡 busbw（8 / 64 / 256 / 1024 MB） | 单机 8 卡 busbw（同上） | 排队到开跑 |
|---|---|---|---|---|
| zoetrope / wc-dev:260617 | **NET/Socket（eth0）**：缺 `libibverbs.so` | 2.5 / 2.5 / 2.4 / 2.5 GB/s | 89 / 175 / 203 / 213 GB/s | 约 7 秒 |
| zoetrope / verl:cu129_…_0709 | **NET/IB GDRDMA** | 74 / 152 / 164 / **169** GB/s | 103 / 181 / 205 / 213 GB/s | 约 1 分钟内 |
| H800 / verl:cu129_…_0709 | 未测到 | — | — | **排了 18 分钟没开跑**（11:29–11:47 UTC），已删 |
| H800 / wc-dev:260617 | 未测到 | — | — | 排了 7 分钟没开跑，确认要换镜像后删掉 |

- H800 上同时要两台空的 8 卡机器，18 分钟内没凑齐。
- 为什么不一直排：记忆里记着，H800 上先交的 8 卡任务会挡住后交的 1 卡任务。本线的多机任务排着不动，可能拖住主线的任务，所以删了。
- **结论**：本线的多机训练先按 zoetrope 计划。H800 的 IB 应该和 zoetrope 一样（同型号网卡），但没实测。

- 两个池的机器都是 4 张 mlx5 网卡 + bond，驱动 595.71.05，torch 2.13.0+cu130，NCCL 2.29.7。
- 换镜像的道理：q38 env 的 Python、torch、NCCL 都在共享盘上，镜像只提供系统库。`verl:cu129_…_0709` 带 `libibverbs` 和 mlx5 驱动，别人的 4 机任务也用它。
- 风险：其他系统库和 wc-dev 不同。S1 会在这个镜像下把 LF 训练完整跑一遍再往下走。
- 另一条路是请平台在 wc-dev 里装 rdma-core，但那要等别人。

**空闲机器数**：sco CLI 只能看任务列表，不给池的容量，所以看不到空闲机器数。旁证：zoetrope 上两次 2 机任务都在 1 分钟内开跑。8 机能不能凑齐、等多久，要到 S3 才知道。

- 本线任务一律 NORMAL 优先级，和主线一样。任务列表里有 SUSPENDED 的任务，说明平台存在抢占，但同优先级之间没见过。
- 如果 8 机长时间排不上，我会报告，不会自己改成单机跑。

**通用数据和小模型**

- 共享目录里没有 LLaVA-Video-178K。D2 若选"用已有数据替代"，目前找不到现成的。
- lmms-eval（cambrian-s 自带）里有 egoschema、videomme、vsibench 三个任务目录，E5 有东西可评。
- 冒烟小模型：`/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e8…`，和 27B 同属 qwen3_5 结构。没有 Qwen3.8-4B。

## 5. 磁盘（阶段 0 第 5 项）

配额 22 TB，已用 15 TB，剩 **7 TB**，比文档写的 10 TB 少 3 TB。计划用量：

| 目录 | 用途 | 估计 |
|---|---|---|
| `q38/s4/data/VSI-590K/` | 9 个 tar.gz（235.65 GB）+ 解压（0.3–0.5 TB） | 0.55–0.75 TB；压缩包在校验后可删，删前问你 |
| `q38/s4/runs/<arm>/` | 每臂 bf16 权重约 7 个 × 55 GB + 1 个完整检查点约 440 GB | 两臂约 1.6–2.2 TB |
| `q38/s4/eval/` | 评测产物 | < 0.1 TB |
| **合计** | | **约 2.3–3.1 TB，在 7 TB 内** |

## 6. 配方差异表（给预登记 v1 用；第 1.3 节的表之外，阶段 0 新读出的）

| # | 项 | 官方 | 本线 | 原因 |
|---|---|---|---|---|
| R1 | 头初始化 | `trunc_normal_` 未定义（公开代码跑不起来） | `nn.init.trunc_normal_`，std 0.02，截断 ±2σ | 补官方意图 |
| R2 | 目标降采样 | 27×27 双线性插值到 8×8 | 2×2 合并窗口内平均，每单元 256 → 64 | 与 27B 自己的 token 布局逐位对齐 |
| R3 | 目标层 | SigLIP2 删最后一层后的输出 | 27B 编码器 blocks[25] 的输出（倒数第二个 block），merger 之前 | 对应"倒数第二层、投影前" |
| R4 | 时间粒度 | 1 帧 = 1 个预测单元 | 2 帧（temporal patch）= 1 个单元 = 1 秒 | D4 |
| R5 | 非方形视频 | 补方形后去 padding，16:9 只剩 32 token/帧 | 保长宽比，每单元约 64 token | 27B 处理器本来就保长宽比 |
| R6 | 帧间文本 | 无 | 每单元前有时间戳和 vision_start/end，不进目标、不算 NFP | 27B 默认格式，D4 写明保留 |
| R7 | NFP 样本文本损失 | 视频前约 6–7 个模板 token 算 CE | 完全不算 | 那几个 token 是 Qwen2 模板的副作用，不像有意设计 |
| R8 | NFP 子集 | 名单未公开 | 第 7 节问题 3 的规则 | 原文没给 |
| R9 | 存盘 | 每 250 步留 1 个 | 每 500 步 bf16 + 1 个完整 | 文档阶段 2 第 8 项 |
| R10 | 序列上限 | 11,264（lfp_s4） | 16,384 | 文档 S2；我们的序列 < 11,264，不影响截断 |
| R11 | 多机镜像 | TPU | sco 多机 + `verl:cu129_…_0709` 镜像 | wc-dev 缺 IB 库 |
| R12 | 损失归一 | SPMD 下 masked mean 在全局批上算 | 在每次优化步的全部微批上按 token 数合计再除 | 与全局 masked mean 等价 |
| R13 | 训练数据 | VSI-590K + 590K 通用视频 + 120K 通用图像 | 只用 VSI-590K（若 D2 按推荐定） | D2 |

## 7. 要你决定的

1. **D2：确认只用 VSI-590K、不加通用数据？**
   - 我同意文档的推荐。补充一个事实：集群上找不到现成的通用视频指令数据，"用已有数据替代"这条备选实际不可行。
2. **同意下载 236 GB 吗？**
   - 内容：第 1 节的 9 个包，下到 `q38/s4/data/VSI-590K/`。
   - 做法：交一个 1 卡 sco 任务，在里面做下载、sha256 校验、解压和 ffprobe 抽检，登录节点不占核。先测计算节点能不能直连 hf-mirror；不能就退回登录节点，`nice -n 19`、4 个连接。
   - 压缩包校验并解压后要不要删，到时再问你。
3. **NFP 子集按这条规则造，可以吗？**
   - 从 374,148 条视频问答里，固定种子、不放回均匀抽 290,000 条，每段视频保留的比例和原来大致一样。
   - 每条只保留视频路径，标 `nfp: true`，用 D4 的随机 128 秒窗口。
   - 平均每段视频约 49 个窗口，窗口每个 epoch 随机。
   - 从 held-out 的 1% 视频里抽出的条目不进训练。
   - 依据：论文只说是"VSI-590K 里约 29 万条视频样本、1 fps、只算预测损失"。5,963 段视频造出 29 万条，只可能是同一段视频取多个窗口，所以按视频问答行来抽最贴近原文。
4. **多机训练换用 `verl:cu129_…_0709` 镜像，可以吗？**
   - 不换就只能走以太网，2.5 GB/s，27B 8 机训练不可行。
   - Python 环境不变。

**我先定了、你可以推翻的**：

- 梯度裁剪用 1.0：照官方。27B 线用的是 0.3，但本线以复刻为主。
- 每单元保长宽比、约 64 token（R5）。
- NFP 样本不算文本损失（R7）。
- 目标用 2×2 平均（R2）。
- 序列上限 16,384（R10）。

## 8. 状态表（第 2.8 节）

训练任务

| # | 项目 | 任务 / 位置 | 状态 | 等什么 |
|---|---|---|---|---|
| T1 | 冒烟 S1（小模型单机） | Qwen3.5-4B | 未交 | 阶段 2 代码 |
| T2 | 冒烟 S2（27B 单机 20 步） | | 未交 | S1 |
| T3 | 扩展测试 S3（2 / 4 / 8 机） | | 未交 | S2；多机已证可行（换镜像） |
| T4 | 对拍 S4（预测权重为 0） | | 未交 | 阶段 2 |
| T5 | C-NFP 正式（1 epoch，8 机） | | 未交 | 真数据、S1–S4、预登记 v1 |
| T6 | C-noNFP 正式（1 epoch，8 机） | | 未交 | 同上 |

评测任务

| # | 项目 | 状态 | 备注 |
|---|---|---|---|
| E1 | 回答长度 / 速度测试（C-NFP 第一个权重） | 未开始 | |
| E2 | 预测头 held-out 误差 vs 复制基线 | 未开始 | 来源只有 6 个视频来源（无房间漫游视频） |
| E3 | EASI-8 / 探针 / P0 | 未开始 | |
| E4 | 705 道 / 674 道动作题零样本 | 未开始 | |
| E5 | 通用视频 benchmark | 未开始 | 有 videomme、egoschema（cambrian-s 的 lmms-eval） |
| E6 | 快评（每 1,000 步） | 未开始 | |

- **数据**：README 和 jsonl 已下并校验，视频包 0 / 9（等问题 2）。
- **磁盘**：剩 7 TB，配额 22 TB。
- **空闲机器**：CLI 看不到。zoetrope 上 2 机任务 1 分钟内开跑；H800 上 2 机任务 18 分钟没开跑。

**下一步**：等你答第 7 节的 4 个问题。阶段 2 的代码不依赖这些答案，可以先做：只加文件、带单测、每个 diff 先给你看，用 InteriorGS 渲染的漫游视频做假数据，不交 GPU 任务。

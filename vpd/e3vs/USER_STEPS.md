# E3VS：你本人要做的几步（拿 splat 数据）

2026-11-09 改版：下载由你自己在集群上跑，我们看不到、也不碰 token。用户同日说"接下来没有卡时和磁盘的上限"，所以下载不用再等磁盘线。

一共六步：
1. 在浏览器里提交两个表单；
2. 建一个只读 token；
3. 自己把 token 放到集群上；
4. 选下载路线；
5. 自己跑下载命令；
6. 吊销 token，然后告诉我们。

两个表单都是 `gated=auto`，提交后马上通过。两个表单和 token 必须用**同一个** HF 账号。**不要把 token 贴进任何对话。**

固定版本的 E3VS 代码仓库（集群内拷贝，23 MB，不联网）我们已经放好了，你不用管。

---

## 第 1 步：第一个表单（SceneSplat-7K 主表单）

- 先登录 huggingface.co，再打开 https://huggingface.co/datasets/GaussianWorld/scene_splat_7k 。
- 填下面这些字段（字段名是网页上的原文）：

| 字段 | 填什么 |
|---|---|
| `Full name` | 你的姓名 |
| `Current affiliation` | 你的单位 |
| `Type of Affiliation` | 下拉框，三选一：`Academia` / `Industry` / `Other` |
| `Institutional email` | 单位邮箱 |
| `Please explain your intended research use` | 建议填：`non-commercial academic research: evaluating embodied active-viewpoint agents on E3VS-Bench (ScanNet++ scenes); no redistribution of renders` |

- 勾上两个框：
  - `I agree to all the terms outlined by original datasets`
  - `I agree to use this the data for non-commercial, academic purposes only`
- 提交。会马上通过。

## 第 2 步：第二个表单（splat 数据集）

- 打开 https://huggingface.co/datasets/GaussianWorld/scannetpp_v2_mcmc_3dgs 。
- 这里只有两个勾选框：
  - `I have read and agreed to the SceneSplat-7K main dataset terms`
  - `I confirm this data will be used only for non-commercial research/educational purposes`
- 勾上，提交。账号要和第 1 步相同。
- E3VS 的 README 里写的是旧名 `scannetpp_v2_mcmc_1.5M_3dgs`，它会自动跳到这个新名字。

## 第 3 步：建一个只读 token

- 打开 https://huggingface.co/settings/tokens/new?tokenType=fineGrained 。
- 名字随意，比如 `fujian-e3vs-read`。
- 权限只给读，范围只限仓库 `GaussianWorld/scannetpp_v2_mcmc_3dgs`。
- **不要给任何写权限。**

## 第 4 步：把 token 放到集群上

在你自己的终端里运行。`read -rs` 不回显，token 不会出现在屏幕上，也不会进 shell 历史：

```bash
ssh fujian
mkdir -p -m 700 ~/.secrets
(umask 077; read -rs T; printf %s "$T" > ~/.secrets/hf_token_e3vs; unset T)
chmod 600 ~/.secrets/hf_token_e3vs
```

- 运行第三行后，粘贴 token，按回车。
- 文件路径是 `/mnt/umm/users/yinbaiqiao/.secrets/hf_token_e3vs`，下载脚本只读这个路径。
- 脚本会检查：文件存在、非空、权限是 600、属于你本人。任何一条不满足就停下，不会下载。

## 第 5 步：选路线，然后自己跑下载

**路线二选一**
- **默认：hf-mirror.com。** 按 10 月实测的 30 MB/s 推算，35 GB 约 20–40 分钟（推断，没实测过这个数据集）。这是第三方镜像，**它的运营方能看到你的 token**，所以 token 要只读、只限这一个仓库，用完马上吊销。
- **备用：`ROUTE=hf`。** 直连 huggingface.co，走公司默认代理，token 一直在 TLS 里。但很慢：10 月测过每个连接约 0.1 MB/s，35 GB 要约 97 小时，基本不可行。
- 第三种：在集群外下好再传上来。要这样做的话，先告诉我们，我们给你文件清单。

**命令**（在登录节点上跑；脚本只开一个进程，主要占网络，约 1 个核，已加 `nice -n 19`）：

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd
bash vpd/e3vs/download_splats.sh check      # 只发一个带 token 的 HEAD 请求，不下载
nohup bash vpd/e3vs/download_splats.sh splats all \
  > /mnt/umm/users/yinbaiqiao/eval_external_assets/vpd/e3vs/download_run.out 2>&1 &
```

- 走备用路线时，两条命令前面都加 `ROUTE=hf `。
- `check` 返回 2xx/3xx 才往下走；返回 401/403 时脚本会停下，并打印备用路线。
- `splats all` 先下验证和测试的 31 个场景（约 11 GB），再下训练的 68 个场景（约 24 GB），合计 99 个场景、35.05 GB。
- 每个场景都核对文件大小，并写进 sha256 清单。中途断了重跑同一条命令即可，已完成的场景会跳过。
- 看进度：`tail -f /mnt/umm/users/yinbaiqiao/eval_external_assets/vpd/e3vs/download_run.out`。
- 跑完以后运行 `bash vpd/e3vs/download_splats.sh verify all`，看到 `verify set=all OK (99 scenes)` 就是全部到位。
- 不联网先看计划：`DRY_RUN=1 bash vpd/e3vs/download_splats.sh splats all`。这一步不读 token，也不写任何文件。

## 第 6 步：吊销 token，告诉我们

- 到 https://huggingface.co/settings/tokens 把这个 token **吊销（revoke）**。
- 删掉 token 文件：`rm ~/.secrets/hf_token_e3vs`。
- 告诉我们一句"E3VS 数据下好了"。我们会先在登录节点上用 1 个核做核对：`verify all`、渲染输入检查、e3vs_prep 的 dry run。然后提交 J5（e3vs-prep），J5 通过后再提交三个起点的 J6。

**许可**：splat 只能用于非商业研究（ScanNet++ 条款），渲染出来的图不能对外分发。E3VS 的代码是 MIT，标注是 CC BY 4.0。

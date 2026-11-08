# E3VS：需要你本人做的几步（拿 splat 的门槛）

**结论先说**：E3VS 试点要用 `GaussianWorld/scannetpp_v2_mcmc_3dgs` 里 99 个场景的 splat（共 35.05 GB）。这个数据集要过 Hugging Face 的门槛。集群连不上 huggingface.co，所以下面四件事只能你来做：

1. 在浏览器里提交两个表单；
2. 建一个只读 token；
3. 自己把 token 放到集群上；
4. 告诉我们两个决定：token 走不走 hf-mirror，下载能不能在登录节点上跑。

做完以后，你只需要说一句"token 文件放好了"。**不要把 token 贴进任何对话。**

两个表单都是 `gated=auto`，提交后马上通过。两个表单和 token 必须用**同一个** HF 账号。

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
- 说明：E3VS 的 README 里写的是旧名 `scannetpp_v2_mcmc_1.5M_3dgs`。它会自动跳到这个新名字（调查时用 API 核实过）。

## 第 3 步：建一个只读 token

- 打开 https://huggingface.co/settings/tokens/new?tokenType=fineGrained 。
- 名字随意，比如 `fujian-e3vs-read`。
- 权限只给读，范围二选一：
  - 只限仓库 `GaussianWorld/scannetpp_v2_mcmc_3dgs`（推荐）；
  - 或者勾 `Read access to contents of all public gated repos you can access`。
- **不要给任何写权限。**

## 第 4 步：把 token 放到集群上（你自己操作）

在你自己的终端里运行（`read -rs` 不回显，token 不会出现在屏幕上，也不会进 shell 历史）：

```bash
ssh fujian
mkdir -p -m 700 ~/.secrets
(umask 077; read -rs T; printf %s "$T" > ~/.secrets/hf_token_e3vs; unset T)
chmod 600 ~/.secrets/hf_token_e3vs
```

- 第三行运行后，粘贴 token，按回车。
- 集群上 `~` 就是 `/mnt/umm/users/yinbaiqiao`。所以文件路径是 `/mnt/umm/users/yinbaiqiao/.secrets/hf_token_e3vs`，`vpd/e3vs/download_splats.sh` 只读这个路径。
- 脚本会检查：文件存在、非空、权限是 600、属于你本人。任何一条不满足，它就停下，不会下载。
- 做完只要告诉我们"文件放好了"。**不要把 token 贴进对话。**
- 下载完成后，请到 https://huggingface.co/settings/tokens 把这个 token **吊销（revoke）**，再删掉 `~/.secrets/hf_token_e3vs`。

## 第 5 步：需要你拍板的两件事

1. **token 走不走 hf-mirror.com**
   - 默认路线是 hf-mirror.com。这是第三方镜像，**它的运营方能看到你的 token**。所以 token 要只读、只限这一个仓库，用完马上吊销。
   - 调查时匿名请求镜像返回 403（`not in the authorized list`），不是 401。所以镜像会不会转交你的 token，**未核实**，要等 token 放好以后，脚本发一个 HEAD 请求才知道：
     - 返回 2xx/3xx：继续下载；
     - 返回 401/403：脚本停下，打印备用路线。
   - 备用路线：huggingface.co，经公司默认代理（`ROUTE=hf`）。token 一直在 TLS 里，但很慢：10 月测过每个连接约 0.1 MB/s。换不换路线，会先问你。
   - 你也可以直接要求不走镜像，从一开始就用备用路线。
2. **下载能不能在登录节点上跑**
   - 脚本只开一个进程，主要占网络，大约 1 个核，加了 `nice -n 19`。
   - 顺序：先下验证和测试的 31 个场景（约 11 GB），再下训练的 68 个场景（约 24 GB）。
   - 每个场景都核对文件大小，并写进 sha256 清单。中途断了可以重跑，已完成的场景会跳过。
   - sco 作业里能不能上外网，**未核实**。所以默认在登录节点上跑。

## 你同意以后，我们要运行的命令（现在都没跑）

```bash
cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd
bash vpd/e3vs/download_splats.sh repo-from-copy   # 或 repo：从 GitHub 重新拉，固定在 bb7e603c
bash vpd/e3vs/download_splats.sh check            # 只发一个带 token 的 HEAD，不下载
bash vpd/e3vs/download_splats.sh splats valtest   # 31 个场景，约 11 GB
bash vpd/e3vs/download_splats.sh splats train     # 68 个场景，约 24 GB
bash vpd/e3vs/download_splats.sh verify all
```

- 不联网先看计划：`DRY_RUN=1 bash vpd/e3vs/download_splats.sh splats all`。这一步不读 token，也不写任何文件。
- **许可**：splat 只能用于非商业研究（ScanNet++ 条款）。渲染出来的图不能对外分发。E3VS 的代码是 MIT，标注是 CC BY 4.0。

# MiniMax-M3.1-Flash DP16EP16 on SGLang

## 模型简介

本配方使用 MiniMax-M3.1-Flash W4A8/W8A8、DSpark Smooth INT8 draft、FP8 KV 和原生 gfx938 FlashMLA。四节点启用 DeepEP 和 EPLB，编排为 TP16、Attention DP16、EP16。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | HCU fixed8 镜像 | BW1300 144 GB（gfx938） | 4 节点 × 4 卡 | DP16EP16 + EPLB | [fixed8 基线启动](#fixed8-基线) |
| 同上 | 同上 | v0.5.23 + DP16EP16 HCU Plugin | 同上 | 16 | DP16EP16 + EPLB | [社区版和插件启动](#社区版和插件) |

## 环境准备

- 四台独立计算节点，每台 4 张 BW1300 HCU，每张卡 144 GB 显存，共 16 卡。四台节点需要互通，并在容器内暴露 HCU、RDMA 设备和所需的驱动。
- 四节点的主模型、draft、EP16 EPLB 布局与调优配置须相同。将容器内的模型路径和配置路径设为实际挂载位置。每个节点根据可用网卡设置 `ROCSHMEM_ALLOWED_IBV_DEVICES`。
- EPLB 使用 EP16 的 `expert_distribution_recorder_1794400407.277366.pt`，hipBLASLt 使用 `hipblaslt.decode.accepted.config`。两份资产从对应的运行镜像或已验证的评测资产中取得。
- 使用宿主机间可达的 rank 0 地址。默认 rendezvous 端口为 5001，服务端口为 11011。

在四个节点分别设置环境变量。路径为容器内路径，网卡名和地址用实际值替换。

```bash
export MASTER_IP=192.0.2.10
export M31_MODEL_PATH=/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8
export M31_DSPARK_MODEL_PATH="$M31_MODEL_PATH/dspark-channel-int8-smooth-20261107"
export M31_EPLB_INIT_LOCATION=/assets/expert_distribution_recorder_1794400407.277366.pt
export M31_HIPBLASLT_CONFIG=/assets/hipblaslt.decode.accepted.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0
export DIST_PORT=5001 SERVER_PORT=11011
```

如果模型挂载下另有 `main-w4a8` 子目录，把 `M31_MODEL_PATH` 指向包含主模型 `config.json` 的目录，再将 `M31_DSPARK_MODEL_PATH` 设为实际的 draft 目录。

## 启动命令

在每个节点执行 `export NODE_RANK=0`，并将数值换成该节点的编号 0、1、2 或 3。先加 `--print-args` 核对完整命令，再同时启动四个进程。配置使用 FP8 E4M3 KV、128-token 页、上下文 163,840、Prefill chunk 8,192、静态显存比例 0.95 和 Decode full graph。KV 容量自动定容。插件启动脚本中，普通模式默认全局 160 条请求、graph local BS1–10。HiSparse 模式默认全局 256 条请求、graph local BS1–16。

### Fixed8 基线

使用 `minimax31-w4a8:dp16ep16-eplb-fp8-fixed8-20261116`。该镜像包含 HCU SGLang 基线 `e380d55a79645dfa0a478170e1045f0271cda8a3`、FlashMLA、LightOp、EP16 EPLB 布局与 hipBLASLt 调优配置。直接使用镜像内的资产时，可省略上面的 EPLB 和 hipBLASLt 路径变量。

```bash
bash scripts/serve-dp16ep16-fixed8.sh "$MASTER_IP" "$NODE_RANK" --print-args
bash scripts/serve-dp16ep16-fixed8.sh "$MASTER_IP" "$NODE_RANK"
```

启动日志应显示 `Native FP8 target verify split-KV = 8`，并确认原生 FP8 Stage 1 和 FP8 索引池生效。

### 社区版和插件

使用预装的兼容 HCU 软件栈，四节点保持同一套 Torch、Triton、FlashMLA、LightOp、DeepEP、AITER 和 ROCSHMEM。2026-11-22 四节点实测使用 DTK 26.04、Python 3.10.12、Torch 2.11.0 和 Triton 3.6.0。实测镜像内的社区 SGLang Python 元数据与 adapter wheel 元数据经过适配，沿用了镜像已有的 Torch 和 Triton。

| 组件 | 版本 |
| --- | --- |
| SGLang | `v0.5.23`（`297e54fd105e974cdf401e5b4c20253a22c16f90`） |
| 实测 HCU Plugin | `daf6a74420ba23232e3edc32f161d6b0609b0f1f` |
| 独立发布 HCU Plugin | [`fc47087679b061208d5e87542f78a32090018c6a`](https://github.com/chenyuege/sglang-plugin-das/commit/fc47087679b061208d5e87542f78a32090018c6a) |
| 实测 adapter-das | `d9bb7fa5d2ed7a46e74b06f5c7c2f9ef172bd702` |

独立插件分支包含实测的 EP 分布记录、HIP HiSparse shuffle、FP8 Verify host scalar 和 Target Verify Router 作用域。下表的 GPU 数据取自实测版本。独立分支通过专项回归，待在完整软件栈上重测。

如果镜像已安装社区 SGLang、adapter 与所需二进制依赖，可以安装上述独立插件。使用与镜像匹配的 Python 环境执行以下命令。

```bash
git clone --filter=blob:none --branch perf/minimax-m31-dp16ep16-1122 https://github.com/chenyuege/sglang-plugin-das.git sglang-plugin-dp16
git -C sglang-plugin-dp16 checkout --detach fc47087679b061208d5e87542f78a32090018c6a
HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation ./sglang-plugin-dp16
```

脚本优先使用运行目录里的 `bootstrap/sitecustomize.py`。安装在 `site` 并以 adapter 源码启动的环境也可沿用对应的目录结构。

### 插件版模式选择

四个节点使用相同模式与 HiSparse 配方，分别填写实际 `NODE_RANK`。每台节点四张 144 GB 卡。

```bash
# 真实 KV 的 Decode/RSC
bash scripts/serve-dp16ep16.sh rsc "$MASTER_IP" "$NODE_RANK" --print-args

# Fake Prefill Decode，关闭 HiSparse
bash scripts/serve-dp16ep16.sh fake "$MASTER_IP" "$NODE_RANK" --print-args

# Fake Prefill Decode，开启 HiSparse，默认 headwise
M31_HISPARSE_PROFILE=headwise bash scripts/serve-dp16ep16.sh hisparse "$MASTER_IP" "$NODE_RANK" --print-args
```

核对命令后去掉 `--print-args` 启动。headwise 使用 32,768 热槽、ratio 1.9、稳定槽位和 256×8 分块。整页对照用 `M31_HISPARSE_PROFILE=whole`，对应 65,536 热槽与 256×16 分块。两组均使用 Verify 直索引。当前 headwise 配方包含 Fake PD compact staging，仅用于 Fake Prefill Decode 复测。

DeepEP layered 默认关闭，适用于本次四节点同超节点 IPC 分组。可选的 `M31_TUNABLE_CSV=/assets/tunable.csv` 会从同目录读取设备序号对应的 `tunable0.csv` 到 `tunable3.csv`。实测使用对应 DTK/Torch/rocBLAS 的定点表，签名为 BF16 LM head 的 token M112、K6144、词表 200064。使用适配当前运行时的表，开启只读查表并禁用在线调优。

## API 调用

以下示例用于真实 KV 缓存模式。核对日志和 `/server_info` 的拓扑设置，确认 `/v1/loads` 返回 16 个 DP rank，再发送短请求。

```bash
export M31_URL="http://$MASTER_IP:$SERVER_PORT"
curl -fsS "$M31_URL/health"
curl -fsS "$M31_URL/server_info"
curl -fsS "$M31_URL/v1/loads?include=core"
curl -fsS "$M31_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"法国的首都是哪里？只回答城市名称。","sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

## 性能测试

历史四节点对照于 **2026 年 11 月 16 日**使用 fixed8 基线镜像和真实 KV 的 Decode/RSC 模式。每请求输入 147,456 tokens，其中 147,328 tokens 预热命中，输出 1,024 tokens。性能计时从预热完成后开始。客户端按 BS 同批发送，固定路由到 16 个 DP rank。自动定容为 **1,460,352 tokens/rank**。

| BS | 完成请求 | 整批输出吞吐 (tok/s) | 1 秒峰值 (tok/s) | 平均 TPOT (ms) | 最忙 rank 的 KV 峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128/128 | 1,486.90 | 5,655 | 29.55 | 81.32% |
| 144 | 144/144 | 2,048.00 | 4,952 | 32.51 | 91.44% |

- BS144 实际最高为每 rank 9 路。正式轮次中，原异常样本 #34 和 #98 均未复现交替循环，全批零接受和回撤均为 0。
- BS128 的三轮整批吞吐为 1,486.90、1,499.85 和 1,497.08 tok/s。请求 #99 的验证轮数升至 746–749 次，拖长整批完成时间。1 秒峰值与整批吞吐分别描述短时出 token 速率和完整负载完成速率。
### 插件版 Fake Prefill Decode

2026 年 11 月 22 日测试固定 **147,456 输入 tokens、1,024 输出 tokens**。关闭 HiSparse 的最新 BS144 为 **2,992.04 tok/s**，最大 rank KV 峰值 **91.3%**。关闭组的长输入测试到 BS144。

| 配方 | BS | 负载历史 | 整批输出吞吐中位 (tok/s) | P99 TPOT (ms) | GPU 热池峰值 |
| --- | ---: | --- | ---: | ---: | ---: |
| HiSparse 整页 | 256 | BS16 预热、首轮 BS256、再依次跑 BS64/144/224 | 2,135.08 | 100.22 | 83% |
| HiSparse headwise | 256 | 同序列 | 2,364.05 | 90.11 | 42% |
| HiSparse headwise 独立重启复验 | 256 | 同序列 | 2,336.45 | — | 42% |

正式四节点对照使用同一份 M112 定点 TunableOp 表。上表中位取两轮完整请求，整页和 headwise 每轮均观测到 256 条同时运行。主机池峰值约 99%。按 head 切换后，在同序列 Prof 的 16-rank 稳态中位里，KV 换页从 20.01 降至 4.16 ms/Verify。真实 KV 的输出一致性需单独验证。

[社区版启动脚本](../../../scripts/serve-dp16ep16.sh) · [Fixed8 基线启动脚本](../../../scripts/serve-dp16ep16-fixed8.sh)

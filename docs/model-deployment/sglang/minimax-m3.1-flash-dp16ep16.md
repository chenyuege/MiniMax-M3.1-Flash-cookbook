# MiniMax-M3.1-Flash DP16EP16 on SGLang

## 模型简介

本配方使用 MiniMax-M3.1-Flash W4A8/W8A8、DSpark Smooth INT8 draft、FP8 KV 和原生 gfx938 FlashMLA。四节点启用 DeepEP 和 EPLB，编排为 TP16、Attention DP16、EP16。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | HCU fixed8 镜像 | BW1300 144 GB（gfx938） | 4 节点 × 4 卡 | DP16EP16 + EPLB | [fixed8 基线启动](#fixed8-基线) |
| 同上 | 同上 | v0.5.23 + DP16EP16 HCU Plugin | 同上 | 16 | DP16EP16 + EPLB | [加载 1126 运行镜像](#加载-1126-运行镜像) |

## 环境准备

- 四台独立计算节点，每台 4 张 BW1300 HCU，每张卡 144 GB 显存，共 16 卡。四台节点需要互通，并在容器内暴露 HCU、RDMA 设备和所需的驱动。
- 1126 HiSparse 测试容器每节点内存限额为 728 GiB。启动前同时核对宿主机 `MemAvailable` 和容器 cgroup 余量，取两者较小值。本次压测驱动要求每节点至少有 640 GiB 可用，实测为约 688–692 GiB。该预算对应本页的 1.9 Host/Device ratio、4 卡/节点与长上下文负载。
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

### 加载 1126 运行镜像

四个节点使用同一份 `minimax31-w4a8-dp16ep16-eplb-plugin-sglang0523-hisparse-20261126.docker.tar`。镜像内有运行依赖、启动脚本、EP16 EPLB 布局和 hipBLASLt 调优文件。主模型和 DSpark 权重在启动时另外挂载。

| 项目 | 固定版本 |
| --- | --- |
| 镜像标签 | `minimax31-w4a8:dp16ep16-eplb-plugin-sglang0523-hisparse-20261126` |
| 镜像配置摘要 | `sha256:1578e6a4ceecf18c810a3a31fb20e70bab6156b64f4b2c329772b94201b37f57` |
| tar SHA-256 | `c5b5772fb4099d95e936488d353f5238cc55607903236e55c19dd8c265c1a2bb` |
| SGLang | `0.5.23`，`297e54fd105e974cdf401e5b4d20253a22c16f90` |
| HCU Plugin | [`b1ce50a29a45ec803b0354523bc887158a85722c`](https://github.com/chenyuege/sglang-plugin-das/commit/b1ce50a29a45ec803b0354523bc887158a85722c) |
| adapter-das | `d9bb7fa5d2ed7a46e74b06f5c7c2f9ef172bd702` |
| 运行环境 | DTK 26.04、Python 3.10.12、Torch 2.11、Triton 3.6 |

插件从已同步上游的 `proj/minimax-m3.1-cp8ep8` 重新构建，保留 [DP16 兼容修复](https://github.com/chenyuege/sglang-plugin-das/commit/ea07e624f70f97c51daf294942d2cd94efd4a7b7)与 Verify 写回元数据复用。镜像继承 1120 运行包的 Torch、Triton、FlashMLA、LightOp、DeepEP 和 adapter 二进制栈，用于复现 1126 的测试环境。

将归档和同名 `.sha256` 放在同一目录，从该目录加载。

```bash
archive=minimax31-w4a8-dp16ep16-eplb-plugin-sglang0523-hisparse-20261126.docker.tar
sha256sum -c "$archive.sha256"
docker load -i "$archive"
docker image inspect --format '{{.Id}}' \
  minimax31-w4a8:dp16ep16-eplb-plugin-sglang0523-hisparse-20261126
```

按[环境准备](#环境准备)启动容器并透传四张 HCU、RDMA 和模型挂载。容器内可以直接使用 `/opt/minimax31-w4a8-plugin/serve.sh`，它与 [DP16EP16 启动脚本](../../../scripts/serve-dp16ep16.sh)相同。使用镜像内的 EPLB 布局和调优表时，可省略 `M31_EPLB_INIT_LOCATION` 与 `M31_HIPBLASLT_CONFIG`。通过 `/opt/minimax31-w4a8-plugin/install-verified` 查看构建所用的源码版本。

在每个计算节点设置 `CHECKPOINTS_DIR`、`MASTER_IP`、`NODE_RANK` 和该节点的 `ROCSHMEM_ALLOWED_IBV_DEVICES`，并设置前面确认过的容器内权重路径。以下命令启动用于 1126 复现的 Fake Prefill + headwise HiSparse。关闭组将 `hisparse` 改为 `fake`。测 BS128/144 同配置 A/B 前，在宿主机 `export M31_MAX_RUNNING_REQUESTS=160 M31_GRAPH_MAX_LOCAL_BS=10`，开关两侧均使用同值。跑完整扫档时可取消这两个覆盖，按模式采用脚本默认值。

```bash
M31_IMAGE=minimax31-w4a8:dp16ep16-eplb-plugin-sglang0523-hisparse-20261126
docker run -d --name "minimax-m31-dp16-$NODE_RANK" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v /module:/module:ro \
  -v "$CHECKPOINTS_DIR:/model:ro" \
  -e M31_MODEL_PATH -e M31_DSPARK_MODEL_PATH -e ROCSHMEM_ALLOWED_IBV_DEVICES \
  -e DIST_PORT -e SERVER_PORT -e M31_HISPARSE_PROFILE=headwise \
  -e M31_MAX_RUNNING_REQUESTS -e M31_GRAPH_MAX_LOCAL_BS \
  -e SGLANG_SIMULATE_ACC_LEN=3.6 -e SGLANG_MINIMAX_EXTEND_ROUTER_GEMV=0 \
  --entrypoint /bin/bash "$M31_IMAGE" \
  /opt/minimax31-w4a8-plugin/serve.sh hisparse "$MASTER_IP" "$NODE_RANK"
```

归档通过了 SHA-256 和独立 containerd 导入验证，四种启动模式通过四个节点编号的参数检查。性能数据来自 1126 原四节点实测，本次重新打包后未另起 HCU 服务重测。

### 插件版模式选择

从镜像启动时，以镜像内的 `serve.sh` 替换示例中的仓库脚本路径。四个节点使用相同模式与 HiSparse 配方，分别填写实际 `NODE_RANK`。每台节点四张 144 GB 卡。

```bash
# 真实 KV 的 Decode/RSC
SGLANG_SIMULATE_ACC_LEN=0 bash scripts/serve-dp16ep16.sh rsc "$MASTER_IP" "$NODE_RANK" --print-args

# Fake Prefill Decode，关闭 HiSparse
bash scripts/serve-dp16ep16.sh fake "$MASTER_IP" "$NODE_RANK" --print-args

# Fake Prefill Decode，开启 HiSparse，默认 headwise
M31_HISPARSE_PROFILE=headwise bash scripts/serve-dp16ep16.sh hisparse "$MASTER_IP" "$NODE_RANK" --print-args
```

核对命令后去掉 `--print-args` 启动。headwise 使用 32,768 热槽、ratio 1.9、稳定槽位和 256×8 分块。整页对照用 `M31_HISPARSE_PROFILE=whole`，对应 65,536 热槽与 256×16 分块。两组均使用 Verify 直索引。当前 headwise 配方包含 Fake PD compact staging，仅用于 Fake Prefill Decode 复测。

DeepEP layered 默认关闭，适用于本次四节点同超节点 IPC 分组。可选的 `M31_TUNABLE_CSV=/assets/tunable.csv` 会从同目录读取设备序号对应的 `tunable0.csv` 到 `tunable3.csv`。1122 历史对照使用对应 DTK/Torch/rocBLAS 的定点表，签名为 BF16 LM head 的 token M112、K6144、词表 200064。使用适配当前运行时的表，开启只读查表并禁用在线调优。

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

### 复跑 Fake Prefill Decode

[压测脚本](../../../scripts/bench-dp16ep16-fake.py)调用 SGLang v0.5.23 的原生 serving benchmark。每条请求使用 147,456 个输入 token，生成 1,024 个输出 token。同一批请求按序号轮询分配到最多 16 个 DP rank，默认无请求率限制。对比时保持模型、tokenizer、EPLB 布局、调优表和客户端环境一致，并让服务独占压测流量。

#### 1. 启动四节点服务

按[环境准备](#环境准备)在四个节点设置路径、网卡、`MASTER_IP` 和各自的 `NODE_RANK`。以下命令每次选一组，在四节点同时执行。换配方或复跑历史序列前，重启四个服务进程。

```bash
# 复现 1126 测量口径。真实 RSC 保持模拟接受长度默认值 0。
export SGLANG_SIMULATE_ACC_LEN=3.6
export SGLANG_MINIMAX_EXTEND_ROUTER_GEMV=0
unset M31_TUNABLE_CSV

# Fake Prefill Decode，关闭 HiSparse
bash scripts/serve-dp16ep16.sh fake "$MASTER_IP" "$NODE_RANK"

# Fake Prefill Decode，开启 HiSparse，按 head 换页
M31_HISPARSE_PROFILE=headwise bash scripts/serve-dp16ep16.sh hisparse "$MASTER_IP" "$NODE_RANK"

# Fake Prefill Decode，开启 HiSparse，整页换页
M31_HISPARSE_PROFILE=whole bash scripts/serve-dp16ep16.sh hisparse "$MASTER_IP" "$NODE_RANK"
```

在能访问 rank 0 的客户端执行下面的健康检查。确认 `/server_info` 显示 Decode、fake、EPLB 和预期的 HiSparse 设置，`/v1/loads` 返回 16 个空闲 DP rank。

```bash
export SERVER_PORT=11011
export M31_URL="http://$MASTER_IP:$SERVER_PORT"
curl -fsS "$M31_URL/health"
curl -fsS "$M31_URL/server_info"
curl -fsS "$M31_URL/v1/loads?include=core"
```

#### 2. 准备和校验输入

在客户端进入 cookbook 根目录，使用能导入 SGLang v0.5.23 benchmark 模块的 Python 环境。客户端需要访问本地模型的 tokenizer 文件。可以复用已有的冻结输入，也可以用本地 ShareGPT JSON 生成。生成时使用 seed 42，取 64 条 token 序列，按原生 random sampler 取固定长度。已有输入文件时直接执行校验。

```bash
export M31_MODEL_PATH=/path/to/model
export M31_TOKENIZER_PATH="$M31_MODEL_PATH"
export INPUTS="$PWD/inputs/random_sharegpt64_seed42_147456_ids.json.gz"

# 使用已有冻结文件时跳过生成
python3 scripts/bench-dp16ep16-fake.py --inputs "$INPUTS" \
  --prepare-from-sharegpt /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer "$M31_TOKENIZER_PATH"

python3 scripts/bench-dp16ep16-fake.py --inputs "$INPUTS" --check-inputs
```

精确复现本页实测时，在校验和压测命令中加 `--require-reference-inputs`，校验值应为 `28a24124753116c9cc88ea8bb4accb0968d1786cb346474c76cdb412a45c0dfe`。不同 ShareGPT 副本或 tokenizer 可能生成不同摘要。此时可以使用新输入比较各配置，比较双方需使用同一个冻结文件。

#### 3. 运行并发扫描

`--bs` 指一轮提交的全局请求数，`--repeats` 指重复轮数，脚本会按提供的 BS 顺序跑完一轮再重复。用 `--require-residency` 按每 rank 的请求上限、128-token 页、draft 预留和 KV 容量检查整批能否驻留。每次执行使用一个新的 `--output` 目录。

```bash
# 在关闭 HiSparse 的服务上跑到 BS144
python3 scripts/bench-dp16ep16-fake.py --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --bs 16 32 64 96 128 144 --require-residency \
  --output results/fake-off-sweep

# 在开启 headwise HiSparse 的服务上跑到 BS256
python3 scripts/bench-dp16ep16-fake.py --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --hisparse --bs 16 32 64 96 128 144 160 176 192 208 224 240 256 \
  --require-residency --output results/fake-headwise-sweep
```

上述 BS 上限对应本页配方。若脚本报告某个 rank 容量不足，检查该轮的 `capacity-budget.json` 和 `server_info.json`，再根据实测容量选择 BS。单独诊断长尾时可设置 `--request-rate`（请求/秒），并给限速组使用独立的输出目录。原实测对照使用默认的 `inf`。 复测 1126 的 BS128/144 同配置对照时，开启侧也设 `M31_MAX_RUNNING_REQUESTS=160` 和 `M31_GRAPH_MAX_LOCAL_BS=10`。完整开启扫描仍使用 256 和 16。

#### 4. 复现 BS256 的负载历史

四节点先启动 `whole` 并执行第一条命令。重启四节点为 `headwise` 后执行第二条命令。两组均依次执行 BS16 预热、首轮 BS256、BS64/144/224，再跑两轮 BS256。对比每组最后两轮的吞吐中位数与延迟，每组的 `fresh` 子目录对应首轮 BS256。

```bash
python3 scripts/bench-dp16ep16-fake.py --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --hisparse --history --output results/fake-whole-history

python3 scripts/bench-dp16ep16-fake.py --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --hisparse --history --output results/fake-headwise-history
```

#### 5. 读取结果

- `summary.csv` 汇总 BS、完成量、整批输出吞吐、延迟、推测接受长度、采样到的并发和 KV 峰值。对比吞吐使用 `output_tps`，单位为 tok/s。
- 每轮目录的 `measurement.json` 记录实际原生命令参数与输入摘要。`requests.json` 记录逐请求的目标 rank、TTFT、TPOT、接受数据和输出摘要。`native.jsonl` 是原生 benchmark 详细记录，包含生成文本。
- `loads.jsonl` 每约 2 秒采样一次。`sampled-peaks.json` 汇总计时窗口内的并发及各 DP rank 的 `num_used_tokens / max_total_num_tokens`。采样值可能错过瞬时峰值。HiSparse 的 GPU 热池和 CPU 主机池峰值分别从同一窗口内的服务日志 `#gpu token usage` 和 `#cpu token usage` 读取。
- Fake Prefill 用于对比 Decode 性能。真实 KV 输出一致性使用真实 Prefill 的评测。

### 1126 插件版 Fake Prefill Decode

四节点 16 张 BW1300，共用 FP8 KV、128-token 页、DSpark static Verify8、模拟接受长度 3.6。每请求 147,456 输入、1,024 输出。BS128/144 的基线关闭与整页开启均使用全局 max-running 160、每 rank Graph 1–10。候选使用相同的并发与 Graph 上限。下表为无 Prof 三轮输出吞吐的中位数。

| BS | 关闭基线 (tok/s) | 整页 HiSparse (tok/s) | 按 head 候选 (tok/s) | 候选相对整页 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 4,148.90 | 2,881.31 | 3,613.76 | +25.42% |
| 144 | 3,826.69 | 2,743.72 | 3,472.68 | +26.57% |

- 整页模式的 BS144 Verify 内每周期新增约 6.59 ms Host→GPU 换页、1.26 ms 页规划与 0.43 ms 索引展开。按 head 候选将换页降至 2.30 ms，GPU 周期从 131.66 ms 降至 124.33 ms。候选同时启用了 32,768-token 热缓冲、HIP 稳定槽及 Verify 写回复用，增幅对应组合配置。
- BS144 长 Trace 捕获到一次单 rank INT8 GEMM 延长至 92.43 ms，其余 15 个 rank 的 DeepEP dispatch 同期约 92 ms。该样本展示了慢步在 EP 组内的传播。全部秒级低吞吐窗口的触发点尚未定位。
- 最新扫档关闭侧覆盖 BS16–144，开启侧覆盖 BS16–256。开启的 BS256 两轮分别达到 4,300.07 和 4,226.19 tok/s，GPU 热池约 42%，Host KV 峰值 99%，采样中无排队与回撤。按 head 结果适用于 Fake Prefill Decode。

### 1122 插件版 Fake Prefill Decode 历史对照

2026 年 11 月 22 日使用的插件为 `daf6a74420ba23232e3edc32f161d6b0609b0f1f`。历史配方另有[独立插件提交](https://github.com/chenyuege/sglang-plugin-das/commit/fc47087679b061208d5e87542f78a32090018c6a)。该次测试固定 **147,456 输入 tokens、1,024 输出 tokens**。该次关闭 HiSparse 的 BS144 为 **2,992.04 tok/s**，最大 rank KV 峰值 **91.3%**。关闭组的长输入测试到 BS144。

| 配方 | BS | 负载历史 | 整批输出吞吐中位 (tok/s) | P99 TPOT (ms) | GPU 热池峰值 |
| --- | ---: | --- | ---: | ---: | ---: |
| HiSparse 整页 | 256 | BS16 预热、首轮 BS256、再依次跑 BS64/144/224 | 2,135.08 | 100.22 | 83% |
| HiSparse headwise | 256 | 同序列 | 2,364.05 | 90.11 | 42% |
| HiSparse headwise 独立重启复验 | 256 | 同序列 | 2,336.45 | — | 42% |

正式四节点对照使用同一份 M112 定点 TunableOp 表。上表中位取两轮完整请求，整页和 headwise 每轮均观测到 256 条同时运行。主机池峰值约 99%。按 head 切换后，在同序列 Prof 的 16-rank 稳态中位里，KV 换页从 20.01 降至 4.16 ms/Verify。真实 KV 的输出一致性需单独验证。

[社区版启动脚本](../../../scripts/serve-dp16ep16.sh) · [Fixed8 基线启动脚本](../../../scripts/serve-dp16ep16-fixed8.sh)

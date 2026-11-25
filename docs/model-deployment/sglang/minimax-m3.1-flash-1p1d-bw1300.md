# MiniMax M3.1 Flash 1P1D on BW1300 144 GB

## 支持配置

运行一个 Prefill worker、一个 Decode worker 和一个 Mooncake PD Router。两个 worker 各使用 8 张 BW1300 144 GB HCU。单机可见 8 卡时，每个 worker 占一台主机。单机可见 4 卡时，每个 worker 跨两台主机，共四台主机。分布式 worker 的两个节点必须处于同一可用 RDMA 域。

| Worker | 并行和注意力 | 144 GB 默认配置 |
| --- | --- | --- |
| Prefill | TP8、CP8 zigzag、EP8、FA3、normal DeepEP | 显存比例 0.90，chunk 16,384，最多 80 个运行请求 |
| Decode | TP8、attention DP8、EP8、FA3 Decode、DeepEP auto | 显存比例 0.95，最多 64 个运行请求，full graph 捕获局部 BS 1–8，HiSparse 默认关闭 |

两端使用相同的 W4A8/W8A8 主模型与 Smooth INT8 DSpark draft。Mooncake 传输使用 FP8 E4M3 KV、page size 128 和同一套 target/draft KV 布局。DSpark 块大小为 7，draft token 数为 8。首轮真实 PD 验收使用已部署 wheel 的 Verify 配方，模拟接受长度为 0，DeepEP layered Decode 关闭。

## 环境准备

使用同一 DTK/Torch、SGLang 0.5.23、Plugin、adapter、Mooncake、DeepEP、FlashMLA 和 LightOp 组合。沿用通用安装与镜像构建步骤。两端安装主权重、draft 与 tokenizer，检查各自的可见设备数量和 RDMA 设备。Docker 方式需用包含新版 `scripts/serve.sh` 的仓库重新构建镜像，并向所有 worker 分发同一镜像。 [安装与模型准备](minimax-m3.1-flash.md#环境准备) · [镜像构建](minimax-m3.1-flash.md#容器构建与服务管理)

用 `ibv_devices` 和 `ibdev2netdev` 为每台主机选择实际的 Mooncake 和 DeepEP 网卡，分别设置 `M31_PD_IB_DEVICES` 和 `ROCSHMEM_ALLOWED_IBV_DEVICES`。下例的 `mlx5_0` 需按本机替换。调优文件和 EPLB 映射可从匹配角色、模型及并行拓扑的部署挂载中提供。`M31_PREFILL_CHUNK` 可覆盖 Prefill 的 chunk 和单轮 token 上限，默认保留 16,384 作为真实 PD 的起始配置。

## 启动

### 每台主机 8 卡

在各节点设置自己的 RDMA 设备和模型路径。P 和 D 分别运行对应角色。

```bash
export M31_HARDWARE_PROFILE=bw1300-144gb
export M31_PD_IB_DEVICES=mlx5_0
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0

# P 节点
bash scripts/serve.sh prefill --print-args
bash scripts/serve.sh prefill

# D 节点，在另一台主机运行
bash scripts/serve.sh decode --print-args
bash scripts/serve.sh decode
```

### 每台主机 4 卡

P 和 D 各自启动 rank 0、rank 1。每个 worker 的两个 rank 使用同一个 rendezvous 地址。P、D 使用各自 rank 0 的地址。下例需要将 `P_RANK0_IP` 和 `D_RANK0_IP` 设置为实际可达的 IP。

```bash
# 在四台主机分别设置上一节的硬件档、模型与网卡变量。
export M31_NNODES=2

# 仅 P rank 0
M31_NODE_RANK=0 M31_DIST_INIT_ADDR="$P_RANK0_IP:5001" bash scripts/serve.sh prefill
# 仅 P rank 1
M31_NODE_RANK=1 M31_DIST_INIT_ADDR="$P_RANK0_IP:5001" bash scripts/serve.sh prefill
# 仅 D rank 0
M31_NODE_RANK=0 M31_DIST_INIT_ADDR="$D_RANK0_IP:5001" bash scripts/serve.sh decode
# 仅 D rank 1
M31_NODE_RANK=1 M31_DIST_INIT_ADDR="$D_RANK0_IP:5001" bash scripts/serve.sh decode
```

Compose 使用相同的硬件档、节点数、rank 和 rendezvous 变量。在每台主机设置自己的变量后执行 `M31_ROLE=prefill docker compose up -d worker` 或 `M31_ROLE=decode docker compose up -d worker`。单主机 worker 使用默认 rank 0。脚本默认在单主机模式选本地卡 0–7，在双主机模式选本地卡 0–3。

### 启动 Router

Router 只注册两个 worker 的 rank 0 HTTP 地址。确认两个 worker 的 `/server_info` 已可访问，再启动 Router。这里使用默认 worker 端口 30000、Prefill bootstrap 端口 8998 和 Router 端口 30001。

```bash
M31_PREFILL_URL="http://$P_RANK0_IP:30000" \
M31_DECODE_URL="http://$D_RANK0_IP:30000" \
docker compose up -d router
```

## 验收

```bash
curl -fsS "http://$P_RANK0_IP:30000/server_info"
curl -fsS "http://$D_RANK0_IP:30000/server_info"
curl -fsS "http://$ROUTER_IP:30001/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8","messages":[{"role":"user","content":"用一句话解释什么是上下文并行。"}],"max_tokens":16,"temperature":0}'
```

检查短请求的输出和 P/D 两侧 Mooncake 日志。用固定长输入先跑 BS1、8、16，再逐档增加至 32 和 64。记录两端实际 KV 池、每 rank 运行峰值、完成率、TTFT 和 TPOT。Decode 的 163,840 token 上下文上限约束 Router 请求的输入与输出总长度。 [长请求压测步骤](minimax-m3.1-flash.md) · [压测脚本](../../../scripts/bench-pd.py)

144 GB DP8EP8 的参考 Fake Prefill Decode 在 147,456 输入、1,024 输出下观测到每 rank 约 128.8 万设备 KV tokens 和局部 BS8。1P1D 配方已检查脚本语法、P/D 两端启动参数、单主机和双主机环境传递。真实 Mooncake 长请求的数值和吞吐需要在目标机器复验。

# MiniMax-M3.1-Flash DP16EP16 on SGLang

## 模型简介

本配方使用 MiniMax-M3.1-Flash W4A8/W8A8、DSpark Smooth INT8 draft、FP8 KV 和原生 gfx938 FlashMLA。四节点启用 DeepEP 和 EPLB，编排为 TP16、Attention DP16、EP16。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | HCU fixed8 镜像 | BW1300 144 GB（gfx938） | 4 节点 × 4 卡 | DP16EP16 + EPLB | [fixed8 基线启动](#fixed8-基线) |
| 同上 | 同上 | v0.5.23 + HCU Plugin + fixed8 补丁 | 同上 | 16 | DP16EP16 + EPLB | [社区版和插件启动](#社区版和插件) |

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

在每个节点执行 `export NODE_RANK=0`，并将数值换成该节点的编号 0、1、2 或 3。先加 `--print-args` 核对完整命令，再同时启动四个进程。配置使用 FP8 E4M3 KV、128-token 页、上下文 163,840、Prefill chunk 8,192、全局最大运行请求 160、静态显存比例 0.95 和 Decode full graph。KV 容量自动定容。

### Fixed8 基线

使用 `minimax31-w4a8:dp16ep16-eplb-fp8-fixed8-20261116`。该镜像包含 HCU SGLang 基线 `e380d55a79645dfa0a478170e1045f0271cda8a3`、FlashMLA、LightOp、EP16 EPLB 布局与 hipBLASLt 调优配置。直接使用镜像内的资产时，可省略上面的 EPLB 和 hipBLASLt 路径变量。

```bash
bash scripts/serve-dp16ep16-fixed8.sh "$MASTER_IP" "$NODE_RANK" --print-args
bash scripts/serve-dp16ep16-fixed8.sh "$MASTER_IP" "$NODE_RANK"
```

启动日志应显示 `Native FP8 target verify split-KV = 8`，并确认原生 FP8 Stage 1 和 FP8 索引池生效。

### 社区版和插件

使用 Python ≥3.11，优先沿用既有 DTK 26.10 / Python 3.12 的 HCU 环境。四节点使用同一套兼容的 Torch、FlashMLA MSA128、LightOp、DeepEP、AITER 和 ROCSHMEM。通过项目内的补丁将 split=8 修复应用到 HCU Plugin，社区源码可直接使用发布版本。

| 组件 | 版本 |
| --- | --- |
| SGLang | `v0.5.23`（`297e54fd105e974cdf401e5b4d20253a22c16f90`） |
| HCU Plugin | `91dab4b87695f195e0aa487762233607c9806bdb` + [FP8 verify 补丁](../../../patches/minimax-m31-fp8-verify-fixed8.patch) |
| adapter-das | `1c9448e1b09514a8d1424ad0d50d70cb88a6d078` |

四节点各准备一次运行目录。执行下列命令前，当前目录应为本 cookbook 的根目录。

```bash
export M31_RUNTIME_DIR=/opt/minimax31-w4a8-plugin
mkdir -p "$M31_RUNTIME_DIR"/{source,site}
git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git "$M31_RUNTIME_DIR/source/sglang"
git clone --filter=blob:none --branch main https://github.com/HYGON-AI/adapter-das.git "$M31_RUNTIME_DIR/source/adapter-das"
git -C "$M31_RUNTIME_DIR/source/adapter-das" checkout --detach 1c9448e1b09514a8d1424ad0d50d70cb88a6d078
git clone --filter=blob:none --branch proj/minimax-m3.1 https://github.com/HYGON-AI/sglang-plugin-das.git "$M31_RUNTIME_DIR/source/sglang-plugin-das"
git -C "$M31_RUNTIME_DIR/source/sglang-plugin-das" checkout --detach 91dab4b87695f195e0aa487762233607c9806bdb
git -C "$M31_RUNTIME_DIR/source/sglang-plugin-das" apply "$PWD/patches/minimax-m31-fp8-verify-fixed8.patch"
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23 \
  python3 -m pip install --no-deps --no-build-isolation --target "$M31_RUNTIME_DIR/site" "$M31_RUNTIME_DIR/source/sglang/python"
HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation \
  --target "$M31_RUNTIME_DIR/site" "$M31_RUNTIME_DIR/source/sglang-plugin-das"
```

使用 `rsc` 启动真实 KV 缓存模式。切换时在四节点使用相同模式。`fake` 使用 Fake Prefill，`hisparse` 同时启用 HiSparse。该脚本已完成参数检查，插件版四节点 GPU 和性能验证待完成。

```bash
bash scripts/serve-dp16ep16.sh rsc "$MASTER_IP" "$NODE_RANK" --print-args
bash scripts/serve-dp16ep16.sh rsc "$MASTER_IP" "$NODE_RANK"
```

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

最新已完成的四节点正式测试于 **2026 年 11 月 16 日**使用 fixed8 基线镜像和真实 KV 的 Decode/RSC 模式。每请求输入 147,456 tokens，其中 147,328 tokens 预热命中，输出 1,024 tokens。性能计时从预热完成后开始。客户端按 BS 同批发送，固定路由到 16 个 DP rank。自动定容为 **1,460,352 tokens/rank**。

| BS | 完成请求 | 整批输出吞吐 (tok/s) | 1 秒峰值 (tok/s) | 平均 TPOT (ms) | 最忙 rank 的 KV 峰值 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 128/128 | 1,486.90 | 5,655 | 29.55 | 81.32% |
| 144 | 144/144 | 2,048.00 | 4,952 | 32.51 | 91.44% |

- BS144 实际最高为每 rank 9 路。正式轮次中，原异常样本 #34 和 #98 均未复现交替循环，全批零接受和回撤均为 0。
- BS128 的三轮整批吞吐为 1,486.90、1,499.85 和 1,497.08 tok/s。请求 #99 的验证轮数升至 746–749 次，拖长整批完成时间。1 秒峰值与整批吞吐分别描述短时出 token 速率和完整负载完成速率。
- 社区版插件的真实 KV、Fake Prefill 和 HiSparse 四节点性能待实测。

[社区版启动脚本](../../../scripts/serve-dp16ep16.sh) · [Fixed8 基线启动脚本](../../../scripts/serve-dp16ep16-fixed8.sh)

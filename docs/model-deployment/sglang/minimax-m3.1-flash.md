# MiniMax-M3.1-Flash on SGLang

## 模型简介

MiniMax-M3.1-Flash 采用 MoE 和 MiniMax Sparse Attention（MSA）。本配方使用海光的 per-channel 混合量化权重，MoE 专家走 INT4 W4A8，配合 INT8 W8A8 路径和 FP8 E4M3 KV cache。DSpark 采用模型仓随附的 Smooth INT8 draft。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1100 96 GB | 16 | 1P1D | [**`>_`**](#minimax-m31-flash-channel-w4a8-int4-w8a8-int8-1p1d-bw1100-16x-sglang-0523) |

## 环境准备

- 两台 Linux 节点，各有 8 张 BW1100 96 GB 卡（`gfx938`）。在容器内确认 8 张卡、`/dev/kfd`、`/dev/dri`、RDMA 设备及网卡可见，P/D 节点之间可互访 worker HTTP 端口和 Prefill bootstrap 端口。
- 两端使用 Python 3.12、DTK 26.10 和同一套 HCU Torch 运行时，准备兼容的 AITER、DeepEP、ROCSHMEM、Mooncake、FlashMLA（gfx938 MSA128）和 LightOp（W4A8 HIPC、INT8 量化）。使用现成的匹配环境和 wheel。
- 复跑时固定 [SGLang 0.5.23 宿主提交](https://github.com/sgl-project/sglang/tree/297e54fd105e974cdf401e5b4d20253a22c16f90)、[HCU Plugin 的 HiSparse/DSpark 修复](https://github.com/chenyuege/sglang-plugin-das/commit/838ad34bfcb503a7f10f8cb60ab375505b079303)和[配套 adapter 版本](https://github.com/chenyuege/adapter-das/commit/20d18cc)。插件修复提交与 96 GB 验证用插件的 Git 树一致，包含 [HiSparse PR #70](https://github.com/HYGON-AI/sglang-plugin-das/pull/70)。

直接在现有 HCU 容器内启动时，在两端各执行一次。已有兼容的 SGLang 依赖时可用 `--no-deps` 安装宿主代码和插件，保留镜像中的 Torch 与原生扩展。使用本仓库 Dockerfile 时，这些安装步骤已包含在构建过程中。

```bash
mkdir -p runtime/src
git clone https://github.com/sgl-project/sglang.git runtime/src/sglang
git -C runtime/src/sglang checkout --detach 297e54fd105e974cdf401e5b4d20253a22c16f90
git clone https://github.com/chenyuege/adapter-das.git runtime/src/adapter-das
git -C runtime/src/adapter-das fetch origin 20d18cc6c550e32b119fe69a95db2c6ed83d7887
git -C runtime/src/adapter-das checkout --detach FETCH_HEAD
git clone --branch fix/minimax-m31-hisparse-dspark-tp-scope \
  https://github.com/chenyuege/sglang-plugin-das.git runtime/src/sglang-plugin-das
git -C runtime/src/sglang-plugin-das checkout --detach 838ad34bfcb503a7f10f8cb60ab375505b079303

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.23 \
  python3 -m pip install --no-deps --no-build-isolation -e runtime/src/sglang/python
HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation \
  --target "$PWD/runtime/plugin-site" runtime/src/sglang-plugin-das
export PYTHONPATH="$PWD/runtime/src/adapter-das:$PWD/runtime/plugin-site${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import torch, sglang, flash_mla, lightop, mooncake.engine; print(torch.__version__, torch.version.hip, torch.cuda.device_count(), sglang.__file__)'
```

通过 ModelScope 下载完整快照并在两端挂载。模型仓约 229 GB，预留额外空间给缓存和日志。若使用 Docker，下面的模型路径和后续配置路径均以容器内视角为准。

```bash
export M31_MODEL_PATH=/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8
modelscope download --model hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 \
  --local_dir "$M31_MODEL_PATH"
test -r "$M31_MODEL_PATH/config.json"
test -r "$M31_MODEL_PATH/dspark-channel-int8-smooth-20261107/config.json"
```

### 容器构建与服务管理

使用包含上述 HCU 原生依赖的 Linux x86_64 基础镜像。该镜像还需提供 Bash、Git、Python 3.12、pip 和 SGLang 所需的运行及构建依赖。构建使用镜像中的 Torch、DTK、FlashMLA、LightOp、DeepEP 和 Mooncake。把 `HCU_BASE_IMAGE` 设置为可访问的内部镜像地址或本地镜像 ID。

```bash
docker build \
  --build-arg HCU_BASE_IMAGE="$HCU_BASE_IMAGE" \
  --tag minimax-m31-hcu:0.5.23 .
```

将同一镜像分发到两台 worker，Router 节点也使用该镜像。Compose 使用宿主网络和 IPC，挂载模型（只读）、可选调优资产（只读）与 trace 输出目录。若基础镜像要求额外映射宿主的驱动或 HyHAL 目录，按该镜像的运行说明增加 Compose volume。

```bash
docker compose run --rm --no-deps --entrypoint python3 worker -c \
  'import torch, sglang, flash_mla, lightop, mooncake.engine; print(torch.__version__, torch.version.hip, torch.cuda.device_count()); assert torch.cuda.device_count() == 8'
```

在 P 节点执行，按本机 RDMA 枚举替换 `mlx5_0`。在 D 节点用相同命令，将 `M31_ROLE=prefill` 换成 `M31_ROLE=decode`，并填入 D 节点的网卡。

```bash
M31_ROLE=prefill \
  M31_MODEL_ROOT=/path/to/model-parent \
  M31_PD_IB_DEVICES=mlx5_0 \
  ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0 \
  docker compose up -d worker
docker compose logs -f worker
```

`M31_MODEL_ROOT` 指模型目录的父目录。若模型目录名不同，可用 `M31_MODEL_PATH` 设置容器内的绝对路径。使用可选调优文件时，在宿主的 `M31_ASSET_ROOT` 下放置资产，在启动变量中使用 `/assets/...` 路径。

两端 `/server_info` 可用后，在 Router 节点启动 Router，替换 P/D 地址。

```bash
M31_PREFILL_URL=http://192.0.2.10:30000 \
  M31_DECODE_URL=http://192.0.2.11:30000 \
  docker compose up -d router
docker compose logs -f router
```

需要变更环境参数时，带上新参数再次执行对应的 `docker compose up -d worker` 或 `docker compose up -d router`。停止时先在 Router 节点执行 `docker compose stop router`，再在 P/D 节点各执行 `docker compose stop worker`。直接在已有容器内运行时，使用下一节的前台命令。

## 启动命令

### MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8 1P1D BW1100 16x SGLang 0.5.23

在各节点用 `ibv_devices` 和 `ibdev2netdev` 核对实际设备名。分别为 Mooncake 和 DeepEP 设置设备列表，并在对应节点的启动终端保留环境变量。

```bash
export M31_PD_IB_DEVICES=mlx5_0
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0
```

#### P node 0

TP8 / CP8 zigzag / EP8，FA3 eager，DeepEP normal。Prefill chunk 和单轮最大 Prefill token 数均为 16,384，静态显存比例为 0.87。

```bash
bash scripts/serve.sh prefill --print-args
bash scripts/serve.sh prefill
```

复跑原配置时，在 Prefill 上设置 `M31_DEEPEP_CONFIG`（对应 `deepep-sms64-only.json`）、`M31_EPLB_LAYOUT`（对应 CP8/EP8、BS4 的固定布局）和 `M31_PREFILL_MAX_TOTAL_TOKENS=1192960`。常规启动由 DeepEP 使用默认配置，EPLB 按需启用，KV token 数由 SGLang 自动定容。原配方还按 P、D 分别加载了调优后的 `HIPBLASLT_TUNING_OVERRIDE_FILE` 和 `ROCBLAS_TENSILE_LIBPATH`，复跑时应使用与 DTK 和卡型匹配的资产。

#### D node 0

TP8 / attention DP8 / EP8，Triton Prefill + FA3 Decode，DSpark 和 HiSparse 已启用。Decode full graph 捕获本地 BS 1、2、3，全局请求上限为 24，静态显存比例为 0.92，设备 KV 容量自动定容。HiSparse 的 host/device 比为 1。

```bash
bash scripts/serve.sh decode --print-args
bash scripts/serve.sh decode
```

P/D 两端共享 FP8 E4M3 KV 格式、128-token 页及同一 DSpark draft 布局。Decode 使用 `M31_DECODE_GRAPH=disabled` 可切换到 eager 进行对照。复跑原诊断流程时，可以在两端设置 `M31_SKIP_STARTUP_WARMUP=1`，然后由客户端执行预热。

#### Router

在可访问两台 worker 的节点上使用 SGLang Router 0.3.2。将 `P_HOST`、`D_HOST` 换成实际地址，默认 worker 端口为 30000，Prefill bootstrap 为 8998。

```bash
python3 -m pip install 'sglang-router==0.3.2'
export P_HOST=192.0.2.10
export D_HOST=192.0.2.11
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://$P_HOST:30000" 8998 \
  --decode "http://$D_HOST:30000" \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30001
```

## API 调用

### PD 分离

```bash
export ROUTER_URL=http://127.0.0.1:30001
curl -fsS "http://$P_HOST:30000/server_info"
curl -fsS "http://$D_HOST:30000/server_info"
curl -fsS "$ROUTER_URL/v1/models"
curl -fsS "$ROUTER_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8",
    "messages":[{"role":"user","content":"法国的首都是哪里？只回答城市名称。"}],
    "chat_template_kwargs":{"enable_thinking":false},
    "temperature":0,
    "max_tokens":32
  }'
```

检查两端的 `disaggregation_mode`、Mooncake 后端、Decode 的 HiSparse 开关与请求上限，再检查 Router 返回的 `choices[0].message.content`。如果短请求停在首轮 DSpark verify，先核对 Decode 插件是否包含上面固定的 draft 通信组修复。

## 性能测试

使用 [PD 压测脚本](../../../scripts/bench-pd.py)，输入为 64 条、每条 147,456 token 的冻结 RSC 输入，保留 `protocol.json` 与 `inputs/*.json.gz` 的目录关系。输入与协议需从已有 RSC 测试资产取得。先跑 BS1，再测试 8、16、24 的客户端并发。

```bash
python3 scripts/bench-pd.py \
  --router "$ROUTER_URL" \
  --prefill "http://$P_HOST:30000" \
  --decode "http://$D_HOST:30000" \
  --input /path/to/rsc/inputs/random_sharegpt64_seed42_147456_ids.json.gz \
  --output "pd-bench-$(date +%Y%m%d-%H%M%S)" \
  --bs 1 8 16 24
```

- Prefill 预热 131,072 个共享 token，每条追加 16,384 个 token、输出 1 token。检查 `reported_cache_min` 等于 131,072，再比较端到端 TTFT。
- Decode 使用独立输入，每条输出 1,024 token。分别记录平均 TPOT、Decode 窗口吞吐、完成请求数和 `decode_peak_running`。客户端并发与实际运行并发可能不同。
- 将 `summary.csv` 与逐请求结果、`loads.jsonl`、`server-info.json` 一起保留。任何缺失的输出或未完成请求都应在结果中标记。

## Trace 复跑

沿用 dump trace 的分阶段采样口径。Prefill 使用 eager、BS4、128 Ki token 缓存前缀加 16 Ki token 新输入、输出 1 token。Decode 对照 full graph 与 eager，输出 1,024 token，从达到目标运行批大小后跳过 10 步，采集第 11–20 步。96 GB 配方从低并发开始，按各 rank 的实际运行批大小选择采样窗口，旧独立 Decode 的 BS64 是历史负载。

采样请求发给实际执行该阶段的 worker。在支持分阶段 profiler 的运行时，先完成负载预热，再向 Decode worker 发送下列请求。示例在客户端确认窗口起点时触发，`output_dir` 是 worker 容器内的可写目录。

```bash
curl -fsS "http://$D_HOST:30000/start_profile" \
  -H 'Content-Type: application/json' \
  -d '{
    "output_dir":"/traces/decode",
    "profile_by_stage":true,
    "profile_stages":["decode"],
    "activities":["CPU","GPU"],
    "num_steps":10,
    "with_stack":true,
    "record_shapes":true,
    "merge_profiles":false,
    "profile_prefix":"decode"
  }'
```

Prefill 在共享前缀预热后发起采样，将目标改为 P worker，阶段设为 `prefill`，压测请求结束后调用该 worker 的 `POST /stop_profile`。精确复跑第 11–20 步时，使用可在达到目标批大小后按步触发的 profiler 客户端。

分别核对 8 个 rank 的 Python 栈、CPU 算子、HIP API 和 GPU kernel。full graph 额外核对 `hipGraphLaunch` 或等价 replay 事件。把 trace 导出与请求完成率分开记录，采样阶段的延迟单独标记。

96 GB 实卡记录覆盖真实 Mooncake 短请求，以及 8、16、24 客户端并发的长上下文验证。历史插件 trace 与性能采样使用过不同提交和运行环境，比较时应记录各自的宿主、插件、adapter、Torch/DTK 和 kernel wheel 版本。

文档结构参考 [add-model](https://github.com/HYGON-AI/inference-cookbook-das/tree/main/.claude/skills/add-model)与 [MiniMax-M3 on SGLang](https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/sglang/minimax-m3.md)。插件启动参数参考撤回前的 [MiniMax Plugin 脚本归档](https://github.com/chenyuege/sglang-plugin-das/tree/e4a57e6/tools/minimax_m31)。

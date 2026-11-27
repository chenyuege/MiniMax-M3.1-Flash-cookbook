# MiniMax-M3.1-Flash DP8EP8 on BW1300

## 模型简介

本配方在两台 BW1300 节点上运行 MiniMax-M3.1-Flash W4A8/W8A8。每节点 4 张 gfx938 HCU，每卡 144 GB，共 8 卡。并行参数为 TP8、Attention DP8、EP8，专家通信使用 DeepEP，Decode 使用 FP8 KV、FP8 索引和 DSpark Smooth INT8 draft。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin + adapter | BW1300 144 GB/卡 | 8（2 节点 × 4 卡） | DP8EP8 + EPLB | [**`>_`**](#双节点启动) |

## 环境准备

- 两节点使用相同的 SGLang v0.5.23、`hcu_minimax_m31` 插件、adapter-das、FlashMLA、LightOp、DeepEP 和 hipBLASLt 调优表。11 月 23 日的实测使用 DTK 26.04、Python 3.10、Torch 2.11、Triton 3.6 的 HCU 运行栈。使用已安装的兼容环境，保留其中的 Torch 和 Triton。
- 在两节点的容器内确认可见 4 张 BW1300、`/dev/kfd`、`/dev/dri` 和相应 RDMA 设备。两个容器使用可互通的 rank 0 地址和 rendezvous 端口。按各节点的 `ls /sys/class/infiniband` 结果配置 `ROCSHMEM_ALLOWED_IBV_DEVICES`。默认使用已验证的 DeepEP auto/INT8 通信，layered low-latency 关闭。
- 两节点准备同一份主模型、DSpark draft、EP8 EPLB 布局和 `hipblaslt.decode.accepted.config`。启动脚本可读取 `M31_RUNTIME_DIR/bootstrap/sitecustomize.py`。使用已安装到 Python 环境的插件时，启动脚本也支持已有的 `PYTHONPATH`。
- 复现实测可使用 DP8 布局 `expert_distribution_recorder_1795437678.0379474.pt`，SHA-256 为 `448297c0ec563e78a0dd8cc824c6f7a8c41c06b824117a49332e92804a977bf7`。新拓扑或权重按本页[生成 EP8 布局](#生成-ep8-布局)。

在两个节点的容器内设置以下变量。模型路径指向包含 `config.json` 的目录，draft 与主模型同级时按示例设置。rank 0 地址和 RDMA 网卡用实际值替换。

```bash
export MASTER_IP=192.0.2.10
export DIST_PORT=5001 SERVER_PORT=11011
export M31_RUNTIME_DIR=/opt/minimax31-w4a8-plugin
export MODEL_ROOT=/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8
export M31_MODEL_PATH="$MODEL_ROOT/main-w4a8"
export M31_DSPARK_MODEL_PATH="$MODEL_ROOT/dspark-channel-int8-smooth-20261107"
export M31_EPLB_INIT_LOCATION=/assets/expert_distribution_recorder_1795437678.0379474.pt
export M31_HIPBLASLT_CONFIG="$M31_RUNTIME_DIR/config_file1110/hipblaslt.decode.accepted.config"
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_3

HIP_VISIBLE_DEVICES=0,1,2,3 python3 -c \
  'import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count()); assert torch.cuda.device_count() == 4'
sha256sum "$M31_EPLB_INIT_LOCATION" "$M31_HIPBLASLT_CONFIG"
```

通过已有 HCU 运行镜像部署时，按[镜像加载与设备透传示例](minimax-m3.1-flash-dp16ep16.md#加载-1126-运行镜像)准备两台节点。将本仓库挂载到容器内，使用本页的 DP8EP8 启动脚本与 EP8 布局，核对环境变量中的模型与资产路径。容器方式换档需结束两节点的原服务，再启动同一新模式。

## 启动命令

### 双节点启动

使用 [serve-dp8ep8.sh](../../../scripts/serve-dp8ep8.sh)。节点 0 设置 `NODE_RANK=0`，节点 1 设置 `NODE_RANK=1`。两边填写同一个 `MASTER_IP`。每个节点运行下列一种模式，先执行 `--print-args` 预览，再去掉该参数同时启动。

| 模式 | KV 路径 | 全局请求上限 | 每 rank Decode Graph 档位 | 147,456 / 1,024 tokens 的历史实测 |
| --- | --- | ---: | --- | --- |
| `rsc` | 真实 Prefill/KV，保留 Radix Cache | 80 | 1–10 | 用于生成和前缀缓存验证 |
| `fake` | Fake Prefill Decode，关闭 HiSparse | 80 | 1–10 | BS8–64 |
| `hisparse` | Fake Prefill Decode，开启按 head 换页 | 128 | 1–16 | BS8–128 |
| `calibrate` | Fake Prefill Decode，记录 EP8 专家分布 | 80 | 1–10 | 已用 BS32 采样生成 EP8 布局 |

```bash
export NODE_RANK=0                       # 节点 1 改为 1
SGLANG_SIMULATE_ACC_LEN=0 bash scripts/serve-dp8ep8.sh rsc "$MASTER_IP" "$NODE_RANK" --print-args

# Fake Prefill 测试使用 3.6 模拟接受长度
SGLANG_SIMULATE_ACC_LEN=3.6 bash scripts/serve-dp8ep8.sh fake "$MASTER_IP" "$NODE_RANK" --print-args
SGLANG_SIMULATE_ACC_LEN=3.6 bash scripts/serve-dp8ep8.sh hisparse "$MASTER_IP" "$NODE_RANK" --print-args
```

主 KV 与索引使用 FP8 E4M3FN，page size 128，静态显存比例 0.95，KV token 容量自动定容。DSpark 每次提议 7 个位置，主模型一次验证 8 个位置。实际启动使用 native FP8 Verify fixed8。默认 context length 为 163,840，可以用 `M31_DECODE_CONTEXT_LENGTH` 覆盖。更改全局请求上限或局部 Graph 档位时，两节点使用相同的 `M31_MAX_RUNNING_REQUESTS` 和 `M31_GRAPH_MAX_LOCAL_BS`。

`hisparse` 默认使用 `{"device_buffer_size":32768,"host_to_device_ratio":2.29}`、Verify 直索引、HIP 稳定槽位和 `256×8` 按 head 换页。实测 Host KV 单 rank 分配约 **148.26 GB**，每节点 4 rank 合计约 **593 GB**，另需进程和系统内存。两个节点实测可用宿主内存约 **670–673 GiB**。同时检查宿主机 `MemAvailable` 与容器 cgroup 的可用余量。可按实际资源下调 `M31_HISPARSE_CONFIG` 中的比例，再重新核对可驻留 BS。

### Docker 启动示例

在两台宿主机上进入仓库根目录，设置前面的环境变量和各自的 `NODE_RANK`。`M31_IMAGE` 是已加载的 HCU 运行镜像标签。将模型和资产的宿主路径填入下例，确保挂载后的路径与 `M31_MODEL_PATH`、`M31_DSPARK_MODEL_PATH` 和 `M31_EPLB_INIT_LOCATION` 一致。

```bash
export M31_IMAGE=your-hcu-runtime-image:tag
export CHECKPOINTS_DIR=/path/to/model-root
export ASSETS_DIR=/path/to/dp8-assets
docker run --rm --init --name "minimax-m31-dp8-$NODE_RANK" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v /module:/module:ro \
  -v "$PWD:/cookbook:ro" -v "$CHECKPOINTS_DIR:/model:ro" -v "$ASSETS_DIR:/assets:ro" \
  -e M31_RUNTIME_DIR -e M31_MODEL_PATH -e M31_DSPARK_MODEL_PATH \
  -e M31_EPLB_INIT_LOCATION -e M31_HIPBLASLT_CONFIG \
  -e ROCSHMEM_ALLOWED_IBV_DEVICES -e DIST_PORT -e SERVER_PORT \
  -e SGLANG_SIMULATE_ACC_LEN=3.6 --entrypoint /bin/bash "$M31_IMAGE" \
  /cookbook/scripts/serve-dp8ep8.sh fake "$MASTER_IP" "$NODE_RANK"
```

开启 HiSparse 时将最后一行的 `fake` 改成 `hisparse`。如需在 Docker 内执行 `calibrate`，另挂载一个可写目录，将 `M31_EPLB_DUMP_DIR` 指向该挂载点。

### 启动后检查

在能访问 rank 0 的客户端检查服务。`/server_info` 应显示 TP8、Attention DP8、EP8 与 `nnodes=2`。`/v1/loads` 应返回 8 个 DP rank。核对所选模式的 HiSparse、最大请求数和每 rank token 容量，再进行压测。

```bash
export M31_URL="http://$MASTER_IP:$SERVER_PORT"
curl -fsS "$M31_URL/health"
curl -fsS "$M31_URL/server_info"
curl -fsS "$M31_URL/v1/loads?include=core,memory,spec"
```

## API 调用

在 `rsc` 模式检查真实生成。开启模拟接受长度的 Fake Prefill 用于 Decode 计时，真实接受长度和生成质量在 `rsc` 模式验证。

```bash
curl -fsS "$M31_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"法国的首都是哪里？只回答城市名称。","sampling_params":{"temperature":0,"max_new_tokens":32}}'
```

## 性能测试

### 校验冻结输入

在客户端进入仓库根目录，使用能导入 SGLang v0.5.23 serving benchmark 的 Python 环境。`M31_MODEL_PATH` 和 `M31_TOKENIZER_PATH` 使用客户端可读的路径。已有冻结输入可从校验开始。

```bash
export M31_TOKENIZER_PATH="$M31_MODEL_PATH"
export INPUTS="$PWD/inputs/random_sharegpt64_seed42_147456_ids.json.gz"

# 使用本地 ShareGPT JSON 生成输入时执行
python3 scripts/bench-dp16ep16-fake.py --inputs "$INPUTS" \
  --prepare-from-sharegpt /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
  --tokenizer "$M31_TOKENIZER_PATH"

python3 scripts/bench-dp16ep16-fake.py --inputs "$INPUTS" --check-inputs
```

相同输入的完整复现使用 `--require-reference-inputs`，对应 token IDs SHA-256 为 `28a24124753116c9cc88ea8bb4accb0968d1786cb346474c76cdb412a45c0dfe`。自行生成的输入可以省略该参数，开关组使用同一份输入文件和布局。正式轮次每请求输入 147,456 tokens，生成 1,024 tokens，seed 42、请求率 `inf`，按请求序号轮询 8 个 DP rank。

### 复跑 Fake Prefill Decode

同一个[压测脚本](../../../scripts/bench-dp16ep16-fake.py)以 `--dp-size 8` 选择两节点服务。`--bs` 是全局并发。建议每次重启后先跑一次 BS8 预热并独立保存，再用新的输出目录测正式档位。下面两组分别在 `fake`、`hisparse` 服务上运行，启动服务时配置 `SGLANG_SIMULATE_ACC_LEN=3.6`。

```bash
# 关 HiSparse
python3 scripts/bench-dp16ep16-fake.py --dp-size 8 --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --require-reference-inputs --require-residency \
  --bs 8 16 32 48 64 --output results/dp8-fake-off

# 开 HiSparse
python3 scripts/bench-dp16ep16-fake.py --dp-size 8 --host "$MASTER_IP" --port "$SERVER_PORT" \
  --inputs "$INPUTS" --require-reference-inputs --require-residency --hisparse \
  --bs 8 16 32 48 64 80 96 112 --output results/dp8-fake-on
for bs in 120 128; do
  python3 scripts/bench-dp16ep16-fake.py --dp-size 8 --host "$MASTER_IP" --port "$SERVER_PORT" \
    --inputs "$INPUTS" --require-reference-inputs --require-residency --hisparse \
    --bs "$bs" --repeats 2 --output "results/dp8-fake-on-bs$bs"
done
```

`--require-residency` 预查各 rank 的页对齐 KV 和请求预算。查看总览 `summary.csv` 与每档的 `validation.json`、`loads.jsonl` 确认完成率、实际满批、TPOT 与峰值。按页预算，每请求需要 `ceil((147456 + 1024 + 8) / 128) × 128 = 148608` tokens。关闭组实测 GPU 池 **1,288,192 tokens/rank**，预算 8 请求/rank。开启组 GPU 主池 **1,053,696**、有效逻辑池 **2,412,928**、Host 池 **2,413,056 tokens/rank**，预算 16 请求/rank。开关组分别采样到全局 **64** 与 **128** 条同时运行。

### BW1300 实测摘要

数据来自 2026-11-23 的 `dp8-sim36-20261123-203136`，采用上述 Fake Prefill 负载、各档全部请求完成、未开启 Prof。双值对应两轮。

| 全局 BS | HiSparse 关吞吐（tok/s） | 关平均 TPOT（ms） | HiSparse 开吞吐（tok/s） | 开平均 TPOT（ms） |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 554.13 | 13.73 | 536.07 | 14.27 |
| 16 | 984.84 | 14.93 | 881.21 | 15.81 |
| 32 | 1,581.61 | 17.77 | 1,376.90 | 20.85 |
| 48 | 1,993.70 | 20.25 | 1,438.86 | 29.79 |
| 64 | **2,503.77** | **20.56** | 2,121.88 | 25.65 |
| 80 | — | — | 2,055.41 | 32.64 |
| 96 | — | — | 1,946.88 | 41.75 |
| 112 | — | — | 2,251.40 | 39.71 |
| 120 | — | — | 2,159.81 / 2,164.37 | 44.60 / 44.85 |
| 128 | — | — | **2,354.13 / 2,410.72** | 42.38 / 42.84 |

BS64 关闭组最大 rank KV 峰值为 **92.20%**，同档开启组 GPU 热池/Host 池约 **25%/49%**。开启组 BS128 GPU 热池/Host 池峰值约 **50%/98%**。上述容量和吞吐对应实测的模型、镜像与负载，容器和驱动更新后依据启动日志重新核对。

## 生成 EP8 布局

两节点先用 `calibrate` 运行。指定能写入的 `M31_EPLB_DUMP_DIR`。该模式使用初始均匀布局与专家分布记录，可直接导出首份 EP8 PT 文件。

```bash
export M31_EPLB_DUMP_DIR=/path/to/writable/dp8-eplb-layout
SGLANG_SIMULATE_ACC_LEN=3.6 bash scripts/serve-dp8ep8.sh calibrate "$MASTER_IP" "$NODE_RANK"
```

在 rank 0 就绪后先跑一轮 BS8 预热。在客户端开启录制，运行 BS32 样本，然后停止并导出。

```bash
curl -fsS -X POST "$M31_URL/start_expert_distribution_record"
python3 scripts/bench-dp16ep16-fake.py --dp-size 8 --calibration \
  --host "$MASTER_IP" --port "$SERVER_PORT" --inputs "$INPUTS" --require-residency \
  --bs 32 --output results/dp8-eplb-calibration
curl -fsS -X POST "$M31_URL/stop_expert_distribution_record"
curl -fsS -X POST "$M31_URL/dump_expert_distribution_record"
```

在 rank 0 的 `M31_EPLB_DUMP_DIR` 取出新生成的 `expert_distribution_recorder_*.pt`，记录 SHA-256，分发到两节点并设置 `M31_EPLB_INIT_LOCATION`。结束两节点校准进程后，用 `fake`、`hisparse` 或 `rsc` 启动正式服务。正式评测采用 1,000,000 步重平衡间隔，使布局在本轮内保持稳定。

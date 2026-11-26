# MiniMax-M3.1-Flash 1P1D on BW1300 144 GB

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1300 144 GB/卡 | 16（8P + 8D） | Mooncake 1P1D | [启动服务](#启动服务) |

## 环境准备

两台 Linux 节点各有 8 张 gfx938 HCU。分别承载 Prefill 和 Decode，保持两个节点的 worker HTTP 端口及 Prefill bootstrap 端口互通。容器挂载 HCU、RDMA、宿主 HyHAL 和模型目录。RDMA 设备名使用各节点在 `ibdev2netdev` 中显示为 Up 的设备。

验证环境使用相同的 PD 镜像。SGLang 为 `v0.5.23`（`297e54fd105e974cdf401e5b4d20253a22c16f90`），HCU Plugin 为 `6a6f333dd291e5975b99fe0356e42815330aa61f`，Adapter 为 `5d2b1e456c081bf99460b234a7d2ced92b7bb09f`。使用 DTK 26.10、Torch 2.11 和 Triton 3.6，并保持 DeepEP、DeepGEMM、FlashAttention、FlashMLA 和 LightOP 的 wheel 与 Torch ABI 匹配。

主模型默认位于 `/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8`，DSpark 使用其下的 `dspark-channel-int8-smooth-20261107`。两端使用相同的 FP8 E4M3 KV 及 128-token 页。

| Worker | 并行配置 | `--mem-fraction-static` | 最大运行请求数 |
| --- | --- | --- | --- |
| Prefill | TP8、CP8、EP8，zigzag | 0.90 | 80 |
| Decode | TP8、Attention DP8、EP8 | 0.95 | 64 |

Decode 关闭 HiSparse，捕获 1 到 8 的 CUDA Graph batch。脚本默认将 Mooncake KV 等待上限设为 900 秒。147,456-token 的独立输入在 BS48 时有请求超过原来的 300 秒上限。可用 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` 覆盖这一设置。

## 启动服务

在两台节点分别设置 `role=prefill` 和 `role=decode`，并将 `M31_IMAGE`、`CHECKPOINTS_DIR`、`M31_PD_IB_DEVICES` 和 `ROCSHMEM_ALLOWED_IBV_DEVICES` 设为当前节点的实际值。在仓库根目录运行以下命令。

```bash
docker run --rm --name "minimax-m31-$role" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v /module:/module:ro \
  -v "$CHECKPOINTS_DIR:/model:ro" -v "$PWD:/cookbook:ro" \
  -e M31_PORT=21110 -e M31_BOOTSTRAP_PORT=28998 \
  -e M31_PD_IB_DEVICES -e ROCSHMEM_ALLOWED_IBV_DEVICES \
  --entrypoint bash "$M31_IMAGE" /cookbook/scripts/serve-1p1d-bw1300.sh "$role"
```

在 Router 所在节点将 `PREFILL_HOST` 与 `DECODE_HOST` 设为各 worker 的可达地址，使用已安装的 `sglang-router==0.3.2` 启动 Router。

```bash
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://$PREFILL_HOST:21110" 28998 \
  --decode "http://$DECODE_HOST:21110" \
  --host 127.0.0.1 --port 21111 --policy round_robin
```

## 验证部署

在已设置 RDMA 变量的环境中，为两个角色分别执行 `bash scripts/serve-1p1d-bw1300.sh <role> --print-args`。启动后检查 P/D 各自的 `/server_info`，确认 `disaggregation_mode` 分别为 `prefill` 和 `decode`，传输后端均为 `mooncake`。

通过 Router 发送首条请求，确认 HTTP 200 且返回 `PARIS`。

```bash
curl -fsS http://127.0.0.1:21111/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8",
    "messages": [{"role": "user", "content": "Reply with exactly PARIS."}],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

长序列验证使用 147,456-token 输入。Prefill 测试使用 131,072-token 共享前缀，Decode 测试使用独立输入和 1,024 个生成 token。统计 BS8、16、32、40、48、56、64 的完成率、TTFT 和 TPOT。性能测量结束后单独运行 EvalScope。

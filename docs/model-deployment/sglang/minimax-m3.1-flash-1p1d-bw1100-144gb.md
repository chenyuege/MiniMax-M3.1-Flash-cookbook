# MiniMax-M3.1-Flash 1P1D on BW1100 144 GB

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1100 144 GB/卡 | 16（8P + 8D） | Mooncake 1P1D | [启动服务](#启动服务) |

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

### 1. 在两个节点加载镜像

实测镜像标签为 `minimax-m31-w4a8-pd:sglang0.5.23-plugin6a6f333-20261125-205534`。将 `docker save` 导出的 `.docker.tar.zst` 归档和同名 `.sha256` 文件放在两台节点都能读取的位置。将 `M31_ARCHIVE` 指向归档，分别在 Prefill 和 Decode 节点执行以下命令。

```bash
export M31_IMAGE=minimax-m31-w4a8-pd:sglang0.5.23-plugin6a6f333-20261125-205534
M31_ARCHIVE=/srv/images/minimax-m31-w4a8-pd_sglang0.5.23-plugin6a6f333-20261125-205534.docker.tar.zst

(
  cd "$(dirname "$M31_ARCHIVE")"
  sha256sum -c "$(basename "$M31_ARCHIVE").sha256"
) || exit 1
set -o pipefail
zstd -dc "$M31_ARCHIVE" | docker load
docker image inspect --format '{{.Id}}' "$M31_IMAGE"
```

两端应得到同一个镜像 ID，`sha256:fd7e6b748dc9f5caa6b88aebdc9e2e651fda1102b9a2df7a3ea72a6d6ccbbae6`。

### 2. 准备模型与节点参数

两台节点各自从 cookbook 根目录执行。将 `CHECKPOINTS_DIR` 指向主模型目录的父目录，确认主模型和 DSpark draft 的 `config.json` 都可读。

```bash
export CHECKPOINTS_DIR=/srv/checkpoints
model_dir="$CHECKPOINTS_DIR/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8"
test -r "$model_dir/config.json"
test -r "$model_dir/dspark-channel-int8-smooth-20261107/config.json"
hy-smi --showproductname --showmeminfo vram
ibdev2netdev
```

在每台节点根据实际 RDMA 设备名称分别设置 Mooncake 和 DeepEP 的网卡列表。下面以 `mlx5_2` 到 `mlx5_9` 为例。

```bash
export M31_PD_IB_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_ALLOWED_IBV_DEVICES="$M31_PD_IB_DEVICES"
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=900
```

在 Prefill 节点设置 `role=prefill`，在 Decode 节点设置 `role=decode`。先在各自节点展开实际命令，确认角色、显存占比和 RDMA 参数。

```bash
bash scripts/serve-1p1d-bw1100-144gb.sh "$role" --print-args
```

### 3. 启动 Prefill 与 Decode 容器

在两台节点分别用上一步的角色变量执行以下命令。两台节点的 worker 都使用 HTTP 端口 `21110`，其中 Prefill 还监听 Mooncake bootstrap 端口 `28998`。

```bash
docker run -d --name "minimax-m31-$role" \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --device /dev/infiniband \
  --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v /module:/module:ro \
  -v "$CHECKPOINTS_DIR:/model:ro" -v "$PWD:/cookbook:ro" \
  -e M31_PORT=21110 -e M31_BOOTSTRAP_PORT=28998 \
  -e M31_PD_IB_DEVICES -e ROCSHMEM_ALLOWED_IBV_DEVICES \
  -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT \
  --entrypoint bash "$M31_IMAGE" /cookbook/scripts/serve-1p1d-bw1100-144gb.sh "$role"

docker logs --tail 100 "minimax-m31-$role"
curl --noproxy '*' -fsS http://127.0.0.1:21110/health
curl --noproxy '*' -fsS http://127.0.0.1:21110/server_info
```

等待两端日志出现 `Application startup complete`，然后检查 `/server_info` 中的 `disaggregation_mode` 分别为 `prefill` 和 `decode`，`disaggregation_transfer_backend` 均为 `mooncake`。Prefill 与 Decode 的 `mem_fraction_static` 分别为 `0.90` 和 `0.95`。

### 4. 启动 Router 容器

可以将 Router 和客户端放在 Decode 节点。使用与 worker 相同的镜像，将 `sglang-router==0.3.2` 安装到独立目录，然后以只读方式挂载该目录作为 Router 的 Python 模块路径。

```bash
M31_ROUTER_PYTHON="$PWD/runtime/router-0.3.2"
mkdir -p "$M31_ROUTER_PYTHON"
docker run --rm -e HCU_ADAPTER_HOST=none \
  -v "$M31_ROUTER_PYTHON:/router" --entrypoint python3 "$M31_IMAGE" \
  -m pip install --no-deps --target /router 'sglang-router==0.3.2'
```

将下面的示例主机名换成两个 worker 的实际可达地址。先从 Router 节点检查两个 worker 的健康接口，再启动 Router。以下配方将客户端端口 `21111` 绑定到本机。

```bash
PREFILL_HOST=prefill.example.net
DECODE_HOST=decode.example.net
curl --noproxy '*' -fsS "http://$PREFILL_HOST:21110/health"
curl --noproxy '*' -fsS "http://$DECODE_HOST:21110/health"

docker run -d --name minimax-m31-router --network host \
  -e HCU_ADAPTER_HOST=none -e PYTHONPATH=/router \
  -v "$M31_ROUTER_PYTHON:/router:ro" --entrypoint python3 "$M31_IMAGE" \
  -m sglang_router.launch_router --pd-disaggregation \
  --prefill "http://$PREFILL_HOST:21110" 28998 \
  --decode "http://$DECODE_HOST:21110" \
  --host 127.0.0.1 --port 21111 --policy round_robin

docker logs --tail 100 minimax-m31-router
curl --noproxy '*' -fsS http://127.0.0.1:21111/v1/models
```

## 验证部署

在 Router 所在节点发送首条请求，确认 HTTP 200 且返回 `PARIS`。压测与 EvalScope 客户端也使用 Router 的 `/v1` 地址。

```bash
curl --noproxy '*' -fsS http://127.0.0.1:21111/v1/chat/completions \
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

## 查看日志与停止服务

在各容器所在节点用 `docker logs -f minimax-m31-prefill`、`docker logs -f minimax-m31-decode` 或 `docker logs -f minimax-m31-router` 查看实时日志。结束测试后先在 Router 节点执行 `docker stop minimax-m31-router`，再在两台 worker 节点分别执行 `docker stop minimax-m31-prefill` 与 `docker stop minimax-m31-decode`。

使用同一镜像和参数恢复时可以用 `docker start <容器名>`。调整镜像、网卡或服务参数时，在确认已停止并保存日志后，删除对应容器并重新执行该角色的 `docker run`。

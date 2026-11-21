# MiniMax M3.1 Flash 两节点 CP8EP8 Prefill

本配方用于 BW1300 144 GB/卡的两节点 Prefill 服务。每节点 4 卡，TP8、zigzag CP8、DeepEP normal EP8，节点间使用 MNVL。采用 FP8 KV 和 packed FP8 AllGather，静态 EPLB 用 8 个冗余专家槽。RSC 实测使用 65,536-token Prefill chunk、0.90 静态显存比例、FA3 后端和 split 共享专家调度。

[启动脚本](../../../scripts/serve-cp8ep8-prefill.sh)

## 环境准备

两节点使用一致的 HCU/DTK、Torch、FlashMLA、LightOp 和 DeepEP 运行时。社区 SGLang 版本为 0.5.23。Plugin 需要上游 `proj/minimax-m3.1` 含 CP8EP8 packed FP8 KV 支持的版本。Adapter 需包含下面的广播压缩提交。全部 rank 更新后再开启压缩。

[Plugin 评审提交](https://github.com/chenyuege/sglang-plugin-das/commit/722130d00542cafdd51a7eb13cc0a67c7e4df32e) · [Adapter 广播压缩提交](https://github.com/chenyuege/adapter-das/commit/00efd158865dbb496134a907b519b9e489fe012b)

将匹配权重版本的静态 EPLB 映射、DeepEP SMS 配置、hipBLASLt 调优文件、ROCm Tensile 库目录和模型挂载提供给两节点。布局默认文件名为 `m31-cp8ep8-from-dp16-logical-count.pt`，形状 60 × 136。用目标 checkpoint 对应的统计生成映射。

- `M31_RUNTIME_DIR` 默认为 `/opt/minimax-cp8ep8`。`M31_SITE_DIR` 和 `M31_ADAPTER_DIR` 分别覆盖 `site` 和 `adapter` 路径。
- `M31_MODEL_PATH` 默认为 `/model/main-w4a8`。`M31_EPLB_INIT_LOCATION` 覆盖运行时目录中 `layout/` 下的静态映射路径。
- `M31_ASSETS_DIR` 默认为运行时的 `config/`。`M31_DEEPEP_CONFIG`、`M31_HIPBLASLT_CONFIG` 和 `M31_ROCBLAS_TENSILE_LIBPATH` 可覆盖各自路径。
- 启动值采用 `HCU_MINIMAX_CP_PACK_FP8=1`、`HCU_DIST_BROADCAST_COMPRESS=1`、`SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=split` 和 `SGLANG_MINIMAX_CP_PACK_PREFILL_SEGMENTS=0`。统一设置 `HCU_DIST_BROADCAST_COMPRESS=0` 后重启两端可复现固定 EPLB 对照组。

## 启动

两节点使用 rank 0 的可达地址，`DIST_PORT` 默认为 13433。rank 0 的 HTTP 端口由 `SERVER_PORT` 设置，默认为 11010。按运行环境填好上一节的路径覆盖项。

```bash
# 在两个节点先核对各自的命令。RANK0_IP 是实际 rank 0 的通信地址。
bash -n scripts/serve-cp8ep8-prefill.sh
bash scripts/serve-cp8ep8-prefill.sh "$RANK0_IP" 0 --print-args
bash scripts/serve-cp8ep8-prefill.sh "$RANK0_IP" 1 --print-args

# 节点 0
bash scripts/serve-cp8ep8-prefill.sh "$RANK0_IP" 0

# 节点 1，在第二个终端运行
bash scripts/serve-cp8ep8-prefill.sh "$RANK0_IP" 1
```

## API 与性能验证

rank 0 启动完成后，在相同网络内检查服务和短请求。长上下文性能用已固定输入 SHA 的 RSC 负载单独测量。

```bash
curl -fsS "http://$RANK0_IP:11010/health"
curl -fsS "http://$RANK0_IP:11010/server_info"
curl -fsS "http://$RANK0_IP:11010/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"用一句话解释什么是上下文并行。","sampling_params":{"max_new_tokens":16,"temperature":0}}'
```

历史双节点 RSC 用 131,072 个共享缓存 tokens、16,384 个新增 tokens，输出 1 token。相对初始 trivial baseline，BS128 首轮 mean TTFT 为 48,653.27 → 40,647.41 ms，P95 为 83,977.13 → 73,110.05 ms。相对固定 EPLB split，压缩版两轮的 BS128 mean TTFT 中位数为 40,731.55 ms，EPLB split 对照为 45,093.05 ms。八 rank 带栈 trace 的 TP0 最大空泡为 4,203 → 570 ms。此次移植检查脚本语法、rank 0/1 的参数和环境默认值。上述 HCU 性能来自原双节点部署。

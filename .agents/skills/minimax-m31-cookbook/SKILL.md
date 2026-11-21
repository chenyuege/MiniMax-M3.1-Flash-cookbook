---
name: minimax-m31-cookbook
description: 维护和复跑 MiniMax-M3.1-Flash W4A8/W8A8 在 BW1100 96 GB 上的 SGLang HCU Plugin 配方。用于更新模型文档、Mooncake 1P1D 配置、API 验证、性能测试或 GPU trace。
---

# MiniMax-M3.1 Flash Cookbook

1. 阅读 [部署 Cookbook](../../../docs/model-deployment/sglang/minimax-m3.1-flash.md)和 [启动脚本](../../../scripts/serve.sh)。确认主模型、DSpark draft、SGLang、Plugin、adapter、Torch/DTK、FlashMLA 和 LightOp 的版本与实际运行环境一致。
2. 更新配方时，沿用 [上游 add-model](https://github.com/HYGON-AI/inference-cookbook-das/tree/main/.claude/skills/add-model)的模型列表、启动章节和 API 调用结构。同步更新根 README 支持矩阵、脚本参数和章节锚点。
3. 根据实际卡型和显存审核显存比例、KV token 容量、Prefill chunk、Decode 请求上限和 graph 批大小。保持 Mooncake P/D 的 FP8 KV 页布局及 DSpark draft 参数一致。
4. 先用 `bash -n scripts/serve.sh` 和两端的 `--print-args` 验证命令，再在目标机器检查 `/server_info`、短请求和长请求。记录提交、设备、输入、输出长度、实际并发、完成率和失败项。
5. trace 与性能分别验收。按 Cookbook 的口径验证缓存命中、采样步数、CPU/HIP/GPU 事件与 graph replay。性能只采用完整轮次。未能上实卡时，标明仅做过静态验证。

6. CP8EP8 Prefill 使用[双节点配方](../../../docs/model-deployment/sglang/minimax-m3.1-flash-cp8ep8-prefill.md)和[启动脚本](../../../scripts/serve-cp8ep8-prefill.sh)。核对静态布局、8 个冗余槽、packed FP8 KV、TP rank 广播帧兼容和两端开关。先为 rank 0/1 各执行一次 `--print-args`，再验证短请求和 RSC 前缀命中。

# MiniMax-M3.1-Flash-cookbook

在海光 HCU 上部署和验证 MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8。BW1100 的 96 GB 与 144 GB 档均有基于 SGLang 0.5.23、HCU Plugin 和 Mooncake 的双节点 1P1D 配方。BW1300 也支持四节点 DP16EP16 和双节点 CP8EP8 Prefill。

## 支持矩阵

| 模型 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 |
| --- | --- | --- | --- | --- | --- |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1100 96 GB | 16（8P + 8D） | [1P1D](docs/model-deployment/sglang/minimax-m3.1-flash.md) |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1100 144 GB/卡 | 16（8P + 8D） | [1P1D](docs/model-deployment/sglang/minimax-m3.1-flash-1p1d-bw1100-144gb.md) |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | HCU fixed8 基线 / 0.5.23 + HCU Plugin | BW1300 144 GB/卡 | 16（4 节点 × 4 卡） | [DP16EP16 + EPLB](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md) |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin + adapter | BW1300 144 GB/卡 | 8（2 节点 × 4 卡） | [CP8EP8 Prefill](docs/model-deployment/sglang/minimax-m3.1-flash-cp8ep8-prefill.md) |

- [部署 Cookbook](docs/model-deployment/sglang/minimax-m3.1-flash.md)涵盖环境、P/D 与 Router 启动、API 验证、压测和 trace。
- [运行环境](docs/model-deployment/sglang/minimax-m3.1-flash.md#环境准备)使用社区 SGLang `v0.5.23`、海光 `sglang-plugin-das:proj/minimax-m3.1` 和 `adapter-das:main`。
- [容器与服务运行](docs/model-deployment/sglang/minimax-m3.1-flash.md#容器构建与服务管理)涵盖基础镜像要求、镜像构建、Compose 启停及日志。
- [启动脚本](scripts/serve.sh)提供与文档相同的 96 GB 参数，可以输出完整启动命令。
- [BW1100 144 GB 1P1D 启动脚本](scripts/serve-1p1d-bw1100-144gb.sh)提供双节点 Mooncake 启动参数。[部署说明](docs/model-deployment/sglang/minimax-m3.1-flash-1p1d-bw1100-144gb.md)
- [仓库 Skill](.agents/skills/minimax-m31-cookbook/SKILL.md)用于维护配方、复跑验证以及更新支持矩阵。
- [DP16EP16 运行镜像](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md#加载-1126-运行镜像)记录 1126 版本、校验和及容器内启动方式。[BS128/144 Trace 与性能对照](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md#1126-插件版-fake-prefill-decode)记录 HiSparse 开关和按 head 优化结果。
- [DP16EP16 性能测试脚本](scripts/bench-dp16ep16-fake.py)支持关闭和开启 HiSparse 的 Fake Prefill 并发扫描与 BS256 历史序列对照。[运行步骤](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md#复跑-fake-prefill-decode)
- [CP8EP8 Prefill 启动脚本](scripts/serve-cp8ep8-prefill.sh)支持两节点静态 EPLB、packed FP8 AllGather 和请求广播压缩。

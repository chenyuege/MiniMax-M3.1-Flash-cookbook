# MiniMax-M3.1-Flash-cookbook

在海光 HCU 上部署和验证 MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8。96 GB 配方使用 SGLang 0.5.23、HCU Plugin 和 Mooncake，实现单 Prefill 节点加单 Decode 节点的 PD 分离。BW1300 144 GB 配方使用四节点 DP16EP16 和 EPLB。

## 支持矩阵

| 模型 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 |
| --- | --- | --- | --- | --- | --- |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | 0.5.23 + HCU Plugin | BW1100 96 GB | 16（8P + 8D） | [1P1D](docs/model-deployment/sglang/minimax-m3.1-flash.md) |
| [MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8](https://www.modelscope.cn/models/hygon/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8) | INT4 W4A8 + INT8 W8A8 | HCU fixed8 基线 / 0.5.23 + HCU Plugin | BW1300 144 GB/卡 | 16（4 节点 × 4 卡） | [DP16EP16 + EPLB](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md) |

- [部署 Cookbook](docs/model-deployment/sglang/minimax-m3.1-flash.md)涵盖环境、P/D 与 Router 启动、API 验证、压测和 trace。
- [运行环境](docs/model-deployment/sglang/minimax-m3.1-flash.md#环境准备)使用社区 SGLang `v0.5.23`、海光 `sglang-plugin-das:proj/minimax-m3.1` 和 `adapter-das:main`。
- [容器与服务运行](docs/model-deployment/sglang/minimax-m3.1-flash.md#容器构建与服务管理)涵盖基础镜像要求、镜像构建、Compose 启停及日志。
- [启动脚本](scripts/serve.sh)提供与文档相同的 96 GB 参数，可以输出完整启动命令。
- [仓库 Skill](.agents/skills/minimax-m31-cookbook/SKILL.md)用于维护配方、复跑验证以及更新支持矩阵。
- [DP16EP16 启动与性能](docs/model-deployment/sglang/minimax-m3.1-flash-dp16ep16.md)包含 fixed8 基线和社区版插件的启动命令，以及四节点 BS128、BS144 的正式性能。

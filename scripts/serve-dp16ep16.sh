#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 3 ]]; then
    echo "Usage: bash $0 {rsc|fake|hisparse} MASTER_IP NODE_RANK [--print-args | SGLang options]"
    [[ ${1:-} == --help ]] || exit 2
    exit
fi
variant=$1
master=${2:?dist-init host}
rank=${3:?node rank}
shift 3
[[ $rank =~ ^[0-3]$ ]] || { echo "NODE_RANK must be 0 through 3." >&2; exit 2; }
print_args=0
if [[ ${1:-} == --print-args ]]; then
    print_args=1
    shift
fi
runtime=${M31_RUNTIME_DIR:-/opt/minimax31-w4a8-plugin}
py=${M31_PYTHON:-python3}
model=${M31_MODEL_PATH:-/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8}
draft=${M31_DSPARK_MODEL_PATH:-"$model/dspark-channel-int8-smooth-20261107"}
layout=${M31_EPLB_INIT_LOCATION:-$runtime/layout/expert_distribution_recorder_1794400407.277366.pt}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export PYTHONPATH="$runtime/source/adapter-das:$runtime/site"
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_minimax_m31 HCU_SLIMQUANT=0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3} HCU_PREFILL_CP=0 GPU_ARCHS=gfx938 GPU_MAX_HW_QUEUES=4
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1 HCU_PLUGIN_MINIMAX_M31_DEEPEP=1 HCU_DEEPEP_BRIDGE=0
export HCU_PLUGIN_MINIMAX_M31_HISPARSE=0
export SGLANG_USE_INT4_W4A8=1 SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 SGLANG_GROUPGEMM=1
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1 LIGHTOP_PER_TOKEN_INT8_METHOD=triton
export W8A8_SUPPORT_METHODS=3 SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1 SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1
export SGLANG_USE_DEEPGEMM_MOE=0 SGLANG_DEEPEP_USE_MOE_EP_GROUP=1 AITER_AR_TRANSPORT=ipc HCU_AITER_AR=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1 SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1 ROCSHMEM_IPC_MNVL=1 HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_HDP=1 ROCSHMEM_GDR_DISABLE_XDP=1 ROCSHMEM_DISABLE_HDP_FLUSH=1
if (( ! print_args )) && [[ -z ${ROCSHMEM_ALLOWED_IBV_DEVICES:-} ]]; then
    devices=(/sys/class/infiniband/*)
    [[ -e ${devices[0]} ]] || { echo "No RDMA devices. Set ROCSHMEM_ALLOWED_IBV_DEVICES." >&2; exit 1; }
    ROCSHMEM_ALLOWED_IBV_DEVICES=$(printf '%s\n' "${devices[@]##*/}" | paste -sd, -)
fi
export ROCSHMEM_ALLOWED_IBV_DEVICES
export SGLANG_EPLB_AUTO_START_RECORDER=0 SGLANG_MINIMAX_INDEX_FP8=dot SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1 MSA_STAGE1_KERNEL=v8 FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS=${SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS:-8}
export SGLANG_SIMULATE_ACC_LEN=0 SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY=1 SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_DSPARK_FOLDED_PROPOSAL=0 SGLANG_DSPARK_USE_GREEDY_DRAFT=0 M31_STAGE1_GRAPH_MAX_KV_LEN=148608
export HIPBLASLT_TUNING_OVERRIDE_FILE=${M31_HIPBLASLT_CONFIG:-$runtime/config_file1110/hipblaslt.decode.accepted.config}
case "$SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS" in
    0|1|2|4|8) ;;
    *) echo "SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS must be 0, 1, 2, 4 or 8." >&2; exit 2 ;;
esac
options=(
    --model-path "$model" --quantization slimquant_w4a8_marlin --trust-remote-code --random-seed 42
    --tp-size 16 --attn-dp-size 16 --ep-size 16 --moe-dense-tp-size 1 --enable-dp-lm-head
    --enable-dp-attention-local-control-broadcast --load-balance-method round_robin
    --moe-a2a-backend deepep --moe-runner-backend deep_gemm --deepep-mode auto
    --deepep-dispatcher-output-dtype int8 --disable-shared-experts-fusion
    --attention-backend triton --prefill-attention-backend triton --decode-attention-backend fa3
    --kv-cache-dtype fp8_e4m3 --page-size 128 --cuda-graph-backend-prefill disabled
    --cuda-graph-backend-decode full --cuda-graph-bs-decode 1 2 4 5 6 8 9 10
    --speculative-algorithm DSPARK --speculative-draft-model-path "$draft"
    --speculative-draft-model-quantization compressed-tensors --speculative-draft-attention-backend triton
    --speculative-dspark-block-size 7 --speculative-num-draft-tokens 8
    --context-length "${M31_DECODE_CONTEXT_LENGTH:-163840}" --chunked-prefill-size 8192
    --max-running-requests "${M31_MAX_RUNNING_REQUESTS:-160}" --mem-fraction-static "${M31_MEM_FRACTION:-0.95}"
    --enable-eplb --init-expert-location "$layout" --eplb-algorithm deepseek
    --eplb-rebalance-num-iterations 1000000 --expert-distribution-recorder-buffer-size 128
    --eplb-rebalance-layers-per-chunk 4 --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
    --dist-init-addr "$master:${DIST_PORT:-5001}" --nnodes 4 --node-rank "$rank"
    --host 0.0.0.0 --port "${SERVER_PORT:-11011}"
)
case "$variant" in
    rsc) ;;
    fake|hisparse) options+=(--disaggregation-mode decode --disaggregation-transfer-backend fake --disable-radix-cache) ;;
    *) echo "Unknown mode: $variant" >&2; exit 2 ;;
esac
if [[ $variant == hisparse ]]; then
    export HCU_PLUGIN_MINIMAX_M31_HISPARSE=1 M31_HISPARSE_FAKE_VALIDATE=1
    export SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=${SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB:-4}
    hisparse_config=${M31_HISPARSE_CONFIG:-'{"device_buffer_size":65536,"host_to_device_ratio":1}'}
    options+=(--enable-hisparse --hisparse-config "$hisparse_config" --disaggregation-decode-host-receive-threshold 1)
fi
command=("$py" -m sglang.launch_server "${options[@]}" "$@")
if (( print_args )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi
for path in "$model/config.json" "$draft/config.json" "$layout" "$HIPBLASLT_TUNING_OVERRIDE_FILE"; do
    [[ -s $path && -r $path ]] || { echo "Missing or unreadable file: $path" >&2; exit 1; }
done
printf 'rank=%s mode=%s verify_split=%s RDMA=%s\n' \
    "$rank" "$variant" "$SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS" "$ROCSHMEM_ALLOWED_IBV_DEVICES"
exec "${command[@]}"

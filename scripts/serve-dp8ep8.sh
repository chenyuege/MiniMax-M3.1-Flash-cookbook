#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --help || $# -lt 3 ]]; then
    echo "Usage: bash $0 {rsc|fake|hisparse|calibrate} MASTER_IP NODE_RANK [--print-args | SGLang options]"
    [[ ${1:-} == --help ]] || exit 2
    exit
fi

mode=$1
master=${2:?dist-init host}
rank=${3:?node rank}
shift 3
case "$mode" in
    rsc|fake|hisparse|calibrate) ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
[[ $rank =~ ^[01]$ ]] || { echo "NODE_RANK must be 0 or 1." >&2; exit 2; }
print_args=0
if [[ ${1:-} == --print-args ]]; then
    print_args=1
    shift
fi

runtime=${M31_RUNTIME_DIR:-/opt/minimax31-w4a8-plugin}
py=${M31_PYTHON:-python3}
model=${M31_MODEL_PATH:-/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8}
draft=${M31_DSPARK_MODEL_PATH:-${M31_DRAFT_PATH:-"$model/dspark-channel-int8-smooth-20261107"}}
layout=${M31_EPLB_INIT_LOCATION:-/assets/expert_distribution_recorder_1795437678.0379474.pt}
export HIPBLASLT_TUNING_OVERRIDE_FILE=${M31_HIPBLASLT_CONFIG:-$runtime/config_file1110/hipblaslt.decode.accepted.config}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
if [[ -s $runtime/bootstrap/sitecustomize.py ]]; then
    export PYTHONPATH="$runtime/bootstrap"
fi
export SETUPTOOLS_USE_DISTUTILS=${SETUPTOOLS_USE_DISTUTILS:-stdlib}
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_minimax_m31 HCU_SLIMQUANT=0
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3} HCU_PREFILL_CP=0 GPU_ARCHS=gfx938 GPU_MAX_HW_QUEUES=4
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1 HCU_PLUGIN_MINIMAX_M31_DEEPEP=1 HCU_DEEPEP_BRIDGE=0
export HCU_PLUGIN_MINIMAX_M31_HISPARSE=0
export SGLANG_USE_INT4_W4A8=1 SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0 SGLANG_GROUPGEMM=1
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1 LIGHTOP_PER_TOKEN_INT8_METHOD=triton
export W8A8_SUPPORT_METHODS=3 SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1 SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1
export SGLANG_USE_DEEPGEMM_MOE=0 SGLANG_DEEPEP_USE_MOE_EP_GROUP=1 AITER_AR_TRANSPORT=ipc HCU_AITER_AR=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=${DEEPEP_ENABLE_LL_LAYERED_OPT:-0} SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1 ROCSHMEM_IPC_MNVL=1 HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_HDP=1 ROCSHMEM_GDR_DISABLE_XDP=1 ROCSHMEM_DISABLE_HDP_FLUSH=1
if (( ! print_args )) && [[ -z ${ROCSHMEM_ALLOWED_IBV_DEVICES:-} ]]; then
    devices=(/sys/class/infiniband/*)
    [[ -e ${devices[0]} ]] || { echo "No RDMA devices. Set ROCSHMEM_ALLOWED_IBV_DEVICES." >&2; exit 1; }
    ROCSHMEM_ALLOWED_IBV_DEVICES=$(printf '%s\n' "${devices[@]##*/}" | paste -sd, -)
fi
export ROCSHMEM_ALLOWED_IBV_DEVICES

export SGLANG_EPLB_AUTO_START_RECORDER=0 SGLANG_MINIMAX_EPLB_PREFILL_ONLY=0
export SGLANG_MINIMAX_INDEX_FP8=dot SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export SGLANG_MINIMAX_CACHE_DECODE_INDEX_METADATA=1 SGLANG_M3_OPTIMIZE_SPARSE_ATTN_PREP=1
export SGLANG_OPT_USE_BF16_ROUTER_GEMM=1 SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1 SGLANG_MINIMAX_EXTEND_ROUTER_GEMV=0
export MSA_STAGE1_KERNEL=v8 FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export M31_STAGE1_GRAPH_MAX_KV_LEN=${M31_STAGE1_GRAPH_MAX_KV_LEN-148608}
export SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS=${SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS:-8}
export SGLANG_SIMULATE_ACC_LEN=${SGLANG_SIMULATE_ACC_LEN:-0} SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY=1 SGLANG_MINIMAX_M31_DRAFT_SPLITKV=1 SGLANG_HCU_ASYNC_SEQ_LENS_D2H=1
export SGLANG_DSPARK_FOLDED_PROPOSAL=0 SGLANG_DSPARK_USE_GREEDY_DRAFT=0
export SGLANG_MINIMAX_DECODE_SHARED_OVERLAP=${SGLANG_MINIMAX_DECODE_SHARED_OVERLAP:-0}
case "$SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS" in
    0|1|2|4|8) ;;
    *) echo "SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS must be 0, 1, 2, 4 or 8." >&2; exit 2 ;;
esac

default_max_requests=80
default_graph_max=10
if [[ $mode == hisparse ]]; then
    default_max_requests=128
    default_graph_max=16
fi
graph_max=${M31_GRAPH_MAX_LOCAL_BS:-$default_graph_max}
[[ $graph_max =~ ^([1-9]|1[0-6])$ ]] || { echo "Graph batch maximum must be 1 through 16." >&2; exit 2; }
mapfile -t graph_bs < <(seq 1 "$graph_max")
options=(
    --model-path "$model" --quantization slimquant_w4a8_marlin --trust-remote-code --random-seed 42
    --tp-size 8 --attn-dp-size 8 --ep-size 8 --moe-dense-tp-size 1 --enable-dp-lm-head
    --enable-dp-attention-local-control-broadcast --load-balance-method round_robin
    --moe-a2a-backend deepep --moe-runner-backend deep_gemm --deepep-mode auto
    --deepep-dispatcher-output-dtype int8 --disable-shared-experts-fusion
    --attention-backend triton --prefill-attention-backend triton --decode-attention-backend fa3
    --kv-cache-dtype fp8_e4m3 --page-size 128 --cuda-graph-backend-prefill disabled
    --cuda-graph-backend-decode full --cuda-graph-bs-decode "${graph_bs[@]}"
    --speculative-algorithm DSPARK --speculative-draft-model-path "$draft"
    --speculative-draft-model-quantization compressed-tensors --speculative-draft-attention-backend triton
    --speculative-dspark-block-size 7 --speculative-num-draft-tokens 8
    --context-length "${M31_DECODE_CONTEXT_LENGTH:-163840}" --chunked-prefill-size 8192
    --max-running-requests "${M31_MAX_RUNNING_REQUESTS:-$default_max_requests}" --mem-fraction-static "${M31_MEM_FRACTION:-0.95}"
    --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
    --dist-init-addr "$master:${DIST_PORT:-5001}" --nnodes 2 --node-rank "$rank"
    --host 0.0.0.0 --port "${SERVER_PORT:-11011}"
)
if [[ $mode == calibrate ]]; then
    export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=${M31_EPLB_DUMP_DIR:-$PWD/results/dp8-eplb-layout}
    options+=(--init-expert-location trivial --expert-distribution-recorder-mode stat
        --expert-distribution-recorder-buffer-size 4096)
else
    options+=(--enable-eplb --init-expert-location "$layout" --eplb-algorithm deepseek
        --eplb-rebalance-num-iterations 1000000 --expert-distribution-recorder-buffer-size 128
        --eplb-rebalance-layers-per-chunk 4)
fi

if [[ $mode != rsc ]]; then
    options+=(--disaggregation-mode decode --disaggregation-transfer-backend fake --disable-radix-cache)
fi
if [[ $mode == hisparse ]]; then
    export HCU_PLUGIN_MINIMAX_M31_HISPARSE=1 M31_HISPARSE_FAKE_VALIDATE=1
    export M31_HISPARSE_REUSE_VERIFY_METADATA=1 M31_HISPARSE_FUSED_QK_CACHE=1
    export M31_HISPARSE_FAST_PLAN=1 M31_HISPARSE_PIN_SLOT_INDICES=1 M31_HISPARSE_DIRECT_VERIFY_INDICES=1
    export M31_HISPARSE_STABLE_SLOTS=1 M31_HISPARSE_HIP_STABLE_PLAN=1 M31_HISPARSE_HEADWISE_SWAP=1
    export M31_HISPARSE_AB_COMPACT_HOT=1 M31_HISPARSE_VERIFY_QUERY_CHUNK=8 M31_HISPARSE_TILED_SWAP=256,8
    export M31_HISPARSE_BATCH_COMMIT=0 SGLANG_HISPARSE_HCU_PIN_HOST_KV=0
    export SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=${SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB:-4}
    default_hisparse_config='{"device_buffer_size":32768,"host_to_device_ratio":2.29}'
    options+=(--enable-hisparse --hisparse-config "${M31_HISPARSE_CONFIG:-$default_hisparse_config}"
        --disaggregation-decode-host-receive-threshold 1)
fi

command=("$py" -m sglang.launch_server "${options[@]}" "$@")
if (( print_args )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit
fi

required_files=("$model/config.json" "$draft/config.json" "$HIPBLASLT_TUNING_OVERRIDE_FILE")
if [[ $mode == calibrate ]]; then
    mkdir -p -- "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
else
    required_files+=("$layout")
fi
for path in "${required_files[@]}"; do
    [[ -s $path && -r $path ]] || { echo "Missing or unreadable file: $path" >&2; exit 1; }
done

printf 'rank=%s mode=%s verify_split=%s simulate_acc_len=%s RDMA=%s\n' \
    "$rank" "$mode" "$SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS" "$SGLANG_SIMULATE_ACC_LEN" "$ROCSHMEM_ALLOWED_IBV_DEVICES"
exec "${command[@]}"

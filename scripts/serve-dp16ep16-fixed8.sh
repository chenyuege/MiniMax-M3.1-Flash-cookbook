#!/usr/bin/env bash

set -euo pipefail
if [[ ${1:-} == --help || $# -lt 2 ]]; then
  echo "Usage: bash $0 MASTER_IP NODE_RANK [--print-args | SGLang options]"
  [[ ${1:-} == --help ]] || exit 2
  exit
fi
MASTER_IP="${1:?Usage: $0 MASTER_IP NODE_RANK [options]}"
rank="${2:?Missing NODE_RANK}"
shift 2
print_args=0
if [[ ${1:-} == --print-args ]]; then
  print_args=1
  shift
fi
nodes=4
export M31_DP_SIZE=16
[[ $rank =~ ^[0-3]$ ]] || { echo "Invalid node rank: $rank" >&2; exit 2; }
DIST_PORT="${DIST_PORT:-5001}"
DIST_INIT_ADDR="$MASTER_IP:$DIST_PORT"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1,2,3}

export SGLANG_EPLB_AUTO_START_RECORDER=0
unset SGLANG_MINIMAX_EPLB_PREFILL_ONLY
export M31_SIMULATE_ACC_LEN=0
ROOT=${M31_FIXED8_RUNTIME_DIR:-/opt/minimax31-w4a8}
model=${M31_MODEL_PATH:-/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8}
draft=${M31_DSPARK_MODEL_PATH:-"$model/dspark-channel-int8-smooth-20261107"}
layout=${M31_EPLB_INIT_LOCATION:-$ROOT/layout/expert_distribution_recorder_1794400407.277366.pt}
phase=${M31_EPLB_PHASE:-serve}
export LIGHTOP_PER_TOKEN_INT8_METHOD="${LIGHTOP_PER_TOKEN_INT8_METHOD:-triton}"
export SGLANG_USE_LIGHTOP_OAI_INT8_QUANT="${SGLANG_USE_LIGHTOP_OAI_INT8_QUANT:-0}"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export GPU_ARCHS="${GPU_ARCHS:-gfx938}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-4}"

# DeepEP
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
# Discover the actual supernode RDMA devices; allow an explicit validated override.
if (( ! print_args )) && [[ -z ${ROCSHMEM_ALLOWED_IBV_DEVICES:-} ]]; then
  ib_devices=(/sys/class/infiniband/*)
  [[ -e ${ib_devices[0]} ]] || { echo 'No RDMA devices; set/check supernode networking' >&2; exit 1; }
  ROCSHMEM_ALLOWED_IBV_DEVICES=$(printf '%s\n' "${ib_devices[@]##*/}" | paste -sd, -)
fi
export ROCSHMEM_ALLOWED_IBV_DEVICES
[[ -n ${GLOO_SOCKET_IFNAME:-} ]] || unset GLOO_SOCKET_IFNAME
[[ -n ${NCCL_SOCKET_IFNAME:-} ]] || unset NCCL_SOCKET_IFNAME
export ROCSHMEM_GDR_DISABLE_HDP=1

export SGLANG_HCU_M3_NATIVE_DEQUANT=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_KV_LAYOUT_HCU_FA=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_OAI_FP8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_AITER=0
export SGLANG_ROCM_USE_AITER_MOE=0
export AITER_AR_TRANSPORT=ipc
export SGLANG_DP_USE_MAX_LEN=0
export SGLANG_USE_AITER_AG=0
export SGLANG_ROCM_USE_AITER_MOE_WITH_SHUFFLE=0

export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export SGLANG_OPT_DG_HCU_FUSED_QUANT_SCATTER=0

# Shared projections in DeepEP low-latency dispatch/combine receive windows.
export SGLANG_MINIMAX_DECODE_SHARED_OVERLAP="${SGLANG_MINIMAX_DECODE_SHARED_OVERLAP:-1}"

export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_MINIMAX_ROUTER_GEMV_BLOCK_K=1024
export SGLANG_MINIMAX_ROUTER_GEMV_NUM_WARPS=1
export SGLANG_ENABLE_M3_TRITON_FP8_ATTN_GEMM=0

# FlashMLA: native FP8 indexer for prefill/decode/verify.
export SGLANG_MINIMAX_INDEX_FP8=dot
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_INDEXER=1
export SGLANG_OPT_USE_MINIMAX_FLASH_MLA_GFX938_DECODE=1

# DSpark
export SGLANG_DSPARK_FOLDED_PROPOSAL="${SGLANG_DSPARK_FOLDED_PROPOSAL:-0}"
export SGLANG_DSPARK_USE_GREEDY_DRAFT=0
export SGLANG_RAGGED_VERIFY_MODE=static
export SGLANG_SIMULATE_ACC_LEN="${M31_SIMULATE_ACC_LEN:-0}"
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_LOG_DECODE_GRAPH_KEY=0

today=$(date +%Y%m%d)
timestamp=$(date +%H%M%S)
log_dir="$ROOT/logs/$today"
log_file="${M31_LOG_FILE:-$log_dir/MiniMax-M3.1-int8-w4a8_$(hostname -s)_${timestamp}.log}"

# BLAS tuning
# Validated W4A8/W8A8 table; M31 override supports an explicit comparison.
export HIPBLASLT_TUNING_OVERRIDE_FILE="${M31_HIPBLASLT_CONFIG:-$ROOT/config_file1110/hipblaslt.decode.accepted.config}"
# Validated W4A8/W8A8 table; M31 override supports an explicit comparison.
# The decode-only override fails the BF16 router GEMM used in RSC prewarm.
if [[ -n ${M31_ROCBLAS_LIBRARY:-} ]]; then
  export ROCBLAS_TENSILE_LIBPATH="$M31_ROCBLAS_LIBRARY"
else
  unset ROCBLAS_TENSILE_LIBPATH
fi
export SGLANG_MINIMAX_M31_DRAFT_SPLITKV="${SGLANG_MINIMAX_M31_DRAFT_SPLITKV:-1}"
export SGLANG_MINIMAX_FLASH_MLA_PREFILL_LIVE_EXTENT=1
export M31_STAGE1_GRAPH_MAX_KV_LEN=148608 #for a compact verify graph plus full-context fallback.
export MSA_STAGE1_KERNEL=v8
export FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
# Keep native FP8 target verification on split=8 for the validated DP16 workload.
# Set 0 before service startup to restore the wheel's automatic selection.
export SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS="${SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS:-8}"
case "$SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS" in
  0|1|2|4|8) ;;
  *) echo "SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS must be 0, 1, 2, 4 or 8" >&2; exit 2 ;;
esac
unset M31_DEBUG_VERIFY_FORCE_SPLITS M31_DEBUG_VERIFY_IMPL

unset MSA_STAGE1_BATCH_PAIR SGLANG_MINIMAX_FLASH_MLA_FP8_NUM_SPLITS
unset FLASH_MLA_BF16_MSA_FORCE_SPLITS FORCE_AITER_MOE_ASM SGLANG_FORCE_AITER_MOE_ASM

# W4A8: INT4 expert storage, dynamic INT8 activation, HIPC grouped GEMM.
export SGLANG_W4A8_TPMOE_BACKEND=lightop
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_GROUPGEMM=1
export W8A8_SUPPORT_METHODS=3
# Slice unused Q rows only for the validated channel-INT8 DSpark path.
export SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY="${SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY:-1}"
if [[ -n "${SGLANG_MINIMAX_MOE_MODE:-}" ]]; then
  echo "This INT4/INT8 candidate requires DeepEP (unset SGLANG_MINIMAX_MOE_MODE)." >&2
  exit 2
fi

export DEEP_EP_NORMAL_MNVL=1 ROCSHMEM_IPC_MNVL=1 HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1 ROCSHMEM_DISABLE_HDP_FLUSH=1
moe_options=(--moe-a2a-backend deepep --deepep-mode auto
  --deepep-dispatcher-output-dtype int8)

case "$phase" in
  dump)
    export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${M31_EPLB_DUMP_DIR:?Missing dump directory}"
    eplb_options=(--expert-distribution-recorder-mode stat
      --expert-distribution-recorder-buffer-size 4096)
    ;;
  serve)
    eplb_options=(--enable-eplb --init-expert-location "$layout"
      --eplb-algorithm deepseek --eplb-rebalance-num-iterations 1000000
      --expert-distribution-recorder-buffer-size 128 --eplb-rebalance-layers-per-chunk 4)
    ;;
  *) echo "Invalid EPLB phase: $phase" >&2; exit 2 ;;
esac
command=(sglang serve
  --model-path "$model"
  --quantization slimquant_w4a8_marlin
  --trust-remote-code
  --random-seed 42
  --speculative-algorithm DSPARK
  --speculative-draft-model-path "$draft"
  --speculative-draft-model-quantization compressed-tensors
  --speculative-draft-attention-backend triton
  --speculative-dspark-block-size 7
  --speculative-num-draft-tokens 8
  --tp-size 16
  --moe-dense-tp-size 1
  --dp-size 16
  --ep-size 16
  --enable-dp-attention
  --enable-dp-lm-head
  --enable-dp-attention-local-control-broadcast
  --load-balance-method round_robin
  "${moe_options[@]}" "${eplb_options[@]}"
  --moe-runner-backend deep_gemm
  --disable-shared-experts-fusion
  --attention-backend triton
  --prefill-attention-backend triton
  --decode-attention-backend fa3
  --kv-cache-dtype fp8_e4m3
  --page-size 128
  --cuda-graph-backend-decode full
  --cuda-graph-backend-prefill disabled
  --custom-all-reduce-backend aiter
  --context-length "${M31_DECODE_CONTEXT_LENGTH:-163840}"
  --chunked-prefill-size 8192
  --max-running-requests "${M31_MAX_RUNNING_REQUESTS:-160}"
  --mem-fraction-static 0.95
  --cuda-graph-bs-decode 1 2 4 5 6 8 9 10
  --reasoning-parser minimax-m3
  --tool-call-parser minimax-m3
  --dist-init-addr "$DIST_INIT_ADDR" --nnodes "$nodes" --node-rank "$rank"
  --host 0.0.0.0
  --port "${SERVER_PORT:-11011}"
)
if (( print_args )); then
  printf '%q ' "${command[@]}" "$@"
  printf '\n'
  exit 0
fi
cd "${M31_WORKDIR:-$ROOT}"
for path in "$model/config.json" "$draft/config.json"; do
  [[ -r $path ]] || { echo "Missing checkpoint config: $path" >&2; exit 1; }
done
if [[ $phase == serve && ! -s $layout ]]; then
  echo "Missing EPLB layout: $layout" >&2
  exit 1
fi
for path in "$HIPBLASLT_TUNING_OVERRIDE_FILE" ${ROCBLAS_TENSILE_LIBPATH:+"$ROCBLAS_TENSILE_LIBPATH"}; do
  [[ -s $path && -r $path ]] || { echo "Tuning path not readable: $path" >&2; exit 1; }
done
mkdir -p "$(dirname -- "$log_file")"
python3 - <<'CHECK_VERIFY_SPLITS'
from sglang.srt.environ import envs
assert hasattr(envs, "SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS"), (
    "Apply native-fp8-verify-split-fix.patch to the active SGLang checkout."
)
print("Native FP8 target verify split-KV =", envs.SGLANG_MINIMAX_FLASH_MLA_FP8_VERIFY_SPLITS.get())
CHECK_VERIFY_SPLITS
echo "DP=$M31_DP_SIZE EP=$M31_DP_SIZE nodes=$nodes rank=$rank master=$DIST_INIT_ADDR HIP=$HIP_VISIBLE_DEVICES RDMA=$ROCSHMEM_ALLOWED_IBV_DEVICES phase=$phase moe_a2a_backend=deepep"
[[ ${1:-} != --preflight ]] || exit 0
"${command[@]}" "$@" 2>&1 | tee "$log_file"

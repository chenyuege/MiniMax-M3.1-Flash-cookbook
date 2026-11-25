#!/usr/bin/env bash
set -euo pipefail

stage=${1:-}
case "$stage" in
  prefill|decode) shift ;;
  *) echo "Usage: bash scripts/serve.sh {prefill|decode} [--print-args]" >&2; exit 2 ;;
esac
case "${1:-}" in
  ""|--print-args) ;;
  *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo "Expected at most one option after the worker role." >&2
  exit 2
fi

profile=${M31_HARDWARE_PROFILE:-bw1100-96gb}
nnodes=${M31_NNODES:-1}
rank=${M31_NODE_RANK:-0}
devices=0,1,2,3,4,5,6,7
dist_opts=()
case "$profile" in
  bw1100-96gb)
    prefill_mem=0.87
    decode_mem=0.92
    decode_requests=24
    decode_chunk=8192
    decode_graph_sizes=(1 2 3)
    hisparse_default=1
    layered=1
    ;;
  bw1300-144gb)
    prefill_mem=0.90
    decode_mem=0.95
    decode_requests=64
    decode_chunk=1024
    decode_graph_sizes=(1 2 3 4 5 6 7 8)
    hisparse_default=0
    layered=0
    ;;
  *) echo "M31_HARDWARE_PROFILE must be bw1100-96gb or bw1300-144gb." >&2; exit 2 ;;
esac
case "$nnodes:$rank" in
  1:0) ;;
  2:0|2:1)
    [[ $profile == bw1300-144gb ]] || { echo "Two hosts per worker require bw1300-144gb." >&2; exit 2; }
    : "${M31_DIST_INIT_ADDR:?Set the rendezvous address of rank 0 for this worker}"
    devices=0,1,2,3
    dist_opts=(--nnodes "$nnodes" --node-rank "$rank" --dist-init-addr "$M31_DIST_INIT_ADDR")
    ;;
  *) echo "Use M31_NNODES=1 with rank 0, or M31_NNODES=2 with rank 0 or 1." >&2; exit 2 ;;
esac

model=${M31_MODEL_PATH:-/model/MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8}
draft=${M31_DSPARK_MODEL_PATH:-"$model/dspark-channel-int8-smooth-20261107"}
: "${M31_PD_IB_DEVICES:?Set the Mooncake IB devices for this node}"
: "${ROCSHMEM_ALLOWED_IBV_DEVICES:?Set the DeepEP IB devices for this node}"

export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HCU_ADAPTER_HOST=sglang SGLANG_PLUGINS=hcu_minimax_m31 HCU_SLIMQUANT=0
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1 HCU_PLUGIN_MINIMAX_M31_DEEPEP=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$devices}
export GPU_ARCHS=gfx938 ROCSHMEM_GDR_DISABLE_HDP=1 ROCSHMEM_ALLOWED_IBV_DEVICES
export SGLANG_USE_INT4_W4A8=1 SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1 SGLANG_USE_W4A8_MASKED_HIPC=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_DEEPEP_USE_MOE_EP_GROUP=1 AITER_AR_TRANSPORT=ipc HCU_AITER_AR=1
export HCU_DEEPEP_BRIDGE=0 SGLANG_GROUPGEMM=1 SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_USE_LIGHTOP=1 SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1 W8A8_SUPPORT_METHODS=3
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1 SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=1
export SGLANG_MINIMAX_INDEX_FP8=dot SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export MSA_STAGE1_KERNEL=v8 FLASH_MLA_MSA128_FP8_SPLIT_POLICY=verify8
export SGLANG_PROFILE_V2=0 SGLANG_SIMULATE_ACC_LEN=0
export SGLANG_EPLB_AUTO_START_RECORDER=0 SGLANG_MINIMAX_DSPARK_INT8_KV_ONLY=1

opts=(
  --model-path "$model"
  --served-model-name MiniMax-M3.1-Flash-Channel-W4A8-INT4-W8A8-INT8
  --quantization slimquant_w4a8_marlin --trust-remote-code --random-seed 42
  --tp-size 8 --pp-size 1 --ep-size 8 --moe-dense-tp-size 1
  "${dist_opts[@]}"
  --moe-a2a-backend deepep --moe-runner-backend deep_gemm
  --deepep-dispatcher-output-dtype int8 --disable-shared-experts-fusion
  --load-balance-method round_robin --kv-cache-dtype fp8_e4m3 --page-size 128
  --speculative-algorithm DSPARK --speculative-draft-model-path "$draft"
  --speculative-draft-model-quantization compressed-tensors
  --speculative-draft-attention-backend triton
  --speculative-dspark-block-size 7 --speculative-num-draft-tokens 8
  --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
  --cuda-graph-backend-prefill disabled --disable-overlap-schedule
  --disaggregation-mode "$stage" --disaggregation-transfer-backend mooncake
  --disaggregation-ib-device "$M31_PD_IB_DEVICES"
  --host "${M31_HOST:-0.0.0.0}" --port "${M31_PORT:-30000}"
)
if [[ ${M31_SKIP_STARTUP_WARMUP:-0} == 1 ]]; then
  opts+=(--skip-server-warmup)
fi

if [[ $stage == prefill ]]; then
  export HCU_PREFILL_CP=1 LIGHTOP_PER_TOKEN_INT8_METHOD=native
  if [[ $nnodes == 2 ]]; then
    export DEEP_EP_NORMAL_MNVL=1 ROCSHMEM_IPC_MNVL=1
  fi
  prefill_chunk=${M31_PREFILL_CHUNK:-16384}
  export SGLANG_HCU_DEEPEP_NO_M_INDICES_SYNC=1
  opts+=(
    --attn-dp-size 1 --attn-cp-size 8 --enable-prefill-cp --cp-strategy zigzag
    --attention-backend fa3 --cuda-graph-backend-decode disabled
    --deepep-mode normal --context-length 1048576
    --chunked-prefill-size "$prefill_chunk" --max-prefill-tokens "$prefill_chunk"
    --mem-fraction-static "$prefill_mem" --max-running-requests 80
    --disaggregation-bootstrap-port "${M31_BOOTSTRAP_PORT:-8998}"
  )
  if [[ -n ${M31_DEEPEP_CONFIG:-} ]]; then
    test -r "$M31_DEEPEP_CONFIG"
    opts+=(--deepep-config "$M31_DEEPEP_CONFIG")
  fi
  if [[ -n ${M31_PREFILL_MAX_TOTAL_TOKENS:-} ]]; then
    opts+=(--max-total-tokens "$M31_PREFILL_MAX_TOTAL_TOKENS")
  fi
else
  export HCU_PREFILL_CP=0 LIGHTOP_PER_TOKEN_INT8_METHOD=triton
  export GPU_MAX_HW_QUEUES=4 DEEPEP_ENABLE_LL_LAYERED_OPT=$layered
  export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
  export SGLANG_RAGGED_VERIFY_MODE=static SGLANG_DSPARK_FOLDED_PROPOSAL=0
  export SGLANG_DSPARK_USE_GREEDY_DRAFT=0 M31_STAGE1_GRAPH_MAX_KV_LEN=148608
  hisparse=${M31_ENABLE_HISPARSE:-$hisparse_default}
  case "$hisparse" in
    0|1) export HCU_PLUGIN_MINIMAX_M31_HISPARSE=$hisparse ;;
    *) echo "M31_ENABLE_HISPARSE must be 0 or 1." >&2; exit 2 ;;
  esac
  graph=${M31_DECODE_GRAPH:-full}
  case "$graph" in
    full) opts+=(--cuda-graph-bs-decode "${decode_graph_sizes[@]}") ;;
    disabled) ;;
    *) echo "M31_DECODE_GRAPH must be full or disabled." >&2; exit 2 ;;
  esac
  opts+=(
    --attn-dp-size 8 --enable-dp-lm-head --enable-dp-attention-local-control-broadcast
    --attention-backend triton --prefill-attention-backend triton --decode-attention-backend fa3
    --deepep-mode auto --cuda-graph-backend-decode "$graph"
    --context-length 163840 --chunked-prefill-size "$decode_chunk"
    --mem-fraction-static "$decode_mem" --max-running-requests "$decode_requests"
    --disable-radix-cache
  )
  if [[ $hisparse == 1 ]]; then
    export SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=${SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB:-4}
    opts+=(
      --enable-hisparse --disaggregation-decode-host-receive-threshold 1
      --hisparse-config '{"device_buffer_size":65536,"host_to_device_ratio":1}'
    )
  fi
fi

if [[ -n ${M31_EPLB_LAYOUT:-} && ( $stage == prefill || $profile == bw1300-144gb ) ]]; then
  test -r "$M31_EPLB_LAYOUT"
  opts+=(
    --enable-eplb --init-expert-location "$M31_EPLB_LAYOUT" --eplb-algorithm deepseek
    --eplb-rebalance-num-iterations 1000000 --eplb-rebalance-layers-per-chunk 4
    --expert-distribution-recorder-mode stat --expert-distribution-recorder-buffer-size 16
  )
fi

if [[ ${1:-} == --print-args ]]; then
  printf '%q ' sglang serve "${opts[@]}"
  printf '\n'
  exit
fi
for checkpoint in "$model" "$draft"; do
  if [[ ! -r "$checkpoint/config.json" ]]; then
    echo "Missing checkpoint config: $checkpoint/config.json" >&2
    exit 1
  fi
done
exec sglang serve "${opts[@]}"

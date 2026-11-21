#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help || $# -lt 2 ]]; then
    echo "Usage: bash $0 MASTER_IP NODE_RANK [--print-args | SGLang options]"
    [[ ${1:-} == --help ]] || exit 2
    exit 0
fi
master=${1:?dist-init host}
rank=${2:?node rank}
shift 2
[[ $rank =~ ^[01]$ ]] || { echo "NODE_RANK must be 0 or 1." >&2; exit 2; }
print_args=0
if [[ ${1:-} == --print-args ]]; then
    print_args=1
    shift
fi

if (( ! print_args )) && [[ -r /opt/dtk/env.sh ]]; then
    set +u
    source /opt/dtk/env.sh
    set -u
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=${M31_RUNTIME_DIR:-/opt/minimax-cp8ep8}
py=${M31_PYTHON:-python3}
site_dir=${M31_SITE_DIR:-$runtime/site}
adapter_dir=${M31_ADAPTER_DIR:-$runtime/adapter}
assets=${M31_ASSETS_DIR:-$runtime/config}
model=${M31_MODEL_PATH:-/model/main-w4a8}
layout=${M31_EPLB_INIT_LOCATION:-$runtime/layout/m31-cp8ep8-from-dp16-logical-count.pt}
deepep_config=${M31_DEEPEP_CONFIG:-$script_dir/../configs/cp8ep8-prefill-deepep.json}

export PYTHONPATH="$adapter_dir:$site_dir"
export HCU_DIST_BROADCAST_COMPRESS=1
export SGLANG_MINIMAX_CP_PACK_PREFILL_SEGMENTS=0
export PYTHONDONTWRITEBYTECODE=1
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export HCU_ADAPTER_HOST=sglang
export SGLANG_PLUGINS=hcu_minimax_m31
export HCU_PREFILL_CP=1
export SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD=split
export SGLANG_USE_LIGHTOP_MINIMAX_QK_ROPE=0
export HCU_PLUGIN_MINIMAX_M31_FLASHMLA=1
export HCU_PLUGIN_MINIMAX_M31_DEEPEP=1
export HCU_DEEPEP_BRIDGE=0
export DEEP_EP_NORMAL_MNVL=1
export HCU_AITER_AR=1
export PYTHONNOUSERSITE=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export GPU_ARCHS=gfx938

xdp=/sys/module/hycu/parameters/xdp_size
unset ROCSHMEM_GDR_DISABLE_HDP
if [[ -r "$xdp" ]] && read -r xdp_size < "$xdp" && [[ "$xdp_size" =~ ^[0-9]+$ ]] && (( xdp_size > 0 )); then
  export ROCSHMEM_GDR_DISABLE_HDP=1
fi

# FlashMLA and CP.
export MSA128_FP8_PREFILL_KERNEL=v2
export SGLANG_MINIMAX_INDEX_FP8=dot
export SGLANG_MINIMAX_CACHE_PREFILL_INDEX_METADATA=1
export FLASH_MLA_STAGE1_PREFILL_ASYNC_H2D=1
export SGLANG_KV_LAYOUT_HCU_FA=0
export HCU_MINIMAX_CP_PACK_FP8=1

# LightOp and W4A8 MoE.
export LIGHTOP_PER_TOKEN_INT8_METHOD=native
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_OAI_INT8_QUANT=1
export SGLANG_USE_LIGHTOP_CHANNEL_FP8=1
export SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_INT4_W4A8=1
export SGLANG_W4A8_TPMOE_BACKEND=lightop
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0
export SGLANG_USE_W4A8_CONTIGUOUS_HIPC=1
export SGLANG_GROUPGEMM=1
export W8A8_SUPPORT_METHODS=3
export SGLANG_USE_DEEPGEMM_MOE=0
export SGLANG_OPT_USE_MINIMAX_ROUTER_GEMV=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_USE_AITER_AG=0
export SGLANG_MINIMAX_PREFILL_SHARED_OAI_INT8=1

export HIPBLASLT_TUNING_OVERRIDE_FILE="${M31_HIPBLASLT_CONFIG:-$assets/hipblaslt.prefill.accepted.config}"
export ROCBLAS_TENSILE_LIBPATH="${M31_ROCBLAS_TENSILE_LIBPATH:-$assets/w4a8-w8a8/tuning_20261105/rocblas/prefill/candidate_library_gpu6}"

options=(
    --model-path "$model" --quantization slimquant_w4a8_marlin --trust-remote-code --random-seed 42
    --nnodes 2 --node-rank "$rank" --dist-init-addr "$master:${DIST_PORT:-13433}"
    --tp-size 8 --pp-size 1 --attn-dp-size 1 --ep-size 8 --moe-dense-tp-size 1
    --attn-cp-size 8 --enable-prefill-cp --cp-strategy zigzag --disable-overlap-schedule
    --load-balance-method round_robin
    --moe-a2a-backend deepep --deepep-mode normal
    --deepep-config "$deepep_config"
    --deepep-dispatcher-output-dtype int8 --moe-runner-backend deep_gemm --disable-shared-experts-fusion
    --init-expert-location "$layout" --ep-num-redundant-experts 8
    --ep-dispatch-algorithm static --ep-static-dispatch-policy locality_fair
    --attention-backend fa3 --prefill-attention-backend fa3 --decode-attention-backend fa3
    --kv-cache-dtype fp8_e4m3 --page-size 128
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
    --context-length 1048576 --chunked-prefill-size 65536 --max-prefill-tokens 65536
    --max-running-requests 80 --mem-fraction-static 0.9
    --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
    --host 0.0.0.0 --port "${SERVER_PORT:-11010}"
)

command=("$py" -m hcu_sglang "${options[@]}" "$@")
if (( print_args )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

for file in "$model/config.json" "$layout" "$deepep_config" "$HIPBLASLT_TUNING_OVERRIDE_FILE"; do
    [[ -f $file && -s $file && -r $file ]] || { echo "Missing, empty or unreadable file: $file" >&2; exit 1; }
done
for dir in "$site_dir" "$adapter_dir" "$ROCBLAS_TENSILE_LIBPATH"; do
    [[ -d $dir && -r $dir ]] || { echo "Missing or unreadable directory: $dir" >&2; exit 1; }
done
printf 'rank=%s layout=%s deep_ep_mnvl=%s packed_fp8=%s broadcast_compress=%s shared_overlap=%s flashmla_pack_segments=%s\n' \
    "$rank" "$layout" "$DEEP_EP_NORMAL_MNVL" "$HCU_MINIMAX_CP_PACK_FP8" \
    "$HCU_DIST_BROADCAST_COMPRESS" "$SGLANG_MINIMAX_PREFILL_SHARED_OVERLAP_METHOD" \
    "$SGLANG_MINIMAX_CP_PACK_PREFILL_SEGMENTS"
exec "${command[@]}"

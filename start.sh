#!/usr/bin/env bash
# =============================================================================
# DeepSeek V4 Flash – Full Cluster Launch
#
# Orchestrates everything on a single machine: pulls docker image, downloads
# model, syncs both to the worker via passwordless SSH, then launches vLLM
# on both DGX Spark nodes (rank 0 = master, rank 1 = worker).
#
# Usage:
#   ./start.sh
#
# Prerequisites:
#   - passwordless SSH to $WORKER_ADDR
#   - huggingface-cli installed and logged in
#   - docker with GPU support on both nodes
#   - ~/gpu-clear.sh script on both nodes (optional, failures ignored)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source .env (auto-export so docker -e sees them)
set -a; source "$SCRIPT_DIR/.env"; set +a

# --- Validate required configuration ----------------------------------------
: "${MASTER:?MASTER not set in .env}"
: "${WORKER_ADDR:?WORKER_ADDR not set in .env}"
: "${PORT:?PORT not set in .env}"
: "${IF:?IF not set in .env}"
: "${HCA:?HCA not set in .env}"

# --- Settings ---------------------------------------------------------------
IMG_SRC="ghcr.io/drowzeys/vllm-dspark-nvfp4-stage-c:gb10"
IMG_DST="vllm-dspark-runtime:dspark-nvfp4-stage-c"
MODELDIR="${MODELDIR:-$HOME/models/dsv4-flash-dspark-abliterated}"
SERVE_PORT="${SERVE_PORT:-8888}"
CNAME="dsv4_ablit_srv"
# HF model repo
HF_REPO="drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored"

echo "================================================"
echo " DeepSeek V4 Flash – Full Cluster Launch"
echo "================================================"
echo " Master:      $MASTER"
echo " Worker:      $WORKER_ADDR"
echo " Ctrl port:   $PORT"
echo " Serve port:  $SERVE_PORT"
echo " Model dir:   $MODELDIR"
echo " Image:       $IMG_SRC"
echo " IF:          $IF"
echo " HCA:         $HCA"
echo "================================================"

# =============================================================================
# Step 1 – Pull & tag docker image locally
# =============================================================================
echo ""
echo "[1/4] Pulling docker image..."
docker pull "$IMG_SRC"
docker tag "$IMG_SRC" "$IMG_DST"

# =============================================================================
# Step 2 – Sync docker image to worker (save → pipe → load)
# =============================================================================
echo ""
echo "[2/4] Syncing docker image to worker ($WORKER_ADDR)..."
docker save "$IMG_SRC" | ssh -o StrictHostKeyChecking=accept-new "$WORKER_ADDR" \
  "docker load && docker tag $IMG_SRC $IMG_DST"

# =============================================================================
# Step 3 – Download model
# =============================================================================
echo ""
echo "[3/4] Downloading model ($HF_REPO)..."
mkdir -p "$MODELDIR"
hf download "$HF_REPO" --local-dir "$MODELDIR"

# =============================================================================
# Step 4 – Sync model to worker (same path, same folder structure)
# =============================================================================
echo ""
echo "[4/4] Syncing model to worker ($WORKER_ADDR)..."
ssh -o StrictHostKeyChecking=accept-new "$WORKER_ADDR" "mkdir -p '$MODELDIR'"
rsync -av --delete "$MODELDIR/" "$WORKER_ADDR:$MODELDIR/"

# =============================================================================
# Launch containers
# =============================================================================

# --- Build the -lc script for docker's entrypoint ---------------------------
# The vLLM `serve` command with all flags. We build this as a bash variable so
# we can reuse it for both nodes, with rank-specific differences.
#
# Quoting strategy:
#   LC_SCRIPT is a single-quoted bash string. Arguments that need variable
#   interpolation use the '...'$VAR'...' pattern (close single-quote, expand,
#   reopen). The JSON-typed flags (--speculative-config etc.) are double-quoted
#   with escaped inner quotes so docker's bash resolves them correctly.
build_lc_script() {
  local rank="$1"
  local headless="$2"   # empty or "--headless"
  local serve_port="$3"
  local master_addr="$4"
  local master_port="$5"

  # Single-quoted string with inline variable interpolation
  cat <<'LC' | sed "s|%RANK%|$rank|g; s|%HEADLESS%|$headless|g; s|%PORT%|$serve_port|g; s|%MASTER%|$master_addr|g; s|%MPORT%|$master_port|g"
export PATH="/opt/env/bin:/opt/env/nvvm/bin:/opt/env/targets/sbsa-linux/nvvm/bin:${PATH:-}";
export CUDA_HOME="${CUDA_HOME:-/opt/env/targets/sbsa-linux}";
export LD_LIBRARY_PATH="/opt/env/lib:/opt/env/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}";
exec /opt/env/bin/vllm serve /model --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port %PORT% \
  --trust-remote-code --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla --block-size 256 \
  --max-model-len 262144 --max-num-seqs 4 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization 0.82 \
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5}" \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs "{\"thinking\":false}" \
  --generation-config vllm --override-generation-config "{\"temperature\":0.0,\"top_p\":1.0}" \
  --nnodes 2 --node-rank %RANK% --master-addr %MASTER% --master-port %MPORT% %HEADLESS%
LC
}

LC_SCRIPT=$(build_lc_script 1 "--headless" "$SERVE_PORT" "$MASTER" "$PORT")
LC_SCRIPT_MASTER=$(build_lc_script 0 "" "$SERVE_PORT" "$MASTER" "$PORT")

# --- Launch worker (rank 1) on remote node ----------------------------------
echo ""
echo "--- Launching rank 1 on $WORKER_ADDR ---"

# Pre-compute the worker's own IP for VLLM_HOST_IP
SELF_WORKER=$(ssh -o StrictHostKeyChecking=accept-new "$WORKER_ADDR" \
  "ip -4 addr show $IF 2>/dev/null | awk '/inet /{print \$2}' | cut -d/ -f1" || true)
SELF_WORKER="${SELF_WORKER:-$MASTER}"

ssh -o StrictHostKeyChecking=accept-new "$WORKER_ADDR" bash << REMOTE
set -euo pipefail

# Clear GPU state
bash "\$HOME/gpu-clear.sh" >/dev/null 2>&1 || true

# Clean any previous container
docker rm -f $CNAME 2>/dev/null || true

# Launch rank 1 container
docker run --gpus all -d --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "\$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$MODELDIR:/model:ro" \
  --name $CNAME \
  -e HF_HOME=/cache/huggingface -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP="$SELF_WORKER" \
  -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_NET=IB -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash $IMG_DST \
  -lc $(printf '%q' "$LC_SCRIPT")
REMOTE
echo "  rank 1 container launched on $WORKER_ADDR (container=$CNAME)"

# --- Launch master (rank 0) locally -----------------------------------------
echo ""
echo "--- Launching rank 0 locally ---"

# Clear GPU state
bash "$HOME/gpu-clear.sh" >/dev/null 2>&1 || true

# Compute our own IP for VLLM_HOST_IP
SELF=$(ip -4 addr show $IF 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
SELF=${SELF:-$MASTER}

# Clean any previous container
docker rm -f "$CNAME" 2>/dev/null || true

# Launch rank 0 container
docker run --gpus all -d --privileged --network host --ipc host --shm-size 64g \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  -v "$MODELDIR:/model:ro" \
  --name "$CNAME" \
  -e HF_HOME=/cache/huggingface -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_HOST_IP=$SELF \
  -e NCCL_SOCKET_IFNAME=$IF -e GLOO_SOCKET_IFNAME=$IF -e TP_SOCKET_IFNAME=$IF \
  -e NCCL_NET=IB -e NCCL_IB_HCA=$HCA -e NCCL_IB_DISABLE=0 -e NCCL_IB_GID_INDEX=3 -e NCCL_CROSS_NIC=1 \
  -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_DEBUG=WARN \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_WO_PROJECTION=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "$IMG_DST" \
  -lc "$LC_SCRIPT_MASTER"
echo "  rank 0 container launched locally (container=$CNAME)"

# =============================================================================
# Wait for API to become ready
# =============================================================================
echo ""
echo "--- Container logs (waiting for model to become ready) ---"

# Stream docker logs in the background
docker logs -f "$CNAME" 2>/dev/null &
LOG_PID=$!

# Poll health endpoint until it responds
while ! curl -sf "http://localhost:$SERVE_PORT/health" >/dev/null 2>&1; do
  sleep 2
done

# Model is ready — kill the log follower and print success
kill $LOG_PID 2>/dev/null || true
wait $LOG_PID 2>/dev/null || true

echo ""
echo "  Model is ready!"
echo "  http://$MASTER:$SERVE_PORT/v1/chat/completions"

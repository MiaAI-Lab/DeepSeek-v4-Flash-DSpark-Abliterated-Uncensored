#!/usr/bin/env bash
# =============================================================================
# DeepSeek V4 Flash – Cluster Stop
#
# Stops and removes the dsv4_ablit_srv container on both nodes.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
set -a; source "$SCRIPT_DIR/.env"; set +a

: "${WORKER_ADDR:?WORKER_ADDR not set in .env}"

CNAME="dsv4_ablit_srv"

echo "=== Stopping container on worker ($WORKER_ADDR) ==="
ssh -o StrictHostKeyChecking=accept-new "$WORKER_ADDR" \
  "docker stop $CNAME 2>/dev/null; docker rm $CNAME 2>/dev/null; echo '  done'" || true

echo "=== Stopping container locally ==="
docker stop "$CNAME" 2>/dev/null || true
docker rm "$CNAME" 2>/dev/null || true

echo "=== Both nodes cleaned up ==="

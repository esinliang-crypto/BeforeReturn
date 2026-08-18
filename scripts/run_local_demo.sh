#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting BeforeReturn API at http://127.0.0.1:8000"
(
  cd "$ROOT_DIR"
  /opt/anaconda3/envs/before-return/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
) &
API_PID=$!

cleanup() {
  kill "$API_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Starting BeforeReturn web app at http://localhost:3000"
cd "$ROOT_DIR/web"
npm run dev


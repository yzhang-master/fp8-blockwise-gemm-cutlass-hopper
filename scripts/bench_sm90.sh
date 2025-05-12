#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUTLASS_DIR="${CUTLASS_DIR:-}" "$ROOT/scripts/build_sm90.sh"
"$ROOT/build/sm90/fp8_blockwise_sm90" --m=1024 --n=1024 --k=1024 --warmup=10 --iterations=50 "$@"

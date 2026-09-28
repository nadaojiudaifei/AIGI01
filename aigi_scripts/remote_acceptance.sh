#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -lt 2 ]]; then echo "Usage: bash $0 CONFIG OUTPUT [--execute]" >&2; exit 2; fi
config=$1; output=$2; shift 2
python -m aigi acceptance --config "$config" --output "$output" "$@"

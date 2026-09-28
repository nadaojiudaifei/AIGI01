#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -lt 1 ]]; then echo "Usage: bash $0 PLAN [--execute]" >&2; exit 2; fi
plan=$1; shift
python -m aigi run --plan "$plan" "$@"

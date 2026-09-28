#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -lt 2 ]]; then echo "Usage: bash $0 NGPUS CONFIG [additional train arguments]" >&2; exit 2; fi
gpus=$1; config=$2; shift 2
torchrun --standalone --nproc_per_node="$gpus" -m aigi train --config "$config" "$@"

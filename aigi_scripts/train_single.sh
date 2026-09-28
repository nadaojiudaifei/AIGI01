#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ $# -lt 1 ]]; then echo "Usage: bash $0 CONFIG [additional train arguments]" >&2; exit 2; fi
config=$1; shift
python -m aigi train --config "$config" "$@"

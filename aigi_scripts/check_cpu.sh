#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -c aigi_pytest.ini aigi_tests -q
python -m aigi --help >/dev/null
python -m aigi catalog >/dev/null
for script in aigi_scripts/*.sh; do bash -n "$script"; done

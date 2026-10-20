#!/usr/bin/env bash
# 旧コマンドの互換入口。実験条件は smoke.yaml に集約する。
set -euo pipefail
device=cpu
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != --device || ( "$2" != cpu && "$2" != cuda ) ]]; then
    printf '%s\n' '使用法: bash tools/run-local-smoke.sh [--device cpu|cuda]' >&2
    exit 2
  fi
  device=$2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$root/tools/run-experiment.sh" --config "$root/configs/experiments/smoke.yaml" --device "$device"

#!/usr/bin/env bash
# 固有出力先を確保し、失敗した実行のログと終了値も残す。
set -euo pipefail
device=cpu
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != --device || ( "$2" != cpu && "$2" != cuda ) ]]; then
    printf '%s\n' '使用法: bash tools/run-local-smoke.sh [--device cpu|cuda]' >&2
    exit 2
  fi
  device=$2
fi
flake=path:.
if [[ "$device" == cuda ]]; then
  flake=path:.#gpu
fi
root=$(git rev-parse --show-toplevel)
if [[ "$PWD" != "$root" ]]; then
  printf '%s\n' 'リポジトリのルートで実行してください。' >&2
  exit 2
fi
mkdir -p "$root/.local-smoke"
run=$(mktemp -d "$root/.local-smoke/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
printf '%s\n' "nix develop $flake --command timeout 180s python tools/local_smoke.py --device $device --output $run/artifacts" > "$run/command.log"
set +e
nix develop "$flake" --command timeout 180s python tools/local_smoke.py --device "$device" --output "$run/artifacts" 2>&1 | tee "$run/console.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$run/exit-status"
printf '出力: %s\n終了値: %s\n' "$run" "$status"
exit "$status"

#!/usr/bin/env bash
# Nix 起動失敗とログ保存失敗も固有出力に記録する。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
device=cpu
check=false
args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      [[ $# -ge 2 && ( "$2" == cpu || "$2" == cuda ) ]] || { echo 'device は cpu|cuda が必要です' >&2; exit 2; }
      device=$2; shift 2 ;;
    --config|--timeout)
      [[ $# -ge 2 ]] || { echo 'オプション値が必要です' >&2; exit 2; }
      shift 2 ;;
    --check|--dry-run) check=true; shift ;;
    --help|-h) check=true; shift ;;
    --*) echo "未対応オプション: $1" >&2; exit 2 ;;
    *) break ;;
  esac
done
flake=.
[[ "$device" != cuda ]] || flake=.#gpu
if "$check"; then
  exec nix develop "$flake" --command python "$root/tools/run_experiment.py" "${args[@]}"
fi
mkdir -p "$root/.local-experiments"
run=$(mktemp -d "$root/.local-experiments/launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")
# 未検証の値や環境変数は記録しない。明示引数は検証後 command.json に残す。
printf 'nix develop %s --command python tools/run_experiment.py（検証済み引数は experiment/command.json）\n' "$flake" > "$run/command.log"
set +e
nix develop "$flake" --command python "$root/tools/run_experiment.py" --output "$run/experiment" "${args[@]}" 2>&1 | tee "$run/launcher.log"
statuses=("${PIPESTATUS[@]}")
set -e
status=${statuses[0]}
[[ ${statuses[1]} -eq 0 ]] || status=${statuses[1]}
printf '%s\n' "${statuses[*]}" > "$run/pipeline-status"
printf '%s\n' "$status" > "$run/exit-status"
printf '起動ログ: %s\n終了値: %s\n' "$run" "$status"
exit "$status"

#!/usr/bin/env bash
# 并行构建 grid_search 索引：每个配置独立 Python 进程（避免 fork+zvec 死锁）。
#
# 用法:
#   bash run_parallel_build.sh --datasets sift --jobs 4
#   bash run_parallel_build.sh --datasets sift,gist --jobs 4 --force
#
# nohup:
#   nohup bash run_parallel_build.sh --datasets sift --jobs 4 \
#     > output/grid_search/build_sift.log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
GRID="${SCRIPT_DIR}/grid_search_best_config.py"
DATASETS="sift"
INDEX_TYPES="hnsw,vamana"
JOBS=4
FORCE=""
DATA_DIR="/root/zvec_workspace/output"
OUTPUT_ROOT="${SCRIPT_DIR}/output/grid_search"
OPTIMIZE_THREADS=8
BATCH_SIZE=1024
MAX_BUILDS=0

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \?//'
  echo ""
  echo "Options:"
  echo "  --datasets LIST       sift,gist (default: sift)"
  echo "  --index-types LIST    hnsw,vamana (default: both)"
  echo "  --jobs N              max concurrent builds (default: 4)"
  echo "  --optimize-threads N  per-process zvec optimize threads (default: 8)"
  echo "  --force               rebuild existing collections"
  echo "  --max-builds N        limit builds (0=all, for debugging)"
  echo "  --python PATH         python interpreter (default: .venv/bin/python)"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS="$2"; shift 2 ;;
    --index-types) INDEX_TYPES="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --optimize-threads) OPTIMIZE_THREADS="$2"; shift 2 ;;
    --force|--force-build) FORCE="--force-build"; shift ;;
    --max-builds) MAX_BUILDS="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown arg: $1"; usage ;;
  esac
done

if [[ ! -x "${PYTHON}" ]]; then
  echo "python not found: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/build_logs"

LIST_ARGS=(--export-build-list --datasets "${DATASETS}" --index-types "${INDEX_TYPES}"
  --data-dir "${DATA_DIR}" --output-root "${OUTPUT_ROOT}")
[[ -n "${FORCE}" ]] && LIST_ARGS+=("${FORCE}")
[[ "${MAX_BUILDS}" -gt 0 ]] && LIST_ARGS+=(--max-builds "${MAX_BUILDS}")

mapfile -t LINES < <("${PYTHON}" "${GRID}" "${LIST_ARGS[@]}")
TOTAL=${#LINES[@]}
if [[ "${TOTAL}" -eq 0 ]]; then
  echo "[$(date '+%F %T')] nothing to build"
  exit 0
fi

echo "[$(date '+%F %T')] parallel build: total=${TOTAL} jobs=${JOBS} optimize_threads=${OPTIMIZE_THREADS}"

# 初始化 progress.json
"${PYTHON}" - <<PY
import json, time
from pathlib import Path
p = Path("${OUTPUT_ROOT}/progress.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps({
    "phase": "build",
    "total": ${TOTAL},
    "done": 0, "failed": 0, "skipped": 0,
    "current": "starting",
    "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "recent": [],
}, indent=2))
PY

running=0
idx=0
fail=0

for line in "${LINES[@]}"; do
  [[ -z "${line}" ]] && continue
  [[ "${line}" == *$'\t'* ]] || continue
  dataset="${line%%$'\t'*}"
  config_id="${line#*$'\t'}"

  while [[ "${running}" -ge "${JOBS}" ]]; do
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  done

  idx=$((idx + 1))
  logfile="${OUTPUT_ROOT}/build_logs/${dataset}_${config_id}.log"
  echo "[$(date '+%F %T')] spawn [${idx}/${TOTAL}] ${dataset}/${config_id}"

  (
    set +e
    "${PYTHON}" "${GRID}" \
      --build-one \
      --dataset "${dataset}" \
      --config-id "${config_id}" \
      --index-types "${INDEX_TYPES}" \
      --data-dir "${DATA_DIR}" \
      --output-root "${OUTPUT_ROOT}" \
      --batch-size "${BATCH_SIZE}" \
      --optimize-threads "${OPTIMIZE_THREADS}" \
      --progress-file "${OUTPUT_ROOT}/progress.json" \
      --progress-total "${TOTAL}" \
      ${FORCE} \
      > "${logfile}" 2>&1
    exit $?
  ) &

  running=$((running + 1))
done

while [[ "${running}" -gt 0 ]]; do
  if wait -n 2>/dev/null; then
    :
  else
    wait || fail=$((fail + 1))
  fi
  running=$((running - 1))
done

echo "[$(date '+%F %T')] all builds finished (see ${OUTPUT_ROOT}/progress.json)"
cat "${OUTPUT_ROOT}/progress.json"

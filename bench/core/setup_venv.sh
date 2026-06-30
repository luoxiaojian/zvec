#!/usr/bin/env bash
# Create a virtualenv and install dependencies for bench/core Python tools.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

# Prefer editable install from the repo when ZVEC_INSTALL_MODE=local (default).
INSTALL_MODE="${ZVEC_INSTALL_MODE:-local}"
if [[ "${INSTALL_MODE}" == "local" ]]; then
  echo "Installing zvec from source (${REPO_ROOT}) ..."
  if [[ -n "${USE_OSS_MIRROR:-}" ]]; then
    export CMAKE_ARGS="-DUSE_OSS_MIRROR=${USE_OSS_MIRROR}"
    echo "CMAKE_ARGS=${CMAKE_ARGS}"
  fi
  python -m pip install -e "${REPO_ROOT}"
else
  echo "Installing zvec from PyPI ..."
  python -m pip install zvec
fi

echo
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"

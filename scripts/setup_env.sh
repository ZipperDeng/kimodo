#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_VERSION="${KIMODO_PYTHON:-3.11}"
VENV_DIR="${REPO_ROOT}/.venv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      if [[ $# -lt 2 ]]; then
        echo "--venv-dir requires a path" >&2
        exit 2
      fi
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: bash scripts/setup_env.sh [--venv-dir PATH]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: bash scripts/setup_env.sh [--venv-dir PATH]" >&2
      exit 2
      ;;
  esac
done

command -v git >/dev/null 2>&1 || {
  echo "git is required to install the SOMA-X dependency." >&2
  exit 1
}

if ! command -v uv >/dev/null 2>&1; then
  command -v curl >/dev/null 2>&1 || {
    echo "curl is required to install uv." >&2
    exit 1
  }
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi

command -v uv >/dev/null 2>&1 || {
  echo "uv was installed but is not available on PATH. Open a new shell and rerun this script." >&2
  exit 1
}

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

if [[ -n "${KIMODO_TORCH_INDEX_URL:-}" ]]; then
  uv pip install --python "${VENV_DIR}/bin/python" torch --index-url "${KIMODO_TORCH_INDEX_URL}"
else
  uv pip install --python "${VENV_DIR}/bin/python" torch
fi

uv pip install --python "${VENV_DIR}/bin/python" "${REPO_ROOT}[soma,train]"

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import kimodo

print("Kimodo import: OK")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Warning: GPU acceleration is unavailable. Install a CUDA-enabled PyTorch wheel if needed.")
PY

echo
echo "Kimodo environment is ready. Activate it with:"
echo "  source \"${VENV_DIR}/bin/activate\""

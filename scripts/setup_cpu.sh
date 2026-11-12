#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

python3 - <<'PY'
import sys

if sys.version_info < (3, 9) or sys.version_info >= (3, 13):
    raise SystemExit(
        "Нужен Python версии 3.9-3.12. "
        f"Сейчас используется {sys.version.split()[0]}."
    )
PY

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV_DIR/bin/python" -m pip install -r "$PROJECT_DIR/requirements.txt"

"$VENV_DIR/bin/python" - <<'PY'
import platform
import torch

print(f"Python architecture: {platform.machine()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("CPU-окружение готово.")
PY

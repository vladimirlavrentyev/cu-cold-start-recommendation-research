#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_DIR="$PROJECT_DIR/third_party/TaNP"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"
NUM_EPOCHS="${1:-1}"
RUN_NAME="${2:-cpu_run}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Не найдено окружение $PYTHON_BIN"
    echo "Сначала выполните: ./setup_cpu.sh"
    exit 1
fi

if [[ ! -d "$CODE_DIR/data/lastfm_20/training/log" ]]; then
    echo "Не найдены распакованные данные LastFM."
    echo "Выполните: cd TaNP && unzip -q data.zip"
    exit 1
fi

cd "$CODE_DIR"

"$PYTHON_BIN" train_TaNP.py \
  --cpu \
  --data_dir data/lastfm_20 \
  --model_save_dir "save_model_cpu/$RUN_NAME" \
  --id "$RUN_NAME" \
  --save 1 \
  --seed 2020 \
  --first_embedding_dim 32 \
  --second_embedding_dim 16 \
  --z1_dim 32 \
  --z2_dim 32 \
  --z_dim 32 \
  --enc_h1_dim 32 \
  --enc_h2_dim 16 \
  --taskenc_h1_dim 32 \
  --taskenc_h2_dim 32 \
  --taskenc_final_dim 16 \
  --clusters_k 10 \
  --temperature 1.0 \
  --lambda 1.0 \
  --dec_h1_dim 32 \
  --dec_h2_dim 32 \
  --dec_h3_dim 16 \
  --dropout_rate 0 \
  --lr 0.0001 \
  --optim adam \
  --num_epoch "$NUM_EPOCHS" \
  --batch_size 32 \
  --train_ratio 0.7 \
  --valid_ratio 0.1 \
  --support_size 20 \
  --query_size 10 \
  --max_len 200 \
  --context_min 20

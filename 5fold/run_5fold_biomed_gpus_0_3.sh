#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/colon/Data/Users/jhonatan/CLIP_prefix_caption"
FOLDS_DIR="${PROJECT_ROOT}/5fold"
WEIGHTS_PATH="${PROJECT_ROOT}/clip_weights/biomedclip_weights.pt"

run_fold() {
  local fold="$1"
  local gpu="$2"

  local fold_dir="${FOLDS_DIR}/fold_${fold}"
  local train_csv="${fold_dir}/train.csv"
  local train_pkl="${fold_dir}/train_biomed.pkl"
  local out_dir="${fold_dir}/checkpoints"
  local prefix="colono_bio_fold${fold}"

  echo "=== Fold ${fold} | GPU física ${gpu} ==="
  echo "CSV: ${train_csv}"

  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${PROJECT_ROOT}/parse_colono_biomed.py" \
    --csv_path "${train_csv}" \
    --images_root "${PROJECT_ROOT}" \
    --out_path "${train_pkl}" \
    --weights_path "${WEIGHTS_PATH}" \
    --batch_size 64

  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${PROJECT_ROOT}/train.py" \
    --data "${train_pkl}" \
    --out_dir "${out_dir}" \
    --prefix "${prefix}" \
    --epochs 15 \
    --save_every 5 \
    --bs 4 \
    --mapping_type transformer \
    --num_layers 8

  echo "Fold ${fold} terminado."
  echo "Gráfica (PNG si hay matplotlib): ${out_dir}/${prefix}_loss_curve.png"
  echo "Gráfica fallback (SVG): ${out_dir}/${prefix}_loss_curve.svg"
  echo "Gráfica epoch fallback (SVG): ${out_dir}/${prefix}_loss_epoch.svg"
  echo "CSV step-loss: ${out_dir}/${prefix}_loss_per_step.csv"
  echo "CSV epoch-loss: ${out_dir}/${prefix}_loss_per_epoch.csv"
}

run_fold 1 0 & run_fold 2 3 & wait
run_fold 3 0 & run_fold 4 3 & wait
run_fold 5 0

echo "Todos los folds completados."

#!/usr/bin/env bash
set -euo pipefail

# Modelo: resnet101
# GPU: 3
# Comandos de referencia equivalentes al flujo automático.
# Nota: scripts/2fold_models.py calcula ademas val_loss y guarda embeddings pre/post-GPT por fold.

# ===== fold_1 =====
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono.py --csv_path /workspace/fold/2fold/resnet/folds/fold_1/train.csv --images_root . --out_path /workspace/fold/2fold/resnet/folds/fold_1/data/train.pkl --clip_model_type RN101
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono.py --csv_path /workspace/fold/2fold/resnet/folds/fold_1/val.csv --images_root . --out_path /workspace/fold/2fold/resnet/folds/fold_1/data/val.pkl --clip_model_type RN101
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/train.py --data /workspace/fold/2fold/resnet/folds/fold_1/data/train.pkl --out_dir /workspace/fold/2fold/resnet/folds/fold_1/train --prefix positive_vs_negative_fold_1 --epochs 15 --save_every 5 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/test.py --images_root /workspace/fold/2fold/resnet/folds/fold_1/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_1> --output_csv /workspace/fold/2fold/resnet/folds/fold_1/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder rn101 --openai_clip_name RN101

# ===== fold_2 =====
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono.py --csv_path /workspace/fold/2fold/resnet/folds/fold_2/train.csv --images_root . --out_path /workspace/fold/2fold/resnet/folds/fold_2/data/train.pkl --clip_model_type RN101
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono.py --csv_path /workspace/fold/2fold/resnet/folds/fold_2/val.csv --images_root . --out_path /workspace/fold/2fold/resnet/folds/fold_2/data/val.pkl --clip_model_type RN101
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/train.py --data /workspace/fold/2fold/resnet/folds/fold_2/data/train.pkl --out_dir /workspace/fold/2fold/resnet/folds/fold_2/train --prefix positive_vs_negative_fold_2 --epochs 15 --save_every 5 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/test.py --images_root /workspace/fold/2fold/resnet/folds/fold_2/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_2> --output_csv /workspace/fold/2fold/resnet/folds/fold_2/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder rn101 --openai_clip_name RN101

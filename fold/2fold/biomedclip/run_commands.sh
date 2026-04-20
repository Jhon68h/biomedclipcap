#!/usr/bin/env bash
set -euo pipefail

# Modelo: biomedclip
# GPU: 3
# Comandos de referencia equivalentes al flujo automático.
# Nota: la validacion/val_loss se calcula via scripts/validation.py.

# ===== fold_1 =====
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_1/train.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_1/data/train.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_1/val.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_1/data/val.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/train.py --data /workspace/fold/2fold/biomedclip/folds/fold_1/data/train.pkl --out_dir /workspace/fold/2fold/biomedclip/folds/fold_1/train --prefix positive_vs_negative_fold_1 --epochs 15 --save_every 1 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/test.py --images_root /workspace/fold/2fold/biomedclip/folds/fold_1/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_1> --output_csv /workspace/fold/2fold/biomedclip/folds/fold_1/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder biomedclip --biomedclip_model_id hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

# ===== fold_2 =====
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_2/train.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_2/data/train.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_2/val.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_2/data/val.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/train.py --data /workspace/fold/2fold/biomedclip/folds/fold_2/data/train.pkl --out_dir /workspace/fold/2fold/biomedclip/folds/fold_2/train --prefix positive_vs_negative_fold_2 --epochs 15 --save_every 1 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/test.py --images_root /workspace/fold/2fold/biomedclip/folds/fold_2/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_2> --output_csv /workspace/fold/2fold/biomedclip/folds/fold_2/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder biomedclip --biomedclip_model_id hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

# ===== Post-proceso =====
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/scripts/validation.py --model biomedclip --output_root /workspace/fold/2fold --start_epoch 0 --end_epoch 14
CUDA_VISIBLE_DEVICES=3 /opt/conda/bin/python /workspace/scripts/graphics/graficas.py --input_root /workspace/fold/2fold --output_root /workspace/fold/2fold/plots --models biomedclip

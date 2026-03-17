#!/usr/bin/env bash
set -euo pipefail

# Modelo: biomedclip
# GPU: no_forzada
# Comandos de referencia equivalentes al flujo automático.
# Nota: scripts/2fold_models.py calcula ademas val_loss y guarda embeddings pre/post-GPT por fold.

# ===== fold_1 =====
/opt/conda/bin/python3 /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_1/train.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_1/data/train.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
/opt/conda/bin/python3 /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_1/val.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_1/data/val.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
/opt/conda/bin/python3 /workspace/train.py --data /workspace/fold/2fold/biomedclip/folds/fold_1/data/train.pkl --out_dir /workspace/fold/2fold/biomedclip/folds/fold_1/train --prefix positive_vs_negative_fold_1 --epochs 15 --save_every 1 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
/opt/conda/bin/python3 /workspace/test.py --images_root /workspace/fold/2fold/biomedclip/folds/fold_1/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_1> --output_csv /workspace/fold/2fold/biomedclip/folds/fold_1/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder biomedclip --biomedclip_model_id hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

# ===== fold_2 =====
/opt/conda/bin/python3 /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_2/train.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_2/data/train.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
/opt/conda/bin/python3 /workspace/parse_colono_biomed.py --csv_path /workspace/fold/2fold/biomedclip/folds/fold_2/val.csv --images_root . --out_path /workspace/fold/2fold/biomedclip/folds/fold_2/data/val.pkl --weights_path /workspace/clip_weights/biomedclip_weights.pt
/opt/conda/bin/python3 /workspace/train.py --data /workspace/fold/2fold/biomedclip/folds/fold_2/data/train.pkl --out_dir /workspace/fold/2fold/biomedclip/folds/fold_2/train --prefix positive_vs_negative_fold_2 --epochs 15 --save_every 1 --bs 4 --prefix_length 10 --prefix_length_clip 10 --mapping_type transformer --num_layers 8 --only_prefix --normalize_prefix
/opt/conda/bin/python3 /workspace/test.py --images_root /workspace/fold/2fold/biomedclip/folds/fold_2/inference/val_images_run_manual --checkpoint <CHECKPOINT_FOLD_2> --output_csv /workspace/fold/2fold/biomedclip/folds/fold_2/inference/val_predictions_raw.csv --prefix_length 10 --mapping_type transformer --num_layers 8 --beam_search --encoder biomedclip --biomedclip_model_id hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

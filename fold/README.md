# Real CLIPCap Fold Outputs

This directory contains real CLIPCap fold training artifacts.

## For each task
- dataset.csv
- folds/fold_*/train.csv and val.csv
- folds/fold_*/data/train.pkl and val.pkl
- folds/fold_*/train/* (checkpoints, train loss csv/svg)
- folds/fold_*/val_loss_per_epoch.csv
- folds/fold_*/train_vs_val_loss.svg
- folds/fold_*/inference/val_predictions.csv (unless --skip_inference)
- folds/fold_*/metrics.json
- results.json

## Global
- summary.json

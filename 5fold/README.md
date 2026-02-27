# 5-Fold estratificado (listo para correr)

Este directorio contiene los splits 5-fold generados a partir de:

- `experiments_colono/experiments_colono/clipclap_train_labeled_join.csv`

Sin modificar el dataset original.

## Archivos clave

- `case_assignments.csv`: asignación de `case -> fold` y etiqueta de estratificación.
- `dataset_with_folds.csv`: dataset completo con columna `fold`.
- `fold_paths.csv`: rutas de `train.csv` y `val.csv` por fold.
- `fold_{1..5}/train.csv` y `fold_{1..5}/val.csv`: splits para entrenamiento/validación.
- `create_5fold_stratified.py`: script para regenerar todo.
- `run_5fold_biomed_gpus_0_3.sh`: ejecución automática de entrenamiento 5-fold usando GPU 0 y 3.

## Regenerar splits

```bash
python3 5fold/create_5fold_stratified.py \
  --project_root /home/colon/Data/Users/jhonatan/CLIP_prefix_caption
```

## Rutas para entrenamiento

Usa `5fold/fold_paths.csv` como referencia de rutas por fold.

Ejemplo fold 1:

- train CSV: `5fold/fold_1/train.csv`
- val CSV: `5fold/fold_1/val.csv`
- pkl sugerido: `5fold/fold_1/train.pkl`

## Correr 5-fold en GPU 0 y 3

```bash
cd /home/colon/Data/Users/jhonatan/CLIP_prefix_caption
./5fold/run_5fold_biomed_gpus_0_3.sh
```

## Gráficas por fold

Cada fold guarda automáticamente:

- `5fold/fold_X/checkpoints/colono_bio_foldX_loss_curve.png` (si `matplotlib` está disponible)
- `5fold/fold_X/checkpoints/colono_bio_foldX_loss_curve.svg` (fallback sin dependencias)
- `5fold/fold_X/checkpoints/colono_bio_foldX_loss_epoch.svg` (fallback sin dependencias)
- `5fold/fold_X/checkpoints/colono_bio_foldX_loss_per_step.csv`
- `5fold/fold_X/checkpoints/colono_bio_foldX_loss_per_epoch.csv`

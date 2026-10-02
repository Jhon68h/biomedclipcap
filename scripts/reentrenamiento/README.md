# Reentrenamiento — §1 hiperparámetros + §2 selección por F1

Implementa los puntos 1 y 2 de [`reportes/reentrenamiento.md`](../../reportes/reentrenamiento.md)
sobre **SUN** (`scripts/2fold_models.py`, 2-fold estratificado por caso).

No hay un pipeline nuevo: se reutiliza el existente. Lo único nuevo es
[`val_metric_per_epoch.py`](val_metric_per_epoch.py), que produce la pieza que
faltaba para §2 (`val_task_metric_per_epoch.csv`).

---

## Qué cambió en los scripts existentes

### §1 — Hiperparámetros (antes hardcodeados)

| Parámetro | Antes | Ahora |
|---|---|---|
| `lr` | hardcodeado `2e-5` en la firma de `train()` | flag `--lr` |
| `warmup_steps` | hardcodeado `5000` en la firma de `train()` | flag `--warmup_steps` |
| `weight_decay` | nunca se pasaba a `AdamW` (quedaba en `0.0`) | flag `--weight_decay` |
| `dropout` (mapper) | `0.` por defecto en toda la cadena, nunca sobreescrito | flag `--dropout` |

- [`train.py`](../../train.py): los cuatro flags, `weight_decay` sí llega a `AdamW`,
  y `dropout` se propaga `ClipCaptionModel → TransformerMapper → Transformer →
  TransformerLayer → MultiHeadAttention / MlpTransformer`.
- [`scripts/2fold_models.py`](../2fold_models.py): los mismos flags, propagados a
  `train.py`, a `run_commands.sh` y a `run_config.json` (`training.*`), como pide §5.1.

`epochs`, `bs` y `num_layers` ya eran flags; solo cambian los valores que se pasan.

> **Sin early stopping por `val_loss`.** §1 lo menciona, pero §2 demuestra que
> `val_loss` es el criterio equivocado: parar por `val_loss` es el mismo error
> que elegir el checkpoint por `val_loss`. En su lugar se entrenan pocas épocas
> y se elige por F1 (§2). Los checkpoints de todas las épocas se guardan
> (`--save_every 1`), así que no se pierde nada.

> **Nota de comparabilidad:** `--bs` también se usa como batch de validación al
> calcular `val_loss_per_epoch.csv`. Como esa pérdida se promedia por batch, subir
> `bs` de 4 a 16 mueve un poco los valores de `val_loss` respecto al baseline.
> No afecta a las métricas de detección, que es lo que se reporta.

### §2 — Selección de checkpoint por F1

- **Nuevo:** `val_metric_per_epoch.py` → `val_task_metric_per_epoch.csv` por fold,
  con F1 / recall / precisión frame-level y `lesion_detection_rate_50pct` para
  **cada** checkpoint. Reutiliza `binary_metrics` y `lesion_level_metrics` de
  `scripts/evaluate_fold_models.py` tal cual, para que el criterio de selección y
  las tablas finales midan exactamente lo mismo.
- [`scripts/revalidate_epoch.py`](../revalidate_epoch.py) y
  [`scripts/inferiencia.py`](../inferiencia.py): nueva política
  `--checkpoint_policy f1` (+ `--f1_metric` para elegir la columna a maximizar).
  Ambos leen además `num_layers` / `prefix_length` / `mapping_type` desde
  `run_config.json`, así que un reentrenamiento con `num_layers 4` no obliga a
  repetirlos a mano.

---

## Cómo correrlo

### 1. Entrenar con los hiperparámetros nuevos

Valores de la tabla de §1 del reporte:

```bash
python scripts/2fold_models.py \
    --model all \
    --output_root fold/2fold_hp \
    --epochs 6 \
    --bs 16 \
    --lr 1e-5 \
    --warmup_steps 400 \
    --weight_decay 0.01 \
    --dropout 0.1 \
    --num_layers 4 \
    --save_every 1 \
    --gpu 0
```

Con `bs=16` y 5662 muestras son ~354 pasos/época, así que `--warmup_steps 400`
cubre algo más de una época. Con `bs=4` (~1416 pasos/época) el equivalente sería
`--warmup_steps 1400`; `train.py` avisa si el warmup se come todo el
entrenamiento.

`--output_root fold/2fold_hp` deja intactos `fold/2fold` (baseline, época 14) y
`fold/2fold_best` (checkpoint por `val_loss`), que son las dos filas de
comparación de §5.4.

### 2. F1 de detección por época

```bash
python scripts/reentrenamiento/val_metric_per_epoch.py \
    --fold_root fold/2fold_hp \
    --metric f1 \
    --gpu 0
```

No re-ejecuta `test.py`: los embeddings CLIP del fold de validación ya están en
`weights/sun_<experimento>/<modelo>/fold_N/val.pkl`, así que solo corre mapper + GPT-2 con decodificación
greedy en batch. Para acelerar más, `--sample_fraction 0.25`.

Escribe `fold/2fold_hp/<modelo>/folds/fold_N/val_task_metric_per_epoch.csv` y
`fold/2fold_hp/val_task_metric_summary.json`.

Para maximizar detección por lesión en vez de F1 frame-level:
`--metric lesion_detection_rate_50pct`.

### 3. Inferencia final con el checkpoint elegido por F1

```bash
python scripts/revalidate_epoch.py \
    --source_root fold/2fold_hp \
    --output_root fold/2fold_hp_f1 \
    --checkpoint_policy f1 \
    --f1_metric f1 \
    --gpu 0
```

Esto sí usa `test.py` con **beam search** sobre las imágenes, igual que el
pipeline original, así que los números resultantes son comparables uno a uno con
`fold/2fold` y `fold/2fold_best`.

### 4. Tablas del reporte

```bash
python scripts/evaluate_fold_models.py --fold_root fold/2fold_hp_f1
```

Produce `table_i_frame_level_metrics.csv`, `table_i_frame_level_metrics_ci.csv`,
`table_ii_clinical_report_generation_metrics.csv` y `table_ii_lesion_level.csv`
(§5.2).

### 5. (Opcional) IGHO con el mismo checkpoint

```bash
python scripts/inferiencia.py \
    --model all --fold all \
    --checkpoints_root fold/2fold_hp \
    --checkpoint_policy f1 \
    --output_root igho/videos_f1 \
    --gpu 0
```

---

## Dos advertencias que deben quedar en el reporte final

1. **La métrica de selección usa greedy; las tablas usan beam search.** El CSV
   por época sirve para **ordenar** épocas entre sí, no para reportarse como
   resultado. Si se quiere fidelidad exacta al pipeline de reporte,
   `--beam_size 5` (mucho más lento). Los números publicables salen siempre del
   paso 3 + 4.

2. **Seleccionar el checkpoint mirando el fold de validación es seleccionar
   sobre el propio conjunto de evaluación** (§2 del reporte). Con 2 folds y sin
   un split de desarrollo aparte, este sesgo no se puede eliminar, solo
   declarar. Aplica igual al criterio anterior por `val_loss`.

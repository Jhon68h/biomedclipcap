# Auditoria Tecnica de `fold/2fold`

## Resumen ejecutivo
- El pipeline de `scripts/2fold_models.py` si esta tomando el dataset esperado y lo procesa completo: 11,400 muestras (5,700 positivas y 5,700 negativas).
- La estratificacion es correcta por **label** (balance 50/50 en train y val por fold), pero **incorrecta para evaluar generalizacion por paciente/caso** porque no separa por `case`.
- Hay evidencia fuerte de fuga de informacion por caso: en ambos folds hay solapamiento total de casos entre train y val (`77/77`).
- Los resultados altos de Table I y Table II son coherentes con: fuga por caso, captions muy plantillados, y metricas de evaluacion basadas en reglas textuales que facilitan puntajes altos.

## Lo que confirmo el subagente
- Confirmo estructura general del pipeline en `scripts/2fold_models.py`: fusion de CSV positivo/negativo, asignacion de `sample_id`, split 2-fold estratificado por clase y ejecucion parse/train/test.
- No alcanzo a escribir este archivo; el analisis detallado y las verificaciones numericas de abajo fueron completadas en esta auditoria principal.

## Inventario de artefactos en `fold/2fold/`
- Tablas finales:
  - `table_i_frame_level_metrics.csv`
  - `table_ii_clinical_report_generation_metrics.csv`
- Resumen global de evaluacion:
  - `evaluate_fold_models_summary.json`
- Por modelo (`biomedclip`, `vit`, `resnet`):
  - `run_config.json`, `summary.json`, `pipeline.log`
  - `folds/fold_1` y `folds/fold_2` con `train.csv`, `val.csv`, `data/*.pkl`, `inference/val_predictions*.csv`, `val_manifest.csv`, `val_loss_summary.json`, checkpoints, etc.
- Graficas:
  - `plots/*.png`
  - `plots/summary.json`

## Como se obtuvieron los datos (pipeline)
1. Se leen los CSV fuente positivo y negativo.
2. Se construye un dataset unificado con columnas: `sample_id`, `image_path`, `case`, `label`, `caption`, etc.
3. Se aplica `stratified_two_fold(...)` para split 2-fold estratificado por label.
4. Se escriben `train.csv` y `val.csv` por fold.
5. Se generan embeddings (`parse_colono_biomed.py` o `parse_colono.py`) para train/val.
6. Se entrena `train.py` con `train.pkl`.
7. Se infiere sobre imagenes de validacion (`test.py`) y se produce `val_predictions_raw.csv`.
8. Se mergea con `val_manifest.csv` para `val_predictions.csv`.
9. `scripts/evaluate_fold_models.py` calcula Table I y Table II desde `val_predictions.csv`.

Referencias de codigo clave:
- Split por label: `scripts/2fold_models.py` (funcion `stratified_two_fold`, lineas ~234-261).
- Escritura de folds y artefactos: `scripts/2fold_models.py` (lineas ~644-687).
- Parse train/val por fold: `scripts/2fold_models.py` (lineas ~798-840).
- Carga de CSV en parsers:
  - `parse_colono_biomed.py` (lee `image_path` y `caption`, lineas ~114-121 y ~135-142).
  - `parse_colono.py` (lee `image_path` y `caption`, lineas ~19 y ~27-29).
- Entrenamiento usa solo `--data` (train pkl): `train.py` (lineas ~568-592).
- Inference escribe `image_path,generated_caption`: `test.py` (lineas ~377 y ~384-387).

## Verificacion de carga de dataset
- Fuente:
  - `clipclap_train_labeled_positive.csv`: 5,701 lineas (header + 5,700 datos).
  - `clipclap_train_labeled_negative.csv`: 5,701 lineas (header + 5,700 datos).
- Fold (ejemplo biomedclip/fold_1):
  - `train.csv`: 5,701 lineas (header + 5,700 datos).
  - `val.csv`: 5,701 lineas (header + 5,700 datos).
- `run_config.json` y `summary.json` reportan:
  - total_samples: 11,400
  - label_distribution: 5,700/5,700
  - val_missing_images: 0
  - raw_predictions_rows == matched_predictions_rows == 5,700 por fold

Diagnostico: el dataset **si se esta cargando y consumiendo completo**.

## Diagnostico de estratificacion
- Correcta por clase: cada fold queda balanceado (`2850` positivas y `2850` negativas en train/val).
- Incorrecta por caso/paciente:
  - Fold 1: `train_cases=77`, `val_cases=77`, `overlap_cases=77`.
  - Fold 2: `train_cases=77`, `val_cases=77`, `overlap_cases=77`.
- No hay overlap exacto de imagen (`overlap_images=0`), pero si hay overlap total de casos.

Conclusion de split: para clasificacion y generacion en escenario clinico, este split **no valida generalizacion out-of-case**.

## Medidas estadisticas observadas
- Table I (frame-level):
  - Accuracy aprox. `0.916` a `0.933`.
  - Precision/Recall/F1 altos con desviaciones pequenas.
- Table II (reporte clinico):
  - BLEU-1 aprox. `0.903` a `0.921`.
  - BLEU-4 aprox. `0.856` a `0.889`.
  - size MAE bajo (mejor en biomedclip).
- `evaluate_fold_models.py` calcula std como desviacion poblacional sobre solo 2 folds.

## Por que salen resultados tan altos
1. **Fuga por caso**: train y val comparten todos los casos (`77/77`), por lo que el modelo ve durante entrenamiento muchos frames del mismo caso que evalua despues.
2. **Captions altamente plantillados**:
   - negativas: `1` caption unico.
   - positivas: `51` captions unicos para `5,700` filas.
   - cada caso tiene 1 caption unico (sin variacion intracasos).
3. **Alto exact match textual en validacion**:
   - biomedclip: `0.8879`
   - vit: `0.8610`
   - resnet: `0.8546`
4. **Metricas derivadas por heuristica textual**:
   - binario se infiere de caption (`no polyps` => negativo; si no => positivo), no de evaluacion independiente ciega.
5. **`location_accuracy` potencialmente inflada**:
   - `LOCATIONS` incluye `"colon"` y el matching es substring.
   - `"colon"` puede dispararse por `"colonoscopy"` en captions negativos.
   - El soporte de location es `11,400` (todos los frames), lo cual sugiere metrica facil.
6. **BLEU implementado como modified precision (n-gram overlap), sin brevity penalty**, lo que suele inflar el valor frente a BLEU canonico.

## Riesgos de validez
- Riesgo alto de sobreestimar rendimiento real en nuevos pacientes/casos.
- Riesgo de que el modelo memorice plantillas de texto y patrones de caso.
- Riesgo de interpretar como "clinico" un puntaje que en parte mide coincidencia de plantilla.

## Checklist recomendado (sin Python)
1. Verificar interseccion de casos train/val por fold (debe ser 0 para evaluacion por paciente).
2. Contar diversidad real de captions GT y predichos.
3. Revisar que metricas clinicas usen parsing robusto por tokens/patrones con bordes de palabra (evitar `colon` dentro de `colonoscopy`).
4. Reportar BLEU canonico adicional (con BP) o metricas semanticas complementarias.
5. Repetir evaluacion con split por `case` (GroupKFold / holdout por paciente) y comparar contra este baseline.
6. Mantener una tabla separada: `in-case` vs `out-of-case`.

## Conclusion final
Los resultados actuales son utiles como referencia interna del pipeline, pero **no son suficientes para afirmar generalizacion clinica real**.  
La carga de datos esta bien, la estratificacion por label tambien, pero la validacion esta comprometida por no estratificar/particionar por caso-paciente y por metricas de texto que favorecen plantillas.

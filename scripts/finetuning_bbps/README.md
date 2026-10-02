# Fine-tuning IGHO + BBPS

Implementación de [`reportes/finetuning_bbps_igho.md`](../../reportes/finetuning_bbps_igho.md).

**Todo lo que produce el pipeline —CSVs, embeddings, pesos, predicciones y métricas—
cae bajo `igho_training/`.** Los frames se leen de `/frames` (el montaje de
`docker-compose.yml`), nunca se copian.

---

## Qué cambió en los scripts existentes

| Archivo | Cambio | Por qué |
|---|---|---|
| `train.py` | flag `--init_checkpoint` | §3.5, pieza bloqueante: sin esto no hay fine-tuning, se entrenaría IGHO desde cero con ~5.5k frames |
| `train.py` | flag `--val_data` → `val_loss_per_epoch.csv` | permite elegir época sin generar captions |
| `predict.py` | `stop_token_count` en `generate_beam` y `generate2` | ver el aviso de abajo |
| `test.py` | flag `--stop_token_count` | ídem |
| `scripts/evaluate_fold_models.py` | `extract_bbps()` + métrica `bbps_mae` en la Tabla II | §3.8 |

### Aviso: `--stop_token_count 2` no es opcional

`generate_beam` corta el beam en el **primer** punto. Las captions del fine-tuning
son **dos frases**:

```
This is a colonoscopy frame from a patient with a sessile adenoma polyp measuring
7 mm located in the descending colon. Bowel preparation score BBPS is 7/9.
```

Con el valor por defecto (`1`) el modelo nunca llega a emitir la frase del BBPS y la
evaluación daría 0% de emisión aunque el entrenamiento fuese perfecto.
`evaluate_bbps.py` avisa si la tasa de emisión baja del 95%.

---

## Scripts nuevos

### `build_bbps_dataset.py` — §3.1 a §3.4

Recorre las 27 lesiones de `igho/igho_dataset_copia.csv`, selecciona los frames
existentes en `[start, end]`, aplica el cap por lesión con stride uniforme, añade el
10% de negativos y parte los 15 videos en 2 folds.

Reglas que el plan dejaba abiertas y aquí quedan fijadas:

* **Solapamiento de rangos** (`--overlap_rule first_wins`): gana la lesión que aparece
  primero en el CSV. En los datos actuales no hay solapamientos; si aparecen, se avisan
  por stdout y quedan en `build_summary.json`.
* **Reparto de negativos** (`--negative_alloc proportional`): proporcional a los
  positivos de cada video, de modo que la razón sea la misma *dentro* de cada video y
  los 15 estén representados.
* **Margen de guarda** (`--negative_margin 300`, ~5 s a 60 fps): ningún negativo puede
  salir de `[start - N, end + N]`. La anotación solo cubre las lesiones reportadas, así
  que un frame pegado al borde del segmento probablemente sigue mostrando el pólipo.
  Verificado: con `--negative_margin 0` el negativo más cercano quedaba **a 1 frame**
  del inicio de un rango (`2025-12-01_141231_211`, frame 13979 vs rango que empieza en
  13980) — un falso negativo metido a mano en el entrenamiento.
* **Frames inexistentes**: se escanea el disco. Los rangos del CSV son aritméticos y
  varios videos tienen menos frames extraídos que su `end`. Si a un video le faltan
  *todos* los frames, el script **falla** (`--allow_missing_videos` para forzar): un
  video ausente rompe el balance de BBPS del split sin que se note en las métricas.

Salida con los parámetros por defecto: **4.964 positivos + 4.964 negativos = 9.928**
(los positivos no son 5.400 exactos porque 4 lesiones tienen rangos de menos de 200
frames).

| | fold_1 | fold_2 |
|---|---|---|
| train | grupo A, 8 videos, 5.564 | grupo B, 7 videos, 4.364 |
| val | grupo B, 7 videos, 4.364 | grupo A, 8 videos, 5.564 |
| BBPS no vistos en train | 3 (`225252`) | 2 (`321`), 4 (`265`) |

### Por qué el ratio por defecto es 1.0 y no el 10% del plan

El §3.3 propone 10%. El problema no es el volumen de datos sino **el prior de clase**:

| Etapa | % negativos | prior positivo |
|---|---:|---:|
| SUN — el checkpoint del que se parte (`fold/2fold`) | 50,3% | **49,7%** |
| Fine-tuning con `--negative_ratio 0.10` | 9,1% | **90,9%** |
| Fine-tuning con `--negative_ratio 1.0` | 50,0% | 50,0% |
| Video IGHO real, los 15 videos | ~65% | **~35%** |

Con 0.10 el fine-tuning empuja la frontera de decisión hacia "siempre positivo", al
revés de lo que pide el despliegue, y biomedclip **ya** sobre-predice positivos en IGHO
(precisión 0.185, especificidad 0.814 en `igho/metrics`). El motivo que da el plan para
topar en 10% —que los negativos "ahoguen la señal positiva"— no aplica aquí: los
negativos **llevan el BBPS de su video**, así que cada uno es un ejemplo completo de
BBPS. Lo único que no enseñan son atributos de lesión, y esos ya están pseudo-replicados
(27 descripciones únicas repetidas ~184 veces cada una).

`--negative_ratio 0.10` reproduce el plan tal cual si quieres comparar.

### `evaluate_bbps.py` — §3.8 y paso 7

BBPS **por video (n=15)**, consolidando los frames por mediana. Promediar el error por
frame sería la misma pseudo-replicación de
[`camino_1_evaluacion_por_lesion.md`](../../reportes/camino_1_evaluacion_por_lesion.md).

Reporta: `bbps_mae` con IC95% bootstrap sobre videos, el MAE **excluyendo** los videos
cuyo BBPS no estaba en su fold de entrenamiento, la tasa de emisión de la frase, y el
baseline trivial (predecir siempre 7/9 → **MAE 1.333**, que el script recalcula y no
hardcodea).

---

## Runbook

**Todo corre dentro del contenedor.** Fuera no existe `/frames`, y además las rutas que
`build_bbps_dataset.py` escribe en el manifiesto tienen que ser las mismas que va a ver
`test.py`.

```bash
docker compose up -d
docker compose exec biomedCLIPCAP bash
cd /workspace
```

`CUDA_VISIBLE_DEVICES=1` en todos los pasos de GPU: en esta máquina la GPU 1 es la
RTX 4090 (cc 8.9), la más rápida y con soporte bf16. Hay que ponerlo explícito porque
`parse_colono_biomed.py` y `train.py` hacen `os.environ.setdefault` a las GPU 3 y 0.

### 0. Preflight — los 15 videos deben dar > 0

```bash
for v in $(tail -n +2 igho/igho_dataset_copia.csv | cut -d, -f1 | sort -u); do
  echo -n "$v: "; ls /frames/$v 2>/dev/null | wc -l
done
```

### 1. Dataset (§3.1–3.4)

```bash
python scripts/finetuning_bbps/build_bbps_dataset.py \
  --gt_csv igho/igho_dataset_copia.csv \
  --frames_root /frames \
  --output_root igho_training \
  --cap_per_lesion 200 \
  --negative_ratio 1.0 --negative_margin 300 \
  --negative_alloc proportional --overlap_rule first_wins \
  --annotation_fps 60 --video_fps_json igho/video_fps.json \
  --allow_missing_videos \
  --overwrite
```

#### `--annotation_fps 60` no es opcional

Los `start`/`end` de `igho_dataset_copia.csv` estan escritos en una base de 60 fps.
Sin esta bandera no se reescala nada y los rangos de los videos de 30 fps caen al
doble de donde toca. El factor que se aplica es `video_fps.json[id] / 60`.

#### `video_fps.json` guarda los fps EFECTIVOS DE EXTRACCION, no los del contenedor

Es la parte que mas facil se rompe. Dos videos tienen ahi un valor que parece
erroneo y **no lo es**:

| video | cabecera AVI | fps en el JSON | por que |
|---|---:|---:|---|
| `2025-05-28_110348_321` | 60 | **18.19** | el AVI declara 59343 frames pero solo decodifica 17990 reales (verificado con `ffmpeg -i x.avi -vsync 0 -f null -`). En disco estan los 17990, completos. |
| `2025-05-30_102030_811` | 60 | **20.0** | declara 39234, decodifica 13078 reales (exactamente 1/3). En disco hay 12534 (95.8%). |

Con el 60 de la cabecera, el rango de `811` caia entero fuera del video (0 frames
seleccionados, video excluido del entrenamiento) y el de `321` se colapsaba a los
primeros 950 frames de un segmento de 35401. Corregido: `321` pasa a 10733 frames
disponibles y `811` a 2861.

Para recalcular el fps efectivo de un video nuevo:

```bash
N=$(ffmpeg -hide_banner -i VIDEO.avi -vsync 0 -f null - 2>&1 | grep -o 'frame=[ ]*[0-9]*' | tail -1 | grep -o '[0-9]*')
D=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 VIDEO.avi)
python3 -c "print(round($N/$D, 2))"
```

#### `--allow_missing_videos`: 2 lesiones de 27 no se pueden recuperar

`2026-06-01_101053_265#L5` y `#L6` (rangos 40200-40920 y 41640-41700, o sea 12 s y
1 s de video) caen mas alla del ultimo frame extraido de ese video (19284, que
equivale al indice 38568 de la anotacion). La extraccion de `265` se corto a un
92.5%. Las otras 25 lesiones si entran.

### 2. Embeddings (§3.6) — 2 corridas, no 4

`parse_colono_biomed.py` pasa cada imagen del CSV por el encoder de BiomedCLIP y guarda
un `.pkl` con un vector de 512 dims por frame más su caption. ClipCap no entrena sobre
imágenes sino sobre esos vectores, así que este paso se hace **una vez** y `train.py`
solo lee el `.pkl`.

El split es por grupo de videos: **A** (8 videos) y **B** (7 videos). `fold_1` entrena
con A y valida con B; `fold_2` al revés. Es decir `fold_1/train.csv` y `fold_2/val.csv`
son el mismo archivo (grupo A), y `fold_1/val.csv` y `fold_2/train.csv` también (grupo
B) — verificado con `cmp`. Por eso bastan 2 corridas: una por grupo.

```bash
# grupo A  (= fold_1/train.csv = fold_2/val.csv)
CUDA_VISIBLE_DEVICES=1 python parse_colono_biomed.py \
  --csv_path igho_training/folds/fold_1/train.csv --images_root . \
  --out_path weights/igho_bbps_real/group_a.pkl \
  --weights_path clip_weights/biomedclip_weights.pt

# grupo B  (= fold_1/val.csv = fold_2/train.csv)
CUDA_VISIBLE_DEVICES=1 python parse_colono_biomed.py \
  --csv_path igho_training/folds/fold_1/val.csv --images_root . \
  --out_path weights/igho_bbps_real/group_b.pkl \
  --weights_path clip_weights/biomedclip_weights.pt
```

No hace falta copiar ni enlazar nada: en el paso 3 se le pasan estas dos rutas
directamente a `train.py`, intercambiadas según el fold.

> **Los `.pkl` van en `weights/igho_bbps_<etiqueta>/` (`real` o `generated`), nunca en `igho_training/data/`.**
> `--overwrite` borra `data/` y `folds/` en cada rebuild; si los embeddings viven ahí,
> cada reconstrucción se los come y hay que repetir ~30 min de GPU. El builder aborta si
> detecta `.pkl` dentro de las carpetas que va a borrar.

### 3. Fine-tuning (§3.7) — el mismo `--init_checkpoint` en ambos folds

`fold_1` entrena con A y valida con B; `fold_2` intercambia los dos `.pkl` del paso 2.

```bash
for F in fold_1 fold_2; do
  if [ "$F" = "fold_1" ]; then TR=group_a; VA=group_b; else TR=group_b; VA=group_a; fi
  CUDA_VISIBLE_DEVICES=1 python train.py \
    --data     weights/igho_bbps_real/$TR.pkl \
    --val_data weights/igho_bbps_real/$VA.pkl \
    --out_dir  igho_training/folds/$F/train \
    --prefix   bbps_$F \
    --init_checkpoint fold/2fold/biomedclip/folds/fold_1/train/positive_vs_negative_fold_1-001.pt \
    --epochs 5 --save_every 1 --bs 8 \
    --lr 2e-6 --warmup_steps 50 --weight_decay 0.01 --dropout 0.1 \
    --amp auto \
    --prefix_length 10 --prefix_length_clip 10 \
    --mapping_type transformer --num_layers 8 \
    --only_prefix --normalize_prefix
done
```

### 4. Inferencia sobre validación — `--stop_token_count 2` es obligatorio

```bash
for F in fold_1 fold_2; do
  CUDA_VISIBLE_DEVICES=1 python test.py \
    --images_root igho_training/folds/$F/inference/val_images \
    --checkpoint  igho_training/folds/$F/train/bbps_$F-004.pt \
    --output_csv  igho_training/folds/$F/inference/val_predictions_raw.csv \
    --encoder biomedclip \
    --biomedclip_model_id hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --prefix_length 10 --mapping_type transformer --num_layers 8 \
    --beam_search --stop_token_count 2
done
```

### 5. Métricas (§3.8 + paso 7)

```bash
python scripts/finetuning_bbps/evaluate_bbps.py \
  --fold_root igho_training/folds \
  --build_summary igho_training/data/build_summary.json \
  --output_dir igho_training/eval \
  --bootstrap 2000 \
  --compare_csv igho/metrics/promedio_todos_los_videos.csv \
  --compare_model biomedclip_promedio
```

### Opcional — barrer épocas

```bash
cat igho_training/folds/fold_1/train/val_loss_per_epoch.csv

for E in 002 003 004; do
  for F in fold_1 fold_2; do
    CUDA_VISIBLE_DEVICES=1 python test.py \
      --images_root igho_training/folds/$F/inference/val_images \
      --checkpoint  igho_training/folds/$F/train/bbps_$F-$E.pt \
      --output_csv  igho_training/folds/$F/inference/val_predictions_epoch$E.csv \
      --encoder biomedclip --prefix_length 10 --mapping_type transformer \
      --num_layers 8 --beam_search --stop_token_count 2
  done
  python scripts/finetuning_bbps/evaluate_bbps.py \
    --predictions_name val_predictions_epoch$E.csv --tag epoch$E
done
```

---

## Rendimiento

### Los folds comparten los embeddings: 2 corridas, no 4

`fold_1/train.csv` y `fold_2/val.csv` son **byte a byte el mismo archivo** (grupo A), y lo
mismo pasa con `fold_1/val.csv` y `fold_2/train.csv` (grupo B). Codificar los cuatro CSV
son 19.856 imágenes para solo 9.928 únicas — exactamente el doble del trabajo necesario.
Codificando un `.pkl` por grupo y pasándoselos a `train.py` intercambiados (pasos 2 y 3
del runbook) se hace el trabajo una sola vez. El caché de tokens (`*_tokens.pkl`) que
crea `ClipCocoDataset` se escribe junto al `.pkl` y también se comparte entre folds, que
es lo correcto: depende solo del contenido del `.pkl`, no del fold.

### Qué se optimizó en `train.py`

| Cambio | Efecto |
|---|---|
| GPT-2 con `requires_grad=False` en `ClipCaptionPrefix` | era un desperdicio puro: el optimizador nunca tocó esos pesos, pero cada backward calculaba los gradientes de los 124M de parámetros y —como `zero_grad()` usa el `parameters()` sobreescrito— **nunca se limpiaban**, acumulándose toda la corrida (~500 MB vivos). Quita ~1/3 del backward sin cambiar el resultado. |
| `--amp auto` | bf16 en Ampere+ (4090, A2000), fp16 si no; habilita TF32 de paso. ~2×. |
| `pin_memory` + `non_blocking` + `set_to_none=True` | marginal, pero gratis. |

`--amp off` es el valor por defecto y reproduce el fp32 de siempre, para no alterar
las corridas de SUN que lanza `scripts/2fold_models.py`.

### Dónde está de verdad el tiempo

El entrenamiento **no es el cuello de botella**: ~6.200 pasos de GPT-2 small con batch 8
y secuencias de ~55 tokens son minutos, no horas. El orden real de coste es:

1. **Inferencia (`test.py`)** — beam search con `beam_size=5`, **una imagen a la vez** y
   **sin caché KV**: cada uno de los ~50 pasos de generación recalcula la atención sobre
   toda la secuencia. Es de lejos lo más caro y crece con cada época que evalúes.
2. **Embeddings** — dominado por la carga y el `preprocess` de PNG en un solo hilo de CPU,
   no por la GPU. Se reduce a la mitad con lo de arriba.
3. **Entrenamiento** — minutos.

### Elegir época

`igho_training/folds/fold_N/train/val_loss_per_epoch.csv` da la `val_loss` por época sin
generar captions. Pero en este repo ya se verificó que **`val_loss` es mal selector para
métricas de tarea** (ver `find_best_checkpoint_by_task_metric` en
`scripts/inferiencia.py`): úsala para descartar, y confirma corriendo los pasos 4 y 5
sobre 2-3 checkpoints con `--tag epochN`.

---

## Estructura de salida

```
igho_training/
├── (embeddings en weights/igho_bbps_<etiqueta>/: group_a.pkl, group_b.pkl; --overwrite NO los toca)
├── data/                       (--overwrite borra esta carpeta entera)
│   ├── frames_index.csv        todos los frames seleccionados + metadatos
│   ├── split_by_video.csv      video, BBPS, grupo, nº lesiones, positivos, negativos
│   └── build_summary.json      parámetros, solapamientos, rangos truncados, baseline
├── folds/fold_{1,2}/
│   ├── train.csv / val.csv     image_path,caption  (entrada de parse_colono_biomed.py)
│   ├── train_meta.csv / val_meta.csv
│   ├── data/                   train.pkl, val.pkl
│   ├── train/                  PESOS bbps_fold_N-0??.pt + curvas + val_loss_per_epoch.csv
│   └── inference/
│       ├── val_images/         symlinks a /frames que consume test.py
│       ├── val_manifest.csv
│       └── val_predictions_raw.csv
└── eval/
    ├── bbps_per_video.csv
    ├── bbps_summary.json
    └── predictions_merged.csv
```

---

## Limitaciones que hay que declarar en el reporte final

* **n=15 videos.** Ningún IC va a ser estrecho y ninguna diferencia va a ser
  confirmable. Es una prueba de concepto.
* **BBPS 2, 3 y 4 tienen un solo video cada uno.** En el fold donde son validación el
  modelo nunca vio ese valor. `evaluate_bbps.py` los marca y reporta el MAE con y sin
  ellos; no los promedies a ciegas.
* **BBPS por video vs. por segmento de colon.** Clínicamente el BBPS es 0-3 por tercio.
  Si el valor anotado es del video completo pero las lesiones están en segmentos
  distintos, la etiqueta puede no corresponder al segmento visible. Sigue pendiente de
  confirmar con el especialista (§4 del plan, paso 1 del orden de ejecución).

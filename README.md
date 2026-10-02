# Prediccion_malignidad_calidad_colon_reportes

## Información General

- **Autor:** Jhonnatan David Hernandez Martinez
- **Director:** Fabio Martinez
- **Co-Director:** Edgar Rangel
- **Grado:** Pregrado en Ingeniería de Sistemas
- **Institución:** Universidad Industrial de Santander (UIS)
- **Laboratorio:** Biomedical Imaging, Vision and Learning Laboratory
  ([BIVL²ab](https://bivl2ab.uis.edu.co/))
- **Fecha de inicio:** [02/2026]
- **Última publicación de este proyecto:** [02/10/2026]

Este proyecto corresponde a la sección de calidad de colonoscopia dentro del marco de trabajo de grado, siendo asi resultado de un objetivo especifico. Se propuso un modelo de aprendizaje profundo multimodal basado en aprendizaje contrastivo que integre información visual y textual para la caracterización de la malignidad de los pólipos

## Objetivo

### Objetivo general

Desarrollar una representación de aprendizaje profundo para predecir la malignidad de pólipos usando observaciones colonoscópicas, índices de calidad y reportes clínicos.

### Objetivos específicos

- Seleccionar un conjunto de datos que integre secuencias colonoscópicas, reportes clínicos asociados a pólipos y anotaciones de calidad de colonoscopia.
- *Desarrollar un modelo de aprendizaje profundo multimodal basado en aprendizaje contrastivo que integre información visual y textual para la caracterización de la malignidad de los pólipos.*(Repositorio dedicado a este proyecto)
- Desarrollar un modelo de aprendizaje profundo para estimar el índice de preparación intestinal. (Repositorio dedicado a este proyecto)
- Validar los métodos propuestos mediante métricas de clasificación y de evaluación de reportes generados.

## Método propuesto

El método adapta **ClipCap** (*CLIP Prefix for Image Captioning*) al dominio de
colonoscopia: a partir de un cuadro, el modelo genera un reporte clínico breve
que indica si hay pólipo y, cuando lo hay, su tipo, morfología, tamaño y
localización. El pipeline está compuesto por tres etapas:

1. **Codificación visual a nivel de cuadro:** cada cuadro se procesa con un
   encoder visual congelado para obtener un embedding de 512 dimensiones. Se
   comparan tres encoders:

   | Variante | Modelo | Arquitectura visual | Pesos |
   |---|---|---|---|
   | `biomedclip` | BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`) | ViT-B/16 | `clip_weights/biomedclip_weights.pt` |
   | `vit` | CLIP de OpenAI | ViT-B/32 | `clip_weights/ViT-B-32.pt` |
   | `resnet` | CLIP de OpenAI | ResNet-101 | `clip_weights/RN101.pt` |

   El ViT-B/16 del proyecto es el de BiomedCLIP (preentrenado en imágenes
   biomédicas); no se usa el ViT-B/16 de CLIP de OpenAI.

   Los embeddings se precalculan una sola vez con `parse_colono_biomed.py`
   (BiomedCLIP) o `parse_colono.py` (CLIP) y se guardan en archivos `.pkl`
   junto con las captions.
2. **Mapper visual-textual:** un mapper *transformer* (8 capas, prefijo de
   longitud 10) proyecta el embedding visual a una secuencia de vectores que
   GPT-2 interpreta como prefijo. GPT-2 se mantiene congelado (`--only_prefix`)
   y solo se entrena el mapper.
3. **Generación del reporte:** GPT-2 genera la caption token a token mediante
   *beam search* (`beam_size = 5`). De la caption se extraen la predicción
   binaria (pólipo / sin pólipo) y los atributos clínicos (malignidad,
   localización, clasificación de París y tamaño).

Ejemplos de captions:

```text
This is a colonoscopy frame from a patient with a sessile adenoma polyp measuring 7 mm located in the descending colon.
This is a colonoscopy frame from a patient with no polyps.
```

Para la **estimación del índice de preparación intestinal**, el modelo
entrenado en SUN se ajusta (*fine-tuning*) sobre los videos de IGHO añadiendo
una segunda frase a la caption:

```text
... located in the descending colon. Bowel preparation score BBPS is 7/9.
```

El BBPS anotado corresponde al **colon completo (0–9)**, por lo que la
predicción por video se consolida tomando la mediana de los BBPS generados
sobre todos sus cuadros.

### Datos

- **SUN Multimodal:** 11.400 cuadros (5.700 con pólipo y 5.700 sin pólipo) de
  77 casos, con captions clínicas. Se evalúa con un esquema 2-fold
  estratificado por caso (`scripts/2fold_models.py`).
- **IGHO (cohorte clínica privada):** 15 videos completos de colonoscopia con
  27 lesiones anotadas (rango de cuadros, segmento, localización, tamaño,
  clasificación de París, diagnóstico NICE) y BBPS por video. Los videos `.avi`
  se convierten a cuadros PNG; `igho_inference/video_fps.json` guarda los fps
  efectivos de extracción, necesarios para reescalar los rangos anotados
  (escritos en base de 60 fps).
  - `igho_dataset_with_bbps.csv`: BBPS estimado empíricamente.
  - `igho_dataset_bbps_gt.csv`: BBPS proporcionado por el especialista.
  - Para el fine-tuning de BBPS se seleccionan hasta 200 cuadros por lesión y
    el mismo número de negativos (ratio 1:1, margen de guarda de 300 cuadros
    alrededor de cada lesión): 8.948 cuadros en total, divididos por video en
    dos grupos (A: 8 videos, B: 7 videos).

### Resultados

**SUN, 2-fold (`fold/2fold_hp_f1`, checkpoint elegido por F1):**

| Encoder | Accuracy | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| BiomedCLIP | 0,786 | 0,880 | 0,662 | **0,756** | 0,910 |
| CLIP RN101 | 0,676 | 0,901 | 0,395 | 0,549 | 0,957 |
| CLIP ViT-B/32 | 0,647 | 0,906 | 0,328 | 0,481 | 0,966 |

BiomedCLIP alcanza BLEU-1 de 0,663, BLEU-4 de 0,273 y un error medio de
tamaño de 3,09 mm.

**IGHO, inferencia sobre videos completos con los modelos de SUN
(`igho_inference/inference_f1`, 7 videos):** BiomedCLIP obtiene F1 de 0,443,
recall de 0,645 y especificidad de 0,876 a nivel de cuadro, frente a F1 de
0,229 (RN101) y 0,191 (ViT-B/32).

**IGHO, fine-tuning con BBPS (n = 15 videos):** la frase de BBPS se emite en
el 100 % de los cuadros, pero el modelo no supera el baseline trivial de
predecir siempre el valor más frecuente:

| Etiqueta BBPS | MAE por video | IC 95 % | Baseline trivial |
|---|---:|---:|---:|
| Estimada (`igho_training_bbps_generated`) | 1,33 | 0,67–2,13 | 1,33 |
| Especialista (`igho_training_bbps_real`) | 2,80 | 1,80–3,73 | 1,33 |

Un *linear probe* (ridge, *leave-one-video-out*) directamente sobre los
embeddings de BiomedCLIP tampoco baja del baseline, lo que indica que, con 15
videos, la señal de BBPS no es extraíble de estos embeddings
(`scripts/finetuning_bbps/probe_bbps.py`).

**Limitaciones:** n = 15 videos; varios valores de BBPS aparecen en un solo
video y no se ven en el entrenamiento de su fold; la selección de checkpoint
se hace sobre el propio fold de validación.

## Estructura del Repositorio

```text
├── train.py                       # Entrenamiento del mapper (ClipCap)
├── test.py                        # Inferencia: genera captions para una carpeta de imágenes
├── predict.py                     # Modelo y decodificación (beam search / top-p)
├── parse_colono.py                # Embeddings con CLIP (ViT-B/32, RN101)
├── parse_colono_biomed.py         # Embeddings con BiomedCLIP
├── clip_weights/                  # Pesos de los encoders: biomedclip_weights.pt, ViT-B-32.pt, RN101.pt
├── experiments_colono/            # CSV de SUN (positivos y negativos con caption)
├── fold/                          # Experimentos 2-fold en SUN
│   ├── 2fold/                     #   baseline (época 14)
│   ├── 2fold_best/                #   checkpoint por val_loss
│   ├── 2fold_hp/                  #   reentrenamiento con nuevos hiperparámetros
│   └── 2fold_hp_f1/               #   checkpoint por F1 (resultado final)
├── igho_inference/                # Inferencia y métricas sobre videos de IGHO
├── igho_training_bbps_generated/  # Fine-tuning BBPS con etiqueta estimada
├── igho_training_bbps_real/       # Fine-tuning BBPS con etiqueta del especialista
├── weights/                       # Embeddings (.pkl) usados en entrenamiento y validación
│   ├── sun_2fold/                 #   SUN baseline: <modelo>/fold_N/{train,val}.pkl
│   ├── sun_2fold_hp/              #   SUN reentrenamiento: <modelo>/fold_N/{train,val}.pkl
│   ├── igho_bbps_generated/       #   IGHO BBPS estimado: group_{a,b}.pkl
│   └── igho_bbps_real/            #   IGHO BBPS especialista: group_{a,b}.pkl
├── weights.zip                    # Copia comprimida de weights/
├── plots_multidataset/            # Estadísticas y figuras de SUN e IGHO
├── scripts/
│   ├── 2fold_models.py            # Pipeline 2-fold: folds, embeddings, entrenamiento, validación
│   ├── evaluate_fold_models.py    # Tablas de métricas (cuadro, reporte, lesión)
│   ├── revalidate_epoch.py        # Re-inferencia con el checkpoint seleccionado
│   ├── validation.py              # val_loss por época
│   ├── inferiencia.py             # Inferencia sobre videos reales
│   ├── avi_to_frame.py            # Extracción de cuadros desde .avi
│   ├── cut_frames.py              # Recorte lateral de cuadros
│   ├── reentrenamiento/           # Selección de época por F1
│   ├── finetuning_bbps/           # Dataset, evaluación y probe de BBPS
│   └── graphics/                  # Gráficas
├── logs/                          # Logs de entrenamiento e inferencia
├── unuse/                         # Scripts y experimentos antiguos
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md                      # Este archivo
```

## Requisitos

### Opción recomendada: Docker

El proyecto incluye un `Dockerfile` basado en PyTorch 2.1.0 con CUDA 11.8 y
cuDNN 8. La imagen instala las dependencias de `requirements.txt`:

- transformers 4.39.3
- open-clip-torch
- CLIP de OpenAI (`git+https://github.com/openai/CLIP.git`)
- NumPy 1.26.4
- pandas
- scikit-learn
- scikit-image 0.20.0
- OpenCV 4.9.0.80
- Pillow
- Matplotlib
- tqdm
- bert-score, evaluate y nltk (métricas de texto)

Para ejecutar los experimentos con GPU se requiere Docker, Docker Compose y
una instalación funcional de NVIDIA Container Toolkit.

### Instalación local

En un entorno Python con PyTorch y CUDA compatibles con el hardware:

```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

La extracción de cuadros requiere además `ffmpeg`/`ffprobe`.

## Instrucciones de Uso

### 1. Preparar los datos y los modelos

1. Colocar SUN Multimodal en `Sun_Multimodal/` (`Train/`, `Train_Negative/`)
   y los CSV de captions en `experiments_colono/experiments_colono/`.
2. Colocar los pesos de los encoders en `clip_weights/`
   (`biomedclip_weights.pt`, `ViT-B-32.pt`, `RN101.pt`). `parse_colono.py` y
   `test.py` cargan CLIP desde esa carpeta y solo descargan los pesos de
   OpenAI si no están. En la inferencia, `test.py` carga BiomedCLIP desde
   Hugging Face (`--biomedclip_model_id`). `ViT-B-16.pt`, `RN50.pt` y
   `RN50x4.pt` no se usan.
3. Extraer los cuadros de los videos de IGHO:

   ```bash
   python scripts/avi_to_frame.py ruta/al/video.avi
   ```

4. Ajustar en `docker-compose.yml` la ruta local de los cuadros, que se monta
   en `/frames` dentro del contenedor.

Los videos clínicos de IGHO no se incluyen en el repositorio y deben
permanecer en una ubicación autorizada para su uso.

### 2. Construir el contenedor

Desde la raíz del proyecto:

```bash
docker compose build
```

### 3. Ejecutar el contenedor

```bash
docker compose up -d
docker compose exec biomedCLIPCAP bash
```

Dentro del contenedor, el repositorio está disponible en `/workspace` y los
cuadros en `/frames`.

### 4. Entrenar y validar en SUN (2-fold)

```bash
python scripts/2fold_models.py \
    --model all --output_root fold/2fold_hp \
    --epochs 6 --bs 16 --lr 1e-5 --warmup_steps 400 \
    --weight_decay 0.01 --dropout 0.1 --num_layers 4 \
    --save_every 1 --gpu 0

python scripts/reentrenamiento/val_metric_per_epoch.py \
    --fold_root fold/2fold_hp --metric f1 --gpu 0

python scripts/revalidate_epoch.py \
    --source_root fold/2fold_hp --output_root fold/2fold_hp_f1 \
    --checkpoint_policy f1 --gpu 0

python scripts/evaluate_fold_models.py --fold_root fold/2fold_hp_f1
```

Los pesos, predicciones y tablas (`table_i_*`, `table_ii_*`) se guardan en la
carpeta indicada en `--output_root`. Más detalle en
[scripts/reentrenamiento/README.md](scripts/reentrenamiento/README.md).

### 5. Ejecutar inferencia sobre videos de IGHO

```bash
python scripts/inferiencia.py \
    --model all --fold all \
    --checkpoints_root fold/2fold_hp --checkpoint_policy f1 \
    --images_root /frames/<video_id> \
    --output_root igho_inference/videos_f1/<video> \
    --gpu 0

python igho_inference/metrics/igho_metrics.py \
    --base_dir igho_inference/videos_f1 \
    --ground_truth_csv igho_inference/igho_dataset_bbps_gt.csv \
    --video_fps_json igho_inference/video_fps.json
```

Cada video genera `predictions.csv` por modelo y fold, y un
`frame_reporte.csv` consolidado.

### 6. Fine-tuning de BBPS

El pipeline completo (construcción del dataset, embeddings por grupo,
fine-tuning desde el checkpoint de SUN, inferencia con
`--stop_token_count 2` y evaluación por video) está documentado en
[scripts/finetuning_bbps/README.md](scripts/finetuning_bbps/README.md).

> Algunos scripts todavía tienen como rutas por defecto `igho/` e
> `igho_training/`; al ejecutarlos, pasar explícitamente las rutas actuales
> (`igho_inference/`, `igho_training_bbps_*`).

## Contacto

- **Autor:** [keyler_sanchez](mail:jhon.68h@gmail.com)
- **GitLab:** [@keyler_sanchez](https://gitlab.com/jhon.68h)
- **Director:** [famarcar@saber.uis.edu.co](mailto:famarcar@saber.uis.edu.co)

## Licencia

El código base de ClipCap se distribuye bajo licencia MIT (ver `LICENSE`,
© 2021 rmokady). Los datos clínicos de IGHO y los pesos entrenados con ellos
deben utilizarse únicamente con autorización de sus autores y respetando las
restricciones de acceso aplicables.

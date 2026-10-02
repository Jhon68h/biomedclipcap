#!/usr/bin/env python3
"""Construye el dataset de fine-tuning de IGHO con BBPS.

Implementa las secciones 3.1 a 3.4 de `reportes/finetuning_bbps_igho.md`:

  3.1  CSV de captions `image_path,caption` a partir de los rangos [start, end]
       anotados en `igho/igho_dataset_copia.csv`, con la caption combinada
       "{report} {report_bbps}."
  3.2  Cap de frames por lesion (200 por defecto) con stride uniforme sobre el
       rango, no los primeros N.
  3.3  Negativos tomados fuera de todos los rangos de lesion del mismo video y con
       el BBPS de ese video en la caption. **El ratio por defecto es 1.0 (1:1), no
       el 10% del plan** -- ver la nota sobre el prior de clase mas abajo.
  3.4  Split 2-fold POR VIDEO (nunca por frame), estratificado por BBPS.

Todo lo que escribe cae bajo --output_root (por defecto `igho_training/`).

Reglas que el plan dejaba abiertas y aqui quedan fijadas
--------------------------------------------------------
* **Solapamiento de rangos** (--overlap_rule, por defecto `first_wins`): un frame
  cubierto por dos lesiones se asigna a la que aparece primero en el CSV. La
  segunda lesion simplemente aporta menos frames; el script avisa por stdout y
  lo registra en build_summary.json. La alternativa `concat` une las dos
  descripciones en una sola caption.
* **Reparto de negativos** (--negative_alloc, por defecto `proportional`):
  proporcional a los positivos que aporta cada video, de forma que la razon
  negativos/positivos sea la misma *dentro* de cada video y ningun video quede sin
  representar. `uniform` reparte la misma cantidad a los 15.
* **Margen de guarda** (--negative_margin, por defecto 300 frames ~ 5 s a 60 fps):
  ningun negativo puede salir de una ventana de +-N frames alrededor de un rango
  anotado. La anotacion solo cubre las lesiones reportadas, asi que un frame pegado
  al borde del segmento probablemente sigue mostrando el polipo; etiquetarlo
  "no polyps" seria un falso negativo metido a mano en el entrenamiento. Con 496
  negativos ese ruido era despreciable; a 1:1 ya pesa.
* **Frames inexistentes**: se escanea el disco y solo se seleccionan frames que
  existen. Los rangos del CSV son aritmeticos y algunos videos tienen menos
  frames extraidos que el `end` anotado.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GT_CSV = "igho/igho_dataset_copia.csv"
DEFAULT_FRAMES_ROOT = "/frames"
DEFAULT_OUTPUT_ROOT = "igho_training"

# El plan (3.3) propone 0.10. Se sube a 1.0 (1:1) a proposito, y la razon es el prior
# de clase, no el volumen de datos:
#
#   SUN, el checkpoint del que se parte ... 49.7% positivo  (50/50, medido en fold/2fold)
#   fine-tuning con 0.10 ................... 90.9% positivo
#   video IGHO real, los 15 videos ......... ~35% positivo
#
# Con 0.10 el fine-tuning empuja la frontera de decision hacia "siempre positivo",
# justo al reves de lo que pide el despliegue, y biomedclip ya sobre-predice positivos
# en IGHO (precision 0.185, especificidad 0.814 en igho/metrics). El motivo que da el
# plan para topar en 10% -que los negativos "ahoguen la senal positiva"- no aplica:
# los negativos LLEVAN el BBPS de su video (3.3), asi que cada uno es un ejemplo
# completo de BBPS. Lo unico que no ensenan son atributos de lesion, y esos ya estan
# pseudo-replicados (27 descripciones unicas repetidas ~184 veces cada una).
# Usa --negative_ratio 0.10 para reproducir el plan tal cual.
DEFAULT_NEGATIVE_RATIO = 1.0
DEFAULT_NEGATIVE_MARGIN = 300

NEGATIVE_TEMPLATE = "This is a colonoscopy frame from a patient with no polyps."
FRAME_RE = re.compile(r"^frame_(?P<number>\d+)\.(?:png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)
BBPS_RE = re.compile(r"BBPS\s+is\s+(\d+)\s*/\s*9", re.IGNORECASE)

# Split propuesto en el plan (3.4), expresado por sufijo del id de video para no
# depender de la fecha completa. Prioriza balancear BBPS -el atributo nuevo y mas
# escaso- por encima del numero de lesiones y de frames.
#   Fold A: BBPS 2, 4, 5, 6, 7, 7, 7, 8   (8 videos, 15 lesiones)
#   Fold B: BBPS 3, 5, 6, 7, 7, 7, 8      (7 videos, 12 lesiones)
DEFAULT_FOLD_GROUPS: "OrderedDict[str, List[str]]" = OrderedDict(
    [
        ("A", ["321", "265", "383", "811", "719", "991", "453", "695"]),
        ("B", ["225252", "093", "674", "211", "225167", "759", "631"]),
    ]
)

# fold_1 entrena con A y valida con B; fold_2 al reves. Cada video aparece exactamente
# una vez como validacion entre los dos folds.
FOLD_DEFINITION: "OrderedDict[str, Dict[str, str]]" = OrderedDict(
    [
        ("fold_1", {"train": "A", "val": "B"}),
        ("fold_2", {"train": "B", "val": "A"}),
    ]
)

META_COLUMNS = [
    "sample_id",
    "video_id",
    "frame",
    "label",
    "lesion_id",
    "bbps",
    "fold_group",
    "image_path",
    "caption",
]

MANIFEST_COLUMNS = [
    "sample_id",
    "video_id",
    "frame",
    "label",
    "lesion_id",
    "bbps",
    "image_path",
    "linked_image_path",
    "caption_gt",
]


# --------------------------------------------------------------------------- utils
def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def uniform_subsample(items: Sequence[object], cap: int) -> List[object]:
    """Submuestreo con stride uniforme conservando ambos extremos.

    Los frames de un segmento son consecutivos a ~60fps: coger los primeros N daria
    N imagenes casi identicas del mismo instante. Con stride uniforme el cap cubre
    todo el rango anotado (3.2 del plan).
    """
    total = len(items)
    if cap <= 0 or total <= cap:
        return list(items)
    if cap == 1:
        return [items[total // 2]]
    picked = sorted({round(index * (total - 1) / (cap - 1)) for index in range(cap)})
    return [items[index] for index in picked]


def parse_bbps_value(text: str) -> Optional[int]:
    match = BBPS_RE.search(text or "")
    return int(match.group(1)) if match else None


def compose_positive_caption(report: str, bbps_text: str) -> str:
    report = " ".join((report or "").split())
    bbps_text = " ".join((bbps_text or "").split()).rstrip(".")
    if report and not report.endswith("."):
        report = f"{report}."
    return f"{report} {bbps_text}." if bbps_text else report


def compose_negative_caption(bbps_text: str) -> str:
    bbps_text = " ".join((bbps_text or "").split()).rstrip(".")
    return f"{NEGATIVE_TEMPLATE} {bbps_text}." if bbps_text else NEGATIVE_TEMPLATE


# ----------------------------------------------------------------------- filesystem
def scan_video_frames(frames_root: Path, video_id: str) -> List[Tuple[int, str]]:
    """Devuelve [(numero_de_frame, nombre_de_archivo)] ordenado, leyendo el disco una vez."""
    video_dir = frames_root / video_id
    if not video_dir.is_dir():
        return []
    found: List[Tuple[int, str]] = []
    with os.scandir(video_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = FRAME_RE.match(entry.name)
            if match:
                found.append((int(match.group("number")), entry.name))
    found.sort(key=lambda item: item[0])
    return found


def synthesize_video_frames(lesions: Sequence[Dict[str, object]], extra_tail: int) -> List[Tuple[int, str]]:
    """Modo --no_scan: asume frames contiguos, solo para previsualizar sin /frames montado."""
    if not lesions:
        return []
    last = max(int(lesion["end"]) for lesion in lesions) + max(0, extra_tail)
    return [(number, f"frame_{number:04d}.png") for number in range(0, last + 1)]


# ------------------------------------------------------------------- base de tiempo
def load_video_fps(path: Optional[str], video_ids: Sequence[str]) -> Dict[str, float]:
    """Lee el mapa {video_id: fps}. Falla si algun video del CSV no tiene valor.

    No se adivina: usar unos fps equivocados desplaza los rangos a otra parte del video
    y el entrenamiento correria sobre frames que no son la lesion, sin ningun error.
    """
    if not path:
        raise SystemExit(
            "[ERROR] --annotation_fps necesita --video_fps_json con los fps reales de cada video.\n"
            "        Genera el mapa con: ffprobe -v error -select_streams v:0 "
            "-show_entries stream=avg_frame_rate -of default=nw=1:nk=1 <video>.avi"
        )
    data = json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))
    faltan = [vid for vid in video_ids if not data.get(vid)]
    if faltan:
        raise SystemExit(
            f"[ERROR] Estos videos no tienen fps en {path}: {faltan}\n"
            "        Rellenalos a mano (probablemente 30) antes de continuar."
        )
    return {vid: float(data[vid]) for vid in video_ids}


def rescale_lesion_ranges(
    lesions_by_video: "OrderedDict[str, List[Dict[str, object]]]",
    video_fps: Dict[str, float],
    annotation_fps: float,
) -> List[Dict[str, object]]:
    """Pasa los numeros de frame del CSV a la base de tiempo real de cada video.

    Los rangos de igho_dataset_copia.csv estan escritos en frames de 60 fps, pero la
    mitad de los videos se grabaron a 30: en esos, el frame anotado N corresponde al
    frame N*30/60 = N/2 del video extraido. Sin esta correccion los rangos de los
    videos de 30 fps caen fuera del video (o peor, caen dentro pero en la escena
    equivocada, como pasaba con 2025-12-01_141231_211).
    """
    cambios: List[Dict[str, object]] = []
    for video_id, lesions in lesions_by_video.items():
        factor = video_fps[video_id] / annotation_fps
        if abs(factor - 1.0) < 1e-9:
            continue
        for lesion in lesions:
            antes = [lesion["start"], lesion["end"]]
            lesion["start"] = int(round(int(lesion["start"]) * factor))
            lesion["end"] = int(round(int(lesion["end"]) * factor))
            cambios.append(
                {
                    "video_id": video_id,
                    "lesion_id": lesion["lesion_id"],
                    "video_fps": video_fps[video_id],
                    "factor": factor,
                    "range_csv": antes,
                    "range_reescalado": [lesion["start"], lesion["end"]],
                }
            )
    return cambios


# ----------------------------------------------------------------------------- split
def resolve_fold_groups(video_ids: Sequence[str], groups: Dict[str, List[str]]) -> Dict[str, str]:
    """Mapea cada id completo de video al grupo A/B a partir del sufijo declarado."""
    assignment: Dict[str, str] = {}
    for group_name, tokens in groups.items():
        for token in tokens:
            matches = [video_id for video_id in video_ids if video_id.endswith(str(token))]
            if not matches:
                raise ValueError(f"El token de split '{token}' (grupo {group_name}) no casa con ningun video del CSV.")
            if len(matches) > 1:
                raise ValueError(f"El token de split '{token}' es ambiguo: casa con {matches}.")
            video_id = matches[0]
            if video_id in assignment:
                raise ValueError(f"El video {video_id} esta declarado en dos grupos del split.")
            assignment[video_id] = group_name
    missing = [video_id for video_id in video_ids if video_id not in assignment]
    if missing:
        raise ValueError(f"Videos del CSV sin grupo asignado en el split: {missing}")
    return assignment


# ------------------------------------------------------------------------ negativos
def allocate_negatives(
    positives_by_video: Dict[str, int],
    available_by_video: Dict[str, int],
    budget: int,
    mode: str,
) -> Dict[str, int]:
    """Reparte `budget` negativos entre los videos respetando su disponibilidad real."""
    video_ids = [vid for vid in positives_by_video if available_by_video.get(vid, 0) > 0]
    if budget <= 0 or not video_ids:
        return {vid: 0 for vid in positives_by_video}

    if mode == "uniform":
        weights = {vid: 1.0 for vid in video_ids}
    else:
        total_positive = sum(positives_by_video[vid] for vid in video_ids) or 1
        weights = {vid: positives_by_video[vid] / total_positive for vid in video_ids}

    weight_sum = sum(weights.values()) or 1.0
    exact = {vid: budget * weights[vid] / weight_sum for vid in video_ids}
    alloc = {vid: min(int(exact[vid]), available_by_video[vid]) for vid in video_ids}

    # Reparto del resto por mayor parte fraccionaria, respetando la disponibilidad.
    remainder = budget - sum(alloc.values())
    order = sorted(video_ids, key=lambda vid: (-(exact[vid] - int(exact[vid])), vid))
    while remainder > 0:
        progressed = False
        for vid in order:
            if remainder <= 0:
                break
            if alloc[vid] < available_by_video[vid]:
                alloc[vid] += 1
                remainder -= 1
                progressed = True
        if not progressed:  # ya no cabe ni un negativo mas en ningun video
            break

    for vid in positives_by_video:
        alloc.setdefault(vid, 0)
    return alloc


# ------------------------------------------------------------------------------ main
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construye train/val por fold del fine-tuning IGHO+BBPS (plan 3.1-3.4).",
    )
    parser.add_argument("--gt_csv", default=DEFAULT_GT_CSV)
    parser.add_argument(
        "--frames_root",
        default=DEFAULT_FRAMES_ROOT,
        help="Raiz con un subdirectorio por video. Dentro del contenedor es /frames.",
    )
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cap_per_lesion", type=int, default=200, help="Cap de frames por lesion (3.2).")
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=DEFAULT_NEGATIVE_RATIO,
        help=(
            "Negativos como fraccion de los positivos. 1.0 = 1:1 (replica el prior 50/50 "
            "con el que se entreno SUN). El plan (3.3) proponia 0.10; ver la nota junto a "
            "DEFAULT_NEGATIVE_RATIO."
        ),
    )
    parser.add_argument(
        "--negative_margin",
        type=int,
        default=DEFAULT_NEGATIVE_MARGIN,
        help=(
            "Margen de guarda en frames alrededor de cada rango anotado. Ningun negativo "
            "sale de [start - N, end + N]. Por defecto 300 (~5 s a 60 fps). 0 lo desactiva."
        ),
    )
    parser.add_argument("--negative_alloc", choices=["proportional", "uniform"], default="proportional")
    parser.add_argument("--overlap_rule", choices=["first_wins", "concat"], default="first_wins")
    parser.add_argument(
        "--annotation_fps",
        type=float,
        default=0.0,
        help=(
            "Base de tiempo en la que estan escritos los numeros de frame del CSV (p.ej. 60). "
            "Cada rango se reescala a los fps reales de su video. 0 = desactivado."
        ),
    )
    parser.add_argument(
        "--video_fps_json",
        default="igho/video_fps.json",
        help='JSON {"video_id": fps}. Obligatorio si se usa --annotation_fps.',
    )
    parser.add_argument(
        "--split_json",
        default=None,
        help='JSON opcional {"A": [...ids o sufijos...], "B": [...]} para sobreescribir el split de 3.4.',
    )
    parser.add_argument(
        "--no_scan",
        action="store_true",
        help="No lee el disco: asume frames contiguos. Solo para previsualizar fuera del contenedor.",
    )
    parser.add_argument(
        "--allow_missing_videos",
        action="store_true",
        help=(
            "Sigue adelante aunque algun video del CSV no tenga frames en --frames_root. "
            "Por defecto es un error: un video vacio se cae del split y rompe el balance de "
            "BBPS de 3.4 sin que se note en las metricas."
        ),
    )
    parser.add_argument(
        "--no_symlinks",
        action="store_true",
        help="No crea el arbol de symlinks de validacion que consume test.py --images_root.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Borra output_root/data y output_root/folds antes de escribir. NUNCA toca "
            "weights/igho_bbps_<etiqueta>/, que es donde deben vivir los .pkl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gt_csv = resolve_repo_path(args.gt_csv)
    frames_root = Path(args.frames_root).expanduser()
    output_root = resolve_repo_path(args.output_root)
    data_dir = output_root / "data"
    folds_dir = output_root / "folds"

    if not gt_csv.is_file():
        raise SystemExit(f"[ERROR] No existe el CSV de ground truth: {gt_csv}")
    if not args.no_scan and not frames_root.is_dir():
        raise SystemExit(
            f"[ERROR] No existe --frames_root: {frames_root}. Dentro del contenedor debe ser /frames "
            "(docker-compose monta ahi los frames de IGHO). Usa --no_scan para previsualizar sin ellos."
        )

    if args.overwrite:
        # Los .pkl de embeddings tardan ~30 min en generarse y NO se regeneran aqui.
        # Antes vivian en output_root/data, que este bloque borra: cada rebuild se los
        # comia en silencio y luego train.py fallaba (o peor, se reentrenaba sobre un
        # split que ya no correspondia a los pkl). Van en weights/igho_bbps_<etiqueta>/,
        # que este script nunca toca; si aun quedan dentro de data/ o folds/, se avisa y se
        # aborta en vez de borrarlos.
        en_peligro = sorted(
            path.as_posix()
            for base in (data_dir, folds_dir)
            if base.exists()
            for path in base.rglob("*.pkl")
        )
        if en_peligro:
            raise SystemExit(
                "[ERROR] --overwrite iba a borrar estos .pkl de embeddings:\n"
                + "\n".join(f"          {x}" for x in en_peligro)
                + f"\n        Muevelos a {(REPO_ROOT / 'weights').as_posix()}/igho_bbps_<etiqueta>/ (este script\n"
                "        no toca esa carpeta) y vuelve a lanzar. Regenerarlos cuesta ~30 min de GPU."
            )
        for path in (data_dir, folds_dir):
            if path.exists():
                shutil.rmtree(path)

    rows = read_csv_rows(gt_csv)
    print(f"[INFO] Filas (lesiones) en {gt_csv}: {len(rows)}")

    # ---------------------------------------------------------------- 1. agrupar por video
    lesions_by_video: "OrderedDict[str, List[Dict[str, object]]]" = OrderedDict()
    for index, row in enumerate(rows):
        video_id = (row.get("id") or "").strip()
        if not video_id:
            continue
        bbps_text = (row.get("report_bbps") or "").strip()
        bbps_value = parse_bbps_value(bbps_text)
        if bbps_value is None:
            raise SystemExit(f"[ERROR] Fila {index} ({video_id}) sin BBPS parseable en report_bbps: {bbps_text!r}")
        lesions_by_video.setdefault(video_id, []).append(
            {
                "lesion_id": f"{video_id}#L{len(lesions_by_video.get(video_id, [])) + 1}",
                "csv_row": index,
                "start": int(float(row["start"])),
                "end": int(float(row["end"])),
                "report": (row.get("report") or "").strip(),
                "bbps_text": bbps_text,
                "bbps": bbps_value,
            }
        )

    video_ids = list(lesions_by_video)
    print(f"[INFO] Videos: {len(video_ids)} | lesiones: {sum(len(v) for v in lesions_by_video.values())}")

    # El BBPS es una propiedad del video: si una fila lo contradice, el split y las
    # metricas por video (n=15) dejan de tener sentido.
    bbps_by_video: Dict[str, int] = {}
    for video_id, lesions in lesions_by_video.items():
        values = {lesion["bbps"] for lesion in lesions}
        if len(values) > 1:
            raise SystemExit(f"[ERROR] {video_id} tiene BBPS contradictorio entre sus lesiones: {sorted(values)}")
        bbps_by_video[video_id] = lesions[0]["bbps"]

    # ------------------------------------------------- 1b. reescalado de base de tiempo
    fps_changes: List[Dict[str, object]] = []
    if args.annotation_fps and args.annotation_fps > 0:
        video_fps = load_video_fps(args.video_fps_json, video_ids)
        fps_changes = rescale_lesion_ranges(lesions_by_video, video_fps, args.annotation_fps)
        reescalados = sorted({str(c["video_id"]) for c in fps_changes})
        print(
            f"[INFO] Rangos reescalados de {args.annotation_fps:g} fps a los fps reales: "
            f"{len(fps_changes)} lesiones en {len(reescalados)} videos."
        )
        for vid in reescalados:
            print(f"       {vid} -> x{video_fps[vid] / args.annotation_fps:g}")

    # ---------------------------------------------------------------- 2. split por video
    groups_spec = DEFAULT_FOLD_GROUPS
    if args.split_json:
        with resolve_repo_path(args.split_json).open("r", encoding="utf-8") as handle:
            groups_spec = OrderedDict(json.load(handle))
    group_by_video = resolve_fold_groups(video_ids, groups_spec)

    # ---------------------------------------------------------------- 3. seleccion de frames
    samples: List[Dict[str, object]] = []
    empty_videos: List[str] = []
    empty_lesions: List[Dict[str, object]] = []
    per_video_stats: List[Dict[str, object]] = []
    overlaps: List[Dict[str, object]] = []
    negatives_pool: Dict[str, List[Tuple[int, str]]] = {}
    positives_by_video: Dict[str, int] = {}

    for video_id in video_ids:
        lesions = lesions_by_video[video_id]
        if args.no_scan:
            frames = synthesize_video_frames(lesions, extra_tail=5000)
        else:
            frames = scan_video_frames(frames_root, video_id)
        if not frames:
            print(f"[WARN] {video_id}: 0 frames encontrados en {frames_root / video_id}.")
            empty_videos.append(video_id)

        claimed: set = set()
        video_positive_count = 0
        lesion_stats: List[Dict[str, object]] = []

        for lesion in lesions:
            start, end = lesion["start"], lesion["end"]
            in_range = [(number, name) for number, name in frames if start <= number <= end]
            available = len(in_range)
            free = [(number, name) for number, name in in_range if number not in claimed]
            stolen = available - len(free)
            if stolen:
                overlaps.append(
                    {
                        "video_id": video_id,
                        "lesion_id": lesion["lesion_id"],
                        "frames_taken_by_earlier_lesion": stolen,
                        "rule": args.overlap_rule,
                    }
                )
                print(
                    f"[WARN] Solapamiento en {video_id}: {lesion['lesion_id']} pierde {stolen} frames "
                    f"ya asignados a una lesion anterior (regla={args.overlap_rule})."
                )
            pool = in_range if args.overlap_rule == "concat" else free
            selected = uniform_subsample(pool, args.cap_per_lesion)
            caption = compose_positive_caption(lesion["report"], lesion["bbps_text"])

            for number, name in selected:
                claimed.add(number)
                samples.append(
                    {
                        "video_id": video_id,
                        "frame": number,
                        "label": "positive",
                        "lesion_id": lesion["lesion_id"],
                        "bbps": lesion["bbps"],
                        "fold_group": group_by_video[video_id],
                        "image_path": (frames_root / video_id / name).as_posix(),
                        "caption": caption,
                    }
                )
            if not selected:
                # El rango anotado no tiene NI UN frame en disco. Casi siempre significa
                # que la anotacion y la extraccion no estan en la misma base de tiempo
                # (anotado a 60 fps, extraido a 30) o que la extraccion se trunco.
                empty_lesions.append(
                    {
                        "video_id": video_id,
                        "lesion_id": lesion["lesion_id"],
                        "range": [start, end],
                        "frames_expected": end - start + 1,
                        "last_frame_on_disk": frames[-1][0] if frames else 0,
                    }
                )
            video_positive_count += len(selected)
            lesion_stats.append(
                {
                    "lesion_id": lesion["lesion_id"],
                    "range": [start, end],
                    "frames_in_range_arithmetic": end - start + 1,
                    "frames_in_range_on_disk": available,
                    "frames_selected": len(selected),
                }
            )

        # Negativos candidatos: fuera de TODOS los rangos anotados de este video, con
        # un margen de guarda a cada lado (ver --negative_margin).
        ranges = [(lesion["start"], lesion["end"]) for lesion in lesions]
        margin = max(0, int(args.negative_margin))
        strictly_outside = [
            (number, name)
            for number, name in frames
            if not any(start <= number <= end for start, end in ranges)
        ]
        outside = [
            (number, name)
            for number, name in strictly_outside
            if not any(start - margin <= number <= end + margin for start, end in ranges)
        ]
        dropped_by_margin = len(strictly_outside) - len(outside)
        negatives_pool[video_id] = outside
        positives_by_video[video_id] = video_positive_count

        per_video_stats.append(
            {
                "video_id": video_id,
                "bbps": bbps_by_video[video_id],
                "fold_group": group_by_video[video_id],
                "n_lesions": len(lesions),
                "frames_on_disk": len(frames),
                "positives_selected": video_positive_count,
                "negatives_available": len(outside),
                "negatives_dropped_by_margin": dropped_by_margin,
                "lesions": lesion_stats,
            }
        )

    if empty_lesions and not args.allow_missing_videos:
        detalle = "\n".join(
            f"          {e['video_id']:26} rango {str(e['range']):>18} "
            f"(esperados {e['frames_expected']}, ultimo frame en disco: {e['last_frame_on_disk']})"
            for e in empty_lesions
        )
        raise SystemExit(
            f"[ERROR] {len(empty_lesions)} de {sum(len(v) for v in lesions_by_video.values())} "
            "lesiones no tienen NI UN frame en disco dentro de su rango anotado:\n"
            f"{detalle}\n"
            "        Construir el dataset asi produce un split mutilado y sin avisar: se caen\n"
            "        videos enteros, se rompe el balance de BBPS de 3.4, y el entrenamiento\n"
            "        corre sobre la mitad de los datos como si nada.\n"
            "        Causas tipicas: (a) la anotacion esta en base de tiempo de 60 fps y el\n"
            "        video se extrajo a 30 -> los numeros de frame van al doble; (b) la\n"
            "        extraccion del .avi se trunco (comparar con `ffprobe -count_frames`).\n"
            "        Usa --allow_missing_videos SOLO si aceptas entrenar sin esas lesiones."
        )

    if empty_videos and not args.allow_missing_videos:
        raise SystemExit(
            "[ERROR] Estos videos del CSV no tienen ningun frame en "
            f"{frames_root}: {empty_videos}\n"
            "        El split 2-fold de 3.4 esta balanceado por BBPS con los 15 videos; si "
            "falta alguno,\n"
            "        el balance se rompe en silencio. Revisa que la copia de frames haya "
            "terminado y que\n"
            "        la estructura sea <frames_root>/<video_id>/frame_NNNN.png (ojo con un "
            "'frames/' anidado).\n"
            "        Usa --allow_missing_videos si de verdad quieres continuar sin ellos."
        )

    total_positives = sum(positives_by_video.values())
    negative_budget = int(round(total_positives * args.negative_ratio))
    alloc = allocate_negatives(
        positives_by_video,
        {vid: len(pool) for vid, pool in negatives_pool.items()},
        negative_budget,
        args.negative_alloc,
    )

    allocated = sum(alloc.values())
    if allocated < negative_budget:
        print(
            f"[WARN] Solo caben {allocated} negativos de los {negative_budget} pedidos: "
            "los videos se quedaron sin frames candidatos. Baja --negative_ratio o "
            "--negative_margin si necesitas el presupuesto completo."
        )
    for video_id in video_ids:
        quota = alloc.get(video_id, 0)
        if quota <= 0:
            print(f"[WARN] {video_id}: 0 negativos asignados (candidatos disponibles: {len(negatives_pool[video_id])}).")
            continue
        caption = compose_negative_caption(lesions_by_video[video_id][0]["bbps_text"])
        for number, name in uniform_subsample(negatives_pool[video_id], quota):
            samples.append(
                {
                    "video_id": video_id,
                    "frame": number,
                    "label": "negative",
                    "lesion_id": "",
                    "bbps": bbps_by_video[video_id],
                    "fold_group": group_by_video[video_id],
                    "image_path": (frames_root / video_id / name).as_posix(),
                    "caption": caption,
                }
            )

    for stats in per_video_stats:
        stats["negatives_selected"] = alloc.get(stats["video_id"], 0)

    samples.sort(key=lambda item: (item["video_id"], item["label"], item["frame"]))
    for sample_id, sample in enumerate(samples):
        sample["sample_id"] = sample_id

    total_negatives = sum(1 for s in samples if s["label"] == "negative")
    print(
        f"[INFO] Positivos: {total_positives} | negativos: {total_negatives} "
        f"({(total_negatives / total_positives * 100 if total_positives else 0):.1f}% de los positivos) "
        f"| total: {len(samples)}"
    )

    write_csv(data_dir / "frames_index.csv", samples, META_COLUMNS)
    write_csv(
        data_dir / "split_by_video.csv",
        [
            {
                "video_id": stats["video_id"],
                "bbps": stats["bbps"],
                "fold_group": stats["fold_group"],
                "n_lesions": stats["n_lesions"],
                "frames_on_disk": stats["frames_on_disk"],
                "positives_selected": stats["positives_selected"],
                "negatives_selected": stats["negatives_selected"],
                "negatives_available": stats["negatives_available"],
                "negatives_dropped_by_margin": stats["negatives_dropped_by_margin"],
            }
            for stats in per_video_stats
        ],
        [
            "video_id",
            "bbps",
            "fold_group",
            "n_lesions",
            "frames_on_disk",
            "positives_selected",
            "negatives_selected",
            "negatives_available",
            "negatives_dropped_by_margin",
        ],
    )

    # ---------------------------------------------------------------- 4. escribir folds
    fold_summaries: List[Dict[str, object]] = []
    for fold_name, spec in FOLD_DEFINITION.items():
        fold_dir = folds_dir / fold_name
        train_rows = [s for s in samples if s["fold_group"] == spec["train"]]
        val_rows = [s for s in samples if s["fold_group"] == spec["val"]]

        write_csv(fold_dir / "train.csv", train_rows, ["image_path", "caption"])
        write_csv(fold_dir / "val.csv", val_rows, ["image_path", "caption"])
        write_csv(fold_dir / "train_meta.csv", train_rows, META_COLUMNS)
        write_csv(fold_dir / "val_meta.csv", val_rows, META_COLUMNS)
        for subdir in ("data", "train", "inference"):
            (fold_dir / subdir).mkdir(parents=True, exist_ok=True)

        # test.py recibe un directorio, no un CSV: se materializa la validacion como
        # symlinks con nombre unico (los frames se llaman igual en todos los videos).
        manifest: List[Dict[str, object]] = []
        val_images_dir = fold_dir / "inference" / "val_images"
        if not args.no_symlinks and not args.no_scan:
            if val_images_dir.exists():
                shutil.rmtree(val_images_dir)
            val_images_dir.mkdir(parents=True, exist_ok=True)
        for row in val_rows:
            source = Path(str(row["image_path"]))
            linked = val_images_dir / f"{int(row['sample_id']):06d}_{source.name}"
            if not args.no_symlinks and not args.no_scan:
                if not source.exists():
                    continue
                if not linked.exists():
                    linked.symlink_to(source)
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "video_id": row["video_id"],
                    "frame": row["frame"],
                    "label": row["label"],
                    "lesion_id": row["lesion_id"],
                    "bbps": row["bbps"],
                    "image_path": row["image_path"],
                    "linked_image_path": linked.as_posix(),
                    "caption_gt": row["caption"],
                }
            )
        write_csv(fold_dir / "inference" / "val_manifest.csv", manifest, MANIFEST_COLUMNS)

        train_videos = sorted({str(r["video_id"]) for r in train_rows})
        val_videos = sorted({str(r["video_id"]) for r in val_rows})
        bbps_in_train = sorted({bbps_by_video[v] for v in train_videos})
        unseen = sorted(
            {v: bbps_by_video[v] for v in val_videos if bbps_by_video[v] not in bbps_in_train}.items()
        )
        fold_summaries.append(
            {
                "fold": fold_name,
                "train_group": spec["train"],
                "val_group": spec["val"],
                "train_videos": train_videos,
                "val_videos": val_videos,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "train_positives": sum(1 for r in train_rows if r["label"] == "positive"),
                "train_negatives": sum(1 for r in train_rows if r["label"] == "negative"),
                "val_positives": sum(1 for r in val_rows if r["label"] == "positive"),
                "val_negatives": sum(1 for r in val_rows if r["label"] == "negative"),
                "bbps_in_train": bbps_in_train,
                # Limitacion declarada en 3.4: BBPS 2/3/4 tienen un solo video, asi que
                # en el fold donde son validacion el modelo nunca los vio entrenando.
                "val_videos_with_bbps_unseen_in_train": [
                    {"video_id": vid, "bbps": value} for vid, value in unseen
                ],
                "val_manifest": (fold_dir / "inference" / "val_manifest.csv").as_posix(),
                "val_images_dir": val_images_dir.as_posix(),
            }
        )
        print(
            f"[INFO] {fold_name}: train={len(train_rows)} (grupo {spec['train']}, {len(train_videos)} videos) | "
            f"val={len(val_rows)} (grupo {spec['val']}, {len(val_videos)} videos) | "
            f"BBPS no vistos en train: {[u['bbps'] for u in fold_summaries[-1]['val_videos_with_bbps_unseen_in_train']]}"
        )

    # ---------------------------------------------------------------- 5. baseline trivial
    bbps_values = [bbps_by_video[v] for v in video_ids]
    constant = median(bbps_values)
    baseline_mae = sum(abs(value - constant) for value in bbps_values) / len(bbps_values)
    print(f"[INFO] Baseline trivial (predecir siempre BBPS {constant:g}/9): MAE = {baseline_mae:.4f} sobre n={len(bbps_values)} videos")

    summary = {
        "gt_csv": gt_csv.as_posix(),
        "frames_root": frames_root.as_posix(),
        "output_root": output_root.as_posix(),
        "scanned_filesystem": not args.no_scan,
        "params": {
            "cap_per_lesion": args.cap_per_lesion,
            "negative_ratio": args.negative_ratio,
            "negative_alloc": args.negative_alloc,
            "negative_margin": args.negative_margin,
            "overlap_rule": args.overlap_rule,
            "annotation_fps": args.annotation_fps,
        },
        "fps_rescaling": fps_changes,
        "totals": {
            "videos": len(video_ids),
            "lesions": sum(len(v) for v in lesions_by_video.values()),
            "positives": total_positives,
            "negatives": total_negatives,
            "negative_budget_requested": negative_budget,
            "negatives_dropped_by_margin": sum(
                int(stats["negatives_dropped_by_margin"]) for stats in per_video_stats
            ),
            "samples": len(samples),
        },
        "caption_format": {
            "positive": "{report} {report_bbps}.",
            "negative": f"{NEGATIVE_TEMPLATE} {{report_bbps}}.",
            "sentences_per_caption": 2,
            "nota": "test.py necesita --stop_token_count 2 o el beam search corta antes del BBPS.",
        },
        "bbps_baseline": {
            "constant_prediction": constant,
            "mae": baseline_mae,
            "n_videos": len(bbps_values),
            "nota": "Cualquier bbps_mae del modelo debe quedar por debajo de este numero (3.8).",
        },
        "bbps_distribution": {
            str(value): sorted(v for v in video_ids if bbps_by_video[v] == value)
            for value in sorted(set(bbps_values))
        },
        "empty_videos": empty_videos,
        "empty_lesions": empty_lesions,
        "lesions_with_truncated_range": [
            {
                "video_id": stats["video_id"],
                "lesion_id": lesion["lesion_id"],
                "range": lesion["range"],
                "frames_expected": lesion["frames_in_range_arithmetic"],
                "frames_on_disk": lesion["frames_in_range_on_disk"],
            }
            for stats in per_video_stats
            for lesion in stats["lesions"]
            if lesion["frames_in_range_on_disk"] < lesion["frames_in_range_arithmetic"]
        ],
        "overlaps": overlaps,
        "folds": fold_summaries,
        "videos": per_video_stats,
    }
    summary_path = data_dir / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Resumen: {summary_path}")
    print(f"[OK] Dataset construido en: {output_root}")


if __name__ == "__main__":
    main()

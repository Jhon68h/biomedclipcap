#!/usr/bin/env python3
"""Solo inferencia sobre frames reales reutilizando test.py."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SCRIPT = REPO_ROOT / "test.py"

DEFAULT_IMAGES_ROOT = "igho/frames/2025-03-17_094605_719"
DEFAULT_DATASET_ROOT = "igho/video_1"
DEFAULT_CHECKPOINTS_ROOT = "fold/2fold"
DEFAULT_OUTPUT_ROOT = "inferiencia"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MODEL_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "biomedclip": {
        "subdir": "biomedclip",
        "encoder": "biomedclip",
        "biomedclip_model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "openai_clip_name": None,
    },
    "vit": {
        "subdir": "vit",
        "encoder": "vit",
        "biomedclip_model_id": None,
        "openai_clip_name": "ViT-B/32",
    },
    "resnet101": {
        "subdir": "resnet",
        "encoder": "rn101",
        "biomedclip_model_id": None,
        "openai_clip_name": "RN101",
    },
}

MODEL_ALIAS = {
    "biomedclip": "biomedclip",
    "vit": "vit",
    "resnet101": "resnet101",
    "resnet": "resnet101",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline solo inferencia (2-fold).")
    parser.add_argument(
        "--model",
        default="biomedclip",
        choices=sorted([*MODEL_ALIAS.keys(), "all"]),
        help="biomedclip, vit, resnet101/resnet o all.",
    )
    parser.add_argument(
        "--fold",
        default="all",
        choices=["1", "2", "all"],
        help="Fold a usar: 1, 2 o all.",
    )
    parser.add_argument("--images_root", default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--dataset_root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoints_root", default=DEFAULT_CHECKPOINTS_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint manual para un solo model/fold.")
    parser.add_argument(
        "--checkpoint_policy",
        default="best",
        choices=["best", "latest", "epoch", "f1"],
        help=(
            "Como elegir el checkpoint: 'f1' (mayor metrica de deteccion segun "
            "val_task_metric_per_epoch.csv, criterio recomendado; ver §2 de "
            "reportes/reentrenamiento.md), 'best' (menor val_loss, por defecto "
            "historico), 'latest' (ultima epoca, comportamiento antiguo y "
            "sobreajustado) o 'epoch' (usa --epoch)."
        ),
    )
    parser.add_argument(
        "--f1_metric",
        default="f1",
        help="Columna a maximizar con --checkpoint_policy f1 (f1, recall, lesion_detection_rate_50pct...).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoca a usar cuando --checkpoint_policy epoch.",
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, por ejemplo 0 o 1.")
    parser.add_argument("--frame_start", type=int, default=None, help="Frame inicial (incluyente).")
    parser.add_argument("--frame_end", type=int, default=None, help="Frame final (incluyente).")
    parser.add_argument(
        "--frame_window",
        action="append",
        default=[],
        help="Ventana por prefijo: <prefijo>:<inicio>-<fin>. Repetir para multiples.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Cantidad de imagenes codificadas por el encoder CLIP en cada lote (ver test.py --batch_size).",
    )
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_token", type=str, default=".")

    parser.add_argument(
        "--video_id",
        default=None,
        help=(
            "id del video en --bbps_csv (columna 'id', ej. 2025-03-17_094605_719). "
            "Si se pasa, se agrega la frase de BBPS al final de cada reporte generado "
            "(frame_reporte.csv). El modelo NO genera el BBPS; se conoce de antemano y "
            "se concatena como texto despues de la generacion, sin tocar el modelo."
        ),
    )
    parser.add_argument(
        "--bbps_csv",
        default="igho/igho_dataset_copia.csv",
        help="CSV con columnas 'id' y 'report_bbps' (usado solo si se pasa --video_id).",
    )

    parser.add_argument("--overwrite", action="store_true", help="Reemplaza output_root si existe.")
    parser.add_argument("--dry_run", action="store_true", help="Solo arma comandos, no ejecuta inferencia.")
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def select_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return ["biomedclip", "vit", "resnet101"]
    return [MODEL_ALIAS[model_arg]]


def select_folds(fold_arg: str) -> List[str]:
    if fold_arg == "all":
        return ["fold_1", "fold_2"]
    return [f"fold_{int(fold_arg)}"]


def build_env(gpu: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def extract_prefix_and_frame(path: Path) -> Optional[Tuple[str, int]]:
    stem = path.stem
    match = re.match(r"^(.+?)_(\d+)$", stem)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_frame_windows(args: argparse.Namespace) -> List[Dict[str, object]]:
    if not args.frame_window:
        if args.frame_start is None and args.frame_end is None:
            return []
        if args.frame_start is None or args.frame_end is None:
            raise ValueError("--frame_start y --frame_end deben usarse juntos.")
        if args.frame_start > args.frame_end:
            raise ValueError("--frame_start no puede ser mayor que --frame_end.")
        return [{"prefix": None, "start": int(args.frame_start), "end": int(args.frame_end)}]

    windows: List[Dict[str, object]] = []
    for raw in args.frame_window:
        text = str(raw).strip()
        match = re.match(r"^([^:]+):(\d+)-(\d+)$", text)
        if not match:
            raise ValueError(f"Formato invalido en --frame_window: {text}")
        prefix = match.group(1).strip()
        start = int(match.group(2))
        end = int(match.group(3))
        if start > end:
            raise ValueError(f"Rango invalido en --frame_window: {text}")
        windows.append({"prefix": prefix, "start": start, "end": end})
    return windows


def matches_any_window(prefix: str, frame: int, windows: Sequence[Dict[str, object]]) -> bool:
    for window in windows:
        window_prefix = window.get("prefix")
        start = int(window["start"])
        end = int(window["end"])
        if window_prefix is not None and str(window_prefix) != prefix:
            continue
        if start <= frame <= end:
            return True
    return False


def select_frames_in_windows(images_root: Path, selected_root: Path, windows: Sequence[Dict[str, object]]) -> List[Path]:

    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)

    selected: List[Path] = []
    selected_sources = set()
    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        parsed = extract_prefix_and_frame(image_path)
        if parsed is None:
            continue
        prefix, frame_number = parsed
        if not matches_any_window(prefix, frame_number, windows):
            continue

        source = image_path.resolve()
        if source in selected_sources:
            continue
        selected_sources.add(source)

        dst = selected_root / image_path.name
        try:
            dst.symlink_to(source)
        except Exception:
            shutil.copy2(image_path, dst)
        selected.append(dst)

    if not selected:
        raise FileNotFoundError(f"No se encontraron frames para las ventanas dadas en {images_root}")
    return selected


def select_all_frames(images_root: Path, selected_root: Path) -> List[Path]:
    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)

    selected: List[Path] = []
    selected_sources = set()
    for image_path in sorted(images_root.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        source = image_path.resolve()
        if source in selected_sources:
            continue
        selected_sources.add(source)

        dst = selected_root / image_path.name
        try:
            dst.symlink_to(source)
        except Exception:
            shutil.copy2(image_path, dst)
        selected.append(dst)

    if not selected:
        raise FileNotFoundError(f"No se encontraron imagenes en {images_root}")
    return selected


def select_frames_by_prefix_folders(
    dataset_root: Path,
    selected_root: Path,
    windows: Sequence[Dict[str, object]],
) -> List[Path]:
    if selected_root.exists():
        shutil.rmtree(selected_root)
    selected_root.mkdir(parents=True, exist_ok=True)

    selected: List[Path] = []
    selected_sources = set()

    for window in windows:
        window_prefix = window.get("prefix")
        start = int(window["start"])
        end = int(window["end"])

        if window_prefix is None:
            continue
        source_dir = dataset_root / f"{window_prefix}_frames"
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"No existe carpeta para prefijo {window_prefix}: {source_dir}")

        for image_path in sorted(source_dir.rglob("*")):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            parsed = extract_prefix_and_frame(image_path)
            if parsed is None:
                continue
            prefix, frame_number = parsed
            if prefix != str(window_prefix):
                continue
            if frame_number < start or frame_number > end:
                continue

            source = image_path.resolve()
            if source in selected_sources:
                continue
            selected_sources.add(source)

            dst = selected_root / image_path.name
            try:
                dst.symlink_to(source)
            except Exception:
                shutil.copy2(image_path, dst)
            selected.append(dst)

    if not selected:
        raise FileNotFoundError(f"No se encontraron frames para las ventanas dadas en {dataset_root}")
    return selected


def extract_frame_id_from_image_path(image_path: str) -> Optional[str]:
    stem = Path((image_path or "").strip()).stem
    if re.match(r"^.+_\d+$", stem):
        return stem
    return None


def read_predictions_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_frame_report_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame", "reporte_medico"])
        writer.writeheader()
        writer.writerows(rows)


def frame_sort_key(frame_id: str) -> Tuple[str, int]:
    match = re.match(r"^(.+?)_(\d+)$", frame_id)
    if not match:
        return frame_id, -1
    return match.group(1), int(match.group(2))


def predictions_to_frame_reports(pred_rows: List[Dict[str, str]], bbps_text: str = "") -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in pred_rows:
        frame_id = extract_frame_id_from_image_path(str(row.get("image_path", "")))
        if frame_id is None:
            continue
        report = str(row.get("generated_caption", "")).strip()
        if bbps_text:
            report = append_bbps_sentence(report, bbps_text)
        rows.append(
            {
                "frame": frame_id,
                "reporte_medico": report,
            }
        )
    rows.sort(key=lambda item: frame_sort_key(item["frame"]))
    return rows


def append_bbps_sentence(report: str, bbps_text: str) -> str:
    """Agrega la frase de BBPS al final del reporte generado, como texto plano.

    El BBPS ya se conoce con certeza (viene del CSV del especialista, no de la
    imagen), asi que no tiene sentido pedirle al modelo que lo "genere" -- eso
    requeriria reentrenar con una senal que hoy no existe en los datos de SUN.
    Se concatena despues de generar, sin tocar el modelo ni el prefix.
    """
    report = report.strip()
    bbps_text = bbps_text.strip().rstrip(".")
    if not bbps_text:
        return report
    if not report:
        return f"{bbps_text}."
    if not report.endswith((".", "!", "?")):
        report += "."
    return f"{report} {bbps_text}."


def load_bbps_lookup(bbps_csv: Path) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    if not bbps_csv.exists():
        return lookup
    with bbps_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = str(row.get("id", "")).strip()
            bbps = str(row.get("report_bbps", "")).strip()
            if video_id and bbps and video_id not in lookup:
                lookup[video_id] = bbps
    return lookup


def extract_epoch(filename: str, prefix: str) -> Optional[int]:
    match = re.match(rf"^{re.escape(prefix)}-(\d+)\.pt$", filename)
    if not match:
        return None
    return int(match.group(1))


def list_checkpoints(train_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe directorio de checkpoints: {train_dir}")

    candidates: List[Tuple[int, Path]] = []
    for ckpt in train_dir.glob(f"{prefix}-*.pt"):
        epoch = extract_epoch(ckpt.name, prefix)
        if epoch is not None:
            candidates.append((epoch, ckpt.resolve()))

    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints con prefijo {prefix} en {train_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates


def find_latest_checkpoint(train_dir: Path, prefix: str) -> Path:
    return list_checkpoints(train_dir, prefix)[-1][1]


def find_checkpoint_for_epoch(train_dir: Path, prefix: str, epoch: int) -> Path:
    for candidate_epoch, path in list_checkpoints(train_dir, prefix):
        if candidate_epoch == epoch:
            return path
    available = ", ".join(str(e) for e, _ in list_checkpoints(train_dir, prefix))
    raise FileNotFoundError(
        f"No existe checkpoint para la epoca {epoch} en {train_dir}. Disponibles: {available}"
    )


def find_best_checkpoint(fold_dir: Path, train_dir: Path, prefix: str) -> Tuple[Path, int, Optional[float]]:
    """Elige el checkpoint con menor val_loss segun val_loss_per_epoch.csv.

    Motivo: el entrenamiento sobreajusta muy temprano (la val_loss toca su
    minimo en la epoca 0-1 y luego crece ~2x hasta la epoca 14 en los 3
    modelos y ambos folds). Usar 'el ultimo checkpoint' desplegaba justamente
    el modelo mas sobreajustado. Si no existe el CSV, cae a la ultima epoca.
    """
    val_loss_csv = fold_dir / "val_loss_per_epoch.csv"
    if not val_loss_csv.exists():
        print(f"[WARN] No existe {val_loss_csv}; se usara el ultimo checkpoint.")
        path = find_latest_checkpoint(train_dir, prefix)
        return path, extract_epoch(path.name, prefix) or -1, None

    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    with val_loss_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                epoch = int(float(row["epoch"]))
                loss = float(row["val_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_epoch = epoch

    if best_epoch is None:
        print(f"[WARN] {val_loss_csv} sin filas validas; se usara el ultimo checkpoint.")
        path = find_latest_checkpoint(train_dir, prefix)
        return path, extract_epoch(path.name, prefix) or -1, None

    return find_checkpoint_for_epoch(train_dir, prefix, best_epoch), best_epoch, best_loss


def find_best_checkpoint_by_task_metric(
    fold_dir: Path, train_dir: Path, prefix: str, metric: str
) -> Tuple[Path, int, Optional[float]]:
    """Elige el checkpoint que MAXIMIZA una metrica de deteccion, no el de menor val_loss.

    `val_loss` mide perplejidad de generacion de texto, no deteccion: un modelo
    subentrenado que siempre dice "...no polyps." (la mitad del dataset) tiene
    buena perplejidad y pesimo recall. Verificado en SUN: elegir por val_loss
    hundio el recall de resnet101 (0.547 -> 0.313) y vit (0.561 -> 0.298).
    Ver §2 de `reportes/reentrenamiento.md`.

    El CSV lo genera `scripts/reentrenamiento/val_metric_per_epoch.py`.
    """
    metric_csv = fold_dir / "val_task_metric_per_epoch.csv"
    if not metric_csv.exists():
        raise FileNotFoundError(
            f"No existe {metric_csv}. Generalo primero con:\n"
            f"  python scripts/reentrenamiento/val_metric_per_epoch.py "
            f"--fold_root <raiz_de_entrenamiento>"
        )

    best_epoch: Optional[int] = None
    best_value: Optional[float] = None
    with metric_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                row_epoch = int(float(row["epoch"]))
                row_value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if best_value is None or row_value > best_value:
                best_value, best_epoch = row_value, row_epoch

    if best_epoch is None:
        raise ValueError(f"{metric_csv} no tiene filas validas para la columna '{metric}'.")

    return find_checkpoint_for_epoch(train_dir, prefix, best_epoch), best_epoch, best_value


def resolve_checkpoint(
    checkpoints_root: Path,
    subdir: str,
    fold_name: str,
    policy: str,
    epoch: Optional[int],
    f1_metric: str = "f1",
) -> Path:
    fold_dir = checkpoints_root / subdir / "folds" / fold_name
    train_dir = fold_dir / "train"
    prefix = f"positive_vs_negative_{fold_name}"

    if policy == "f1":
        path, best_epoch, best_value = find_best_checkpoint_by_task_metric(
            fold_dir, train_dir, prefix, f1_metric
        )
        value_text = f", {f1_metric}={best_value:.6f}" if best_value is not None else ""
        print(f"[CKPT] {subdir}/{fold_name}: mejor epoca por {f1_metric}={best_epoch}{value_text} -> {path.name}")
        return path

    if policy == "epoch":
        if epoch is None:
            raise ValueError("--checkpoint_policy epoch requiere --epoch N.")
        path = find_checkpoint_for_epoch(train_dir, prefix, epoch)
        print(f"[CKPT] {subdir}/{fold_name}: epoca {epoch} (fijada) -> {path.name}")
        return path

    if policy == "latest":
        path = find_latest_checkpoint(train_dir, prefix)
        print(f"[CKPT] {subdir}/{fold_name}: ultima epoca -> {path.name}")
        return path

    path, best_epoch, best_loss = find_best_checkpoint(fold_dir, train_dir, prefix)
    loss_text = f", val_loss={best_loss:.6f}" if best_loss is not None else ""
    print(f"[CKPT] {subdir}/{fold_name}: mejor epoca={best_epoch}{loss_text} -> {path.name}")
    return path


def build_test_command(
    python_exec: str,
    model: str,
    images_root: Path,
    checkpoint: Path,
    output_csv: Path,
    args: argparse.Namespace,
) -> List[str]:
    spec = MODEL_SPECS[model]
    cmd = [
        python_exec,
        str(TEST_SCRIPT),
        "--images_root",
        str(images_root),
        "--checkpoint",
        str(checkpoint),
        "--output_csv",
        str(output_csv),
        "--prefix_length",
        str(args.prefix_length),
        "--mapping_type",
        str(args.mapping_type),
        "--num_layers",
        str(args.num_layers),
        "--entry_length",
        str(args.entry_length),
        "--top_p",
        str(args.top_p),
        "--temperature",
        str(args.temperature),
        "--stop_token",
        str(args.stop_token),
        "--batch_size",
        str(args.batch_size),
        "--beam_search",
        "--encoder",
        str(spec["encoder"]),
    ]

    if model == "biomedclip":
        cmd.extend(["--biomedclip_model_id", str(spec["biomedclip_model_id"])])
    else:
        cmd.extend(["--openai_clip_name", str(spec["openai_clip_name"])])
    return cmd


def run_command(cmd: Sequence[str], env: Dict[str, str]) -> None:
    print(f"[RUN] {shlex.join(list(cmd))}")
    subprocess.run(list(cmd), cwd=str(REPO_ROOT), env=env, check=True)


def main() -> None:
    args = parse_args()

    if not TEST_SCRIPT.exists():
        raise FileNotFoundError(f"No existe test.py en: {TEST_SCRIPT}")

    images_root = resolve_repo_path(args.images_root)
    dataset_root = resolve_repo_path(args.dataset_root)
    checkpoints_root = resolve_repo_path(args.checkpoints_root)
    output_root = resolve_repo_path(args.output_root)

    if not images_root.exists():
        raise FileNotFoundError(f"No existe carpeta de imagenes: {images_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe carpeta base de dataset: {dataset_root}")
    if not checkpoints_root.exists():
        raise FileNotFoundError(f"No existe carpeta de checkpoints: {checkpoints_root}")

    bbps_text = ""
    if args.video_id:
        bbps_lookup = load_bbps_lookup(resolve_repo_path(args.bbps_csv))
        bbps_text = bbps_lookup.get(args.video_id, "")
        if bbps_text:
            print(f"[BBPS] {args.video_id}: '{bbps_text}' se agregara a cada reporte generado.")
        else:
            print(
                f"[WARN] --video_id {args.video_id} no tiene BBPS en {args.bbps_csv}; "
                "los reportes se generan sin esa frase."
            )

    models = select_models(args.model)
    folds = select_folds(args.fold)
    if args.checkpoint and (len(models) != 1 or len(folds) != 1):
        raise ValueError("--checkpoint solo se puede usar con un modelo y un fold.")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    windows = parse_frame_windows(args)
    selected_images_root = output_root / "frames_selected"
    if args.frame_window:
        selected_frames = select_frames_by_prefix_folders(
            dataset_root=dataset_root,
            selected_root=selected_images_root,
            windows=windows,
        )
    elif windows:
        selected_frames = select_frames_in_windows(
            images_root=images_root,
            selected_root=selected_images_root,
            windows=windows,
        )
    else:
        selected_frames = select_all_frames(images_root=images_root, selected_root=selected_images_root)

    env = build_env(args.gpu)
    python_exec = sys.executable
    runs: List[Dict[str, str]] = []
    consolidated_report_by_frame: Dict[str, str] = {}

    for model in models:
        spec = MODEL_SPECS[model]
        for fold_name in folds:
            if args.checkpoint:
                checkpoint_path = resolve_repo_path(args.checkpoint)
            else:
                checkpoint_path = resolve_checkpoint(
                    checkpoints_root=checkpoints_root,
                    subdir=str(spec["subdir"]),
                    fold_name=fold_name,
                    policy=args.checkpoint_policy,
                    epoch=args.epoch,
                    f1_metric=args.f1_metric,
                )

            run_dir = output_root / str(spec["subdir"]) / fold_name
            run_dir.mkdir(parents=True, exist_ok=True)
            output_csv = run_dir / "predictions.csv"

            cmd = build_test_command(
                python_exec=python_exec,
                model=model,
                images_root=selected_images_root,
                checkpoint=checkpoint_path,
                output_csv=output_csv,
                args=args,
            )
            runs.append(
                {
                    "model": model,
                    "fold": fold_name,
                    "checkpoint": to_repo_relative(checkpoint_path),
                    "images_root": to_repo_relative(selected_images_root),
                    "output_csv": to_repo_relative(output_csv),
                    "command": shlex.join(cmd),
                }
            )
            if args.dry_run:
                print(f"[DRY-RUN] {shlex.join(cmd)}")
            else:
                run_command(cmd, env)
                pred_rows = read_predictions_csv(output_csv)
                frame_report_rows = predictions_to_frame_reports(pred_rows, bbps_text=bbps_text)
                frame_report_csv = run_dir / "frame_reporte.csv"
                write_frame_report_csv(frame_report_csv, frame_report_rows)
                for item in frame_report_rows:
                    consolidated_report_by_frame.setdefault(item["frame"], item["reporte_medico"])

    config_path = output_root / "run_config.json"
    if not args.dry_run and consolidated_report_by_frame:
        consolidated_rows = [
            {"frame": str(frame), "reporte_medico": report}
            for frame, report in sorted(
                consolidated_report_by_frame.items(),
                key=lambda pair: frame_sort_key(pair[0]),
            )
        ]
        write_frame_report_csv(output_root / "frame_reporte.csv", consolidated_rows)

    config_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "inference_only",
                "images_root": to_repo_relative(images_root),
                "dataset_root": to_repo_relative(dataset_root),
                "selected_images_root": to_repo_relative(selected_images_root),
                "checkpoints_root": to_repo_relative(checkpoints_root),
                "output_root": to_repo_relative(output_root),
                "model": args.model,
                "fold": args.fold,
                "checkpoint_policy": args.checkpoint_policy,
                "checkpoint_epoch": args.epoch,
                "gpu": args.gpu,
                "video_id": args.video_id,
                "bbps_csv": args.bbps_csv if args.video_id else None,
                "bbps_text_appended": bbps_text or None,
                "dry_run": bool(args.dry_run),
                "frame_windows": windows,
                "selected_frames_count": len(selected_frames),
                "consolidated_frame_report_csv": to_repo_relative(output_root / "frame_reporte.csv"),
                "runs": runs,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Runs: {len(runs)}")
    print(f"Frames seleccionados: {len(selected_frames)}")
    print(f"Config: {to_repo_relative(config_path)}")


if __name__ == "__main__":
    main()

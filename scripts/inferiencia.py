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

DEFAULT_IMAGES_ROOT = "dataset_real_colon/001-001_frames"
DEFAULT_DATASET_ROOT = "dataset_real_colon"
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
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, por ejemplo 0 o 1.")
    parser.add_argument("--frame_start", type=int, default=23703, help="Frame inicial (incluyente).")
    parser.add_argument("--frame_end", type=int, default=23731, help="Frame final (incluyente).")
    parser.add_argument(
        "--frame_window",
        action="append",
        default=[],
        help="Ventana por prefijo: <prefijo>:<inicio>-<fin>. Repetir para multiples.",
    )

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_token", type=str, default=".")

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


def predictions_to_frame_reports(pred_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in pred_rows:
        frame_id = extract_frame_id_from_image_path(str(row.get("image_path", "")))
        if frame_id is None:
            continue
        rows.append(
            {
                "frame": frame_id,
                "reporte_medico": str(row.get("generated_caption", "")).strip(),
            }
        )
    rows.sort(key=lambda item: frame_sort_key(item["frame"]))
    return rows


def extract_epoch(filename: str, prefix: str) -> Optional[int]:
    match = re.match(rf"^{re.escape(prefix)}-(\d+)\.pt$", filename)
    if not match:
        return None
    return int(match.group(1))


def find_latest_checkpoint(train_dir: Path, prefix: str) -> Path:
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe directorio de checkpoints: {train_dir}")

    candidates: List[tuple[int, Path]] = []
    for ckpt in train_dir.glob(f"{prefix}-*.pt"):
        epoch = extract_epoch(ckpt.name, prefix)
        if epoch is not None:
            candidates.append((epoch, ckpt.resolve()))

    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints con prefijo {prefix} en {train_dir}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


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
    else:
        selected_frames = select_frames_in_windows(images_root=images_root, selected_root=selected_images_root, windows=windows)

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
                checkpoint_path = find_latest_checkpoint(
                    checkpoints_root / str(spec["subdir"]) / "folds" / fold_name / "train",
                    f"positive_vs_negative_{fold_name}",
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
                frame_report_rows = predictions_to_frame_reports(pred_rows)
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
                "gpu": args.gpu,
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

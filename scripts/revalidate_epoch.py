#!/usr/bin/env python3
"""Regenera las predicciones de validacion de SUN con otra epoca, SIN reentrenar.

Problema que resuelve
---------------------
El pipeline original (`scripts/2fold_models.py`) fija el checkpoint de la
ULTIMA epoca para la inferencia de validacion:

    checkpoint_path = find_checkpoint_for_epoch(train_dir, train_prefix, args.epochs - 1)

Pero la val_loss toca su minimo en la epoca 0-1 y luego crece ~2x hasta la
epoca 14, en los 3 modelos y en ambos folds (ver
`fold/2fold/{modelo}/folds/fold_N/val_loss_per_epoch.csv`). Es decir, las
Table I / Table II actuales se calcularon con el checkpoint MAS sobreajustado.

Como los checkpoints de las 15 epocas y las imagenes de validacion
(`inference/val_images_*`) siguen en disco, no hace falta reentrenar: basta
con volver a correr `test.py` con el checkpoint deseado y re-mergear con
`val_manifest.csv`.

Salida
------
Escribe un arbol con la MISMA estructura que `fold/2fold`, en otra carpeta
(por defecto `fold/2fold_best`), para no sobrescribir los resultados actuales:

    <output_root>/<subdir>/run_config.json
    <output_root>/<subdir>/folds/fold_N/inference/val_predictions.csv
    <output_root>/<subdir>/folds/fold_N/inference/val_predictions_raw.csv

Luego se evalua igual que siempre:

    python scripts/evaluate_fold_models.py --fold_root fold/2fold_best
"""

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

DEFAULT_SOURCE_ROOT = "fold/2fold"
DEFAULT_OUTPUT_ROOT = "fold/2fold_best"

# subdir en disco -> (nombre canonico del modelo, args de encoder para test.py)
MODEL_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "biomedclip": {
        "model_name": "biomedclip",
        "encoder": "biomedclip",
        "biomedclip_model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "openai_clip_name": None,
    },
    "resnet": {
        "model_name": "resnet101",
        "encoder": "rn101",
        "biomedclip_model_id": None,
        "openai_clip_name": "RN101",
    },
    "vit": {
        "model_name": "vit",
        "encoder": "vit",
        "biomedclip_model_id": None,
        "openai_clip_name": "ViT-B/32",
    },
}

VAL_PRED_COLUMNS = [
    "fold",
    "sample_id",
    "label",
    "case",
    "image_path",
    "caption_gt",
    "generated_caption",
    "linked_image_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenera val_predictions.csv de SUN con otra epoca, sin reentrenar."
    )
    parser.add_argument("--source_root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
        help="Subcarpetas de modelo a procesar.",
    )
    parser.add_argument("--folds", nargs="+", default=["fold_1", "fold_2"])
    parser.add_argument(
        "--checkpoint_policy",
        default="best",
        choices=["best", "epoch", "latest", "f1"],
        help=(
            "'f1' = mayor metrica de deteccion segun val_task_metric_per_epoch.csv "
            "(criterio recomendado, §2 de reportes/reentrenamiento.md); "
            "'best' = menor val_loss; 'epoch' usa --epoch; 'latest' = ultima epoca."
        ),
    )
    parser.add_argument(
        "--f1_metric",
        default="f1",
        help="Columna a maximizar con --checkpoint_policy f1 (f1, recall, lesion_detection_rate_50pct...).",
    )
    parser.add_argument("--epoch", type=int, default=None, help="Epoca fija si --checkpoint_policy epoch.")
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, por ejemplo 0.")
    # Por defecto se leen de <source_root>/<modelo>/run_config.json (training.*),
    # para que un reentrenamiento con otro num_layers no tenga que repetirlos aqui.
    parser.add_argument("--prefix_length", type=int, default=None)
    parser.add_argument("--mapping_type", default=None, choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Solo imprime los comandos.")
    return parser.parse_args()


def load_training_hparams(model_root: Path, args: argparse.Namespace) -> Dict[str, object]:
    """Hiperparametros con los que se construyo el mapper (CLI gana sobre run_config.json)."""
    training: Dict[str, object] = {}
    config_path = model_root / "run_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(payload.get("training"), dict):
                training = payload["training"]
        except Exception as exc:
            print(f"[WARN] No se pudo leer {to_repo_relative(config_path)}: {exc}")

    return {
        "prefix_length": args.prefix_length
        if args.prefix_length is not None
        else int(training.get("prefix_length", 10)),
        "mapping_type": args.mapping_type or str(training.get("mapping_type", "transformer")),
        "num_layers": args.num_layers if args.num_layers is not None else int(training.get("num_layers", 8)),
    }


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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def extract_epoch(filename: str, prefix: str) -> Optional[int]:
    match = re.match(rf"^{re.escape(prefix)}-(\d+)\.pt$", filename)
    return int(match.group(1)) if match else None


def list_checkpoints(train_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe directorio de checkpoints: {train_dir}")
    candidates = []
    for ckpt in train_dir.glob(f"{prefix}-*.pt"):
        epoch = extract_epoch(ckpt.name, prefix)
        if epoch is not None:
            candidates.append((epoch, ckpt.resolve()))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints con prefijo {prefix} en {train_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates


def pick_best_by_task_metric(
    fold_dir: Path, candidates: Sequence[Tuple[int, Path]], metric: str
) -> Tuple[Path, int, Optional[float]]:
    """Elige el checkpoint que MAXIMIZA una metrica de deteccion, no el de menor val_loss.

    `val_loss` mide perplejidad de generacion de texto, no deteccion: un modelo
    que siempre dice "...no polyps." tiene buena perplejidad y pesimo recall.
    Verificado en SUN: elegir por val_loss hundio el recall de resnet101
    (0.547 -> 0.313) y vit (0.561 -> 0.298). Ver §2 de
    `reportes/reentrenamiento.md`.

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
    for row in read_csv_rows(metric_csv):
        try:
            row_epoch = int(float(row["epoch"]))
            row_value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if best_value is None or row_value > best_value:
            best_value, best_epoch = row_value, row_epoch

    if best_epoch is None:
        raise ValueError(f"{metric_csv} no tiene filas validas para la columna '{metric}'.")

    for candidate_epoch, path in candidates:
        if candidate_epoch == best_epoch:
            return path, best_epoch, best_value
    available = ", ".join(str(e) for e, _ in candidates)
    raise FileNotFoundError(
        f"No existe el checkpoint de la mejor epoca por {metric} ({best_epoch}). Disponibles: {available}"
    )


def pick_checkpoint(
    fold_dir: Path, policy: str, epoch: Optional[int], f1_metric: str = "f1"
) -> Tuple[Path, int, Optional[float]]:
    train_dir = fold_dir / "train"
    prefix = f"positive_vs_negative_{fold_dir.name}"
    candidates = list_checkpoints(train_dir, prefix)

    if policy == "latest":
        chosen_epoch, path = candidates[-1]
        return path, chosen_epoch, None

    if policy == "epoch":
        if epoch is None:
            raise ValueError("--checkpoint_policy epoch requiere --epoch N.")
        for candidate_epoch, path in candidates:
            if candidate_epoch == epoch:
                return path, epoch, None
        available = ", ".join(str(e) for e, _ in candidates)
        raise FileNotFoundError(f"No hay checkpoint de epoca {epoch}. Disponibles: {available}")

    if policy == "f1":
        return pick_best_by_task_metric(fold_dir, candidates, f1_metric)

    val_loss_csv = fold_dir / "val_loss_per_epoch.csv"
    if not val_loss_csv.exists():
        print(f"[WARN] Falta {val_loss_csv}; se usa la ultima epoca.")
        chosen_epoch, path = candidates[-1]
        return path, chosen_epoch, None

    best_epoch: Optional[int] = None
    best_loss: Optional[float] = None
    for row in read_csv_rows(val_loss_csv):
        try:
            row_epoch = int(float(row["epoch"]))
            row_loss = float(row["val_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if best_loss is None or row_loss < best_loss:
            best_loss, best_epoch = row_loss, row_epoch

    if best_epoch is None:
        print(f"[WARN] {val_loss_csv} sin filas validas; se usa la ultima epoca.")
        chosen_epoch, path = candidates[-1]
        return path, chosen_epoch, None

    for candidate_epoch, path in candidates:
        if candidate_epoch == best_epoch:
            return path, best_epoch, best_loss
    raise FileNotFoundError(f"No existe el checkpoint de la mejor epoca ({best_epoch}) en {train_dir}")


def rebuild_val_images_dir(manifest_rows: Sequence[Dict[str, str]], target_dir: Path) -> Tuple[int, int]:
    """Reconstruye la carpeta de imagenes de validacion a partir del manifest.

    Las carpetas `inference/val_images_*` originales quedaron VACIAS (se
    limpiaron tras la corrida de 2026-03), asi que no se pueden reutilizar.
    El manifest si conserva, por cada muestra:
      - `image_path`        -> ruta relativa al repo de la imagen original
      - `linked_image_path` -> ruta absoluta (del contenedor) del symlink usado
    Se recrea un symlink por muestra conservando EXACTAMENTE el nombre de
    archivo original (`000123_caseX_img_Y.jpg`), que es unico y es la clave
    con la que despues se cruzan las predicciones.
    """
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing = 0
    for row in manifest_rows:
        rel_path = (row.get("image_path") or "").strip()
        link_name = Path((row.get("linked_image_path") or "").strip()).name
        if not rel_path or not link_name:
            missing += 1
            continue

        source = resolve_repo_path(rel_path)
        if not source.exists():
            missing += 1
            continue

        dst = target_dir / link_name
        try:
            dst.symlink_to(source)
        except OSError:
            shutil.copy2(source, dst)
        linked += 1

    return linked, missing


def merge_predictions_with_manifest(
    fold_name: str,
    pred_rows: Sequence[Dict[str, str]],
    manifest_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Cruza predicciones con el manifest usando el NOMBRE de archivo.

    `scripts/2fold_models.py` cruzaba por ruta absoluta completa
    (`linked_image_path`), pero aqui la carpeta de imagenes se reconstruye en
    otra ubicacion, asi que la ruta absoluta ya no coincide. El nombre de
    archivo si es estable y unico (lleva el indice de muestra como prefijo),
    por lo que se usa como clave del cruce.
    """
    manifest_map = {Path(row["linked_image_path"]).name: row for row in manifest_rows}
    merged: List[Dict[str, str]] = []
    for pred in pred_rows:
        pred_path = (pred.get("image_path") or "").strip()
        meta = manifest_map.get(Path(pred_path).name)
        if meta is None:
            continue
        merged.append(
            {
                "fold": fold_name,
                "sample_id": meta["sample_id"],
                "label": meta["label"],
                "case": meta["case"],
                "image_path": meta["image_path"],
                "caption_gt": meta["caption_gt"],
                "generated_caption": pred.get("generated_caption", ""),
                "linked_image_path": pred_path,
            }
        )
    return merged


def build_test_command(
    subdir: str,
    images_root: Path,
    checkpoint: Path,
    output_csv: Path,
    hparams: Dict[str, object],
) -> List[str]:
    spec = MODEL_SPECS[subdir]
    cmd = [
        sys.executable,
        str(TEST_SCRIPT),
        "--images_root",
        str(images_root),
        "--checkpoint",
        str(checkpoint),
        "--output_csv",
        str(output_csv),
        "--prefix_length",
        str(hparams["prefix_length"]),
        "--mapping_type",
        str(hparams["mapping_type"]),
        "--num_layers",
        str(hparams["num_layers"]),
        "--beam_search",
        "--encoder",
        str(spec["encoder"]),
    ]
    if subdir == "biomedclip":
        cmd.extend(["--biomedclip_model_id", str(spec["biomedclip_model_id"])])
    else:
        cmd.extend(["--openai_clip_name", str(spec["openai_clip_name"])])
    return cmd


def main() -> None:
    args = parse_args()

    if not TEST_SCRIPT.exists():
        raise FileNotFoundError(f"No existe test.py en: {TEST_SCRIPT}")

    source_root = resolve_repo_path(args.source_root)
    output_root = resolve_repo_path(args.output_root)
    if not source_root.exists():
        raise FileNotFoundError(f"No existe source_root: {source_root}")

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    report: List[Dict[str, object]] = []

    for subdir in args.models:
        spec = MODEL_SPECS[subdir]
        hparams = load_training_hparams(source_root / subdir, args)
        print(f"\n===== {subdir} ===== mapper: {hparams}")
        for fold_name in args.folds:
            src_fold_dir = source_root / subdir / "folds" / fold_name
            src_inference_dir = src_fold_dir / "inference"
            manifest_csv = src_inference_dir / "val_manifest.csv"

            if not manifest_csv.exists():
                print(f"[SKIP] {subdir}/{fold_name}: no existe {to_repo_relative(manifest_csv)}")
                continue

            checkpoint, epoch, criterion_value = pick_checkpoint(
                src_fold_dir, args.checkpoint_policy, args.epoch, args.f1_metric
            )
            manifest_rows = read_csv_rows(manifest_csv)

            out_inference_dir = output_root / subdir / "folds" / fold_name / "inference"
            out_inference_dir.mkdir(parents=True, exist_ok=True)
            val_images_dir = out_inference_dir / "val_images"
            raw_csv = out_inference_dir / "val_predictions_raw.csv"
            merged_csv = out_inference_dir / "val_predictions.csv"

            criterion_name = args.f1_metric if args.checkpoint_policy == "f1" else "val_loss"
            criterion_text = (
                f", {criterion_name}={criterion_value:.6f}" if criterion_value is not None else ""
            )
            print(f"\n[{subdir}/{fold_name}] epoca={epoch}{criterion_text} -> {checkpoint.name}")

            cmd = build_test_command(subdir, val_images_dir, checkpoint, raw_csv, hparams)

            if args.dry_run:
                print(f"  manifest={len(manifest_rows)} filas -> se reconstruiria {to_repo_relative(val_images_dir)}")
                print(f"  [DRY-RUN] {shlex.join(cmd)}")
                continue

            n_linked, n_missing = rebuild_val_images_dir(manifest_rows, val_images_dir)
            print(f"  imagenes reconstruidas: {n_linked}/{len(manifest_rows)} en {to_repo_relative(val_images_dir)}")
            if n_missing:
                print(f"  [AVISO] {n_missing} imagenes del manifest no se encontraron en disco.")
            if n_linked == 0:
                print("  [ERROR] No se pudo enlazar ninguna imagen; se omite este fold.")
                continue

            print(f"  [RUN] {shlex.join(cmd)}")
            subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)

            pred_rows = read_csv_rows(raw_csv)
            merged_rows = merge_predictions_with_manifest(fold_name, pred_rows, manifest_rows)
            write_csv(merged_csv, merged_rows, VAL_PRED_COLUMNS)

            print(
                f"  raw={len(pred_rows)} filas | manifest={len(manifest_rows)} | "
                f"merged={len(merged_rows)} -> {to_repo_relative(merged_csv)}"
            )
            if merged_rows and len(merged_rows) != len(pred_rows):
                print(
                    f"  [AVISO] {len(pred_rows) - len(merged_rows)} predicciones no cruzaron "
                    "con el manifest (revisa linked_image_path)."
                )

            report.append(
                {
                    "model": spec["model_name"],
                    "subdir": subdir,
                    "fold": fold_name,
                    "epoch": epoch,
                    "selection_criterion": criterion_name,
                    "selection_value": criterion_value,
                    "checkpoint": to_repo_relative(checkpoint),
                    "val_predictions_csv": to_repo_relative(merged_csv),
                    "merged_rows": len(merged_rows),
                }
            )

        # run_config.json por modelo: evaluate_fold_models.py lee "model" de aqui
        # para nombrar las filas de Table I / Table II (resnet -> resnet101).
        if not args.dry_run:
            model_config = output_root / subdir / "run_config.json"
            model_config.parent.mkdir(parents=True, exist_ok=True)
            model_config.write_text(
                json.dumps(
                    {
                        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                        "mode": "revalidation_only",
                        "model": spec["model_name"],
                        "requested_model": spec["model_name"],
                        "output_subdir": subdir,
                        "source_root": to_repo_relative(source_root),
                        "checkpoint_policy": args.checkpoint_policy,
                        "selection_metric": args.f1_metric if args.checkpoint_policy == "f1" else "val_loss",
                        "checkpoint_epoch": args.epoch,
                        "training": hparams,
                        "folds": [item for item in report if item["subdir"] == subdir],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    if not args.dry_run and report:
        summary_path = output_root / "revalidate_epoch_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_root": to_repo_relative(source_root),
                    "output_root": to_repo_relative(output_root),
                    "checkpoint_policy": args.checkpoint_policy,
                    "selection_metric": args.f1_metric if args.checkpoint_policy == "f1" else "val_loss",
                    "runs": report,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nResumen: {to_repo_relative(summary_path)}")
        print(f"Ahora ejecuta:\n  python scripts/evaluate_fold_models.py --fold_root {to_repo_relative(output_root)}")


if __name__ == "__main__":
    main()

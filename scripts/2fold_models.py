#!/usr/bin/env python3
"""2-fold stratified pipeline: prepare, train and validate per model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NEGATIVE_CSV = "experiments_colono/experiments_colono/clipclap_train_labeled_negative.csv"
DEFAULT_POSITIVE_CSV = "experiments_colono/experiments_colono/clipclap_train_labeled_positive.csv"
DEFAULT_OUTPUT_ROOT = "fold/2fold"

TRAIN_SCRIPT = REPO_ROOT / "train.py"
TEST_SCRIPT = REPO_ROOT / "test.py"
PARSE_BIOMED_SCRIPT = REPO_ROOT / "parse_colono_biomed.py"
PARSE_OPENAI_SCRIPT = REPO_ROOT / "parse_colono.py"
VALIDATION_SCRIPT = REPO_ROOT / "scripts" / "validation.py"
GRAFICAS_SCRIPT = REPO_ROOT / "scripts" / "graphics" / "graficas.py"

MODEL_CONFIG = {
    "biomedclip": {
        "weights_path": "clip_weights/biomedclip_weights.pt",
        "clip_model_type": None,
        "test_encoder": "biomedclip",
        "biomedclip_model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "openai_clip_name": None,
    },
    "vit": {
        "weights_path": "clip_weights/ViT-B-16.pt",
        "clip_model_type": "ViT-B/32",
        "test_encoder": "vit",
        "openai_clip_name": "ViT-B/32",
    },
    "resnet101": {
        "weights_path": "clip_weights/RN101.pt",
        "clip_model_type": "RN101",
        "test_encoder": "rn101",
        "openai_clip_name": "RN101",
    },
}

MODEL_ALIAS = {
    "biomedclip": "biomedclip",
    "vit": "vit",
    "resnet101": "resnet101",
    "resnet": "resnet101",
}

MODEL_OUTPUT_DIR = {
    "biomedclip": "biomedclip",
    "vit": "vit",
    "resnet101": "resnet",
    "resnet": "resnet",
}

MODEL_ALL_KEY = "all"
MODEL_ALL_ORDER = ["biomedclip", "vit", "resnet101"]

ROW_COLUMNS = [
    "sample_id",
    "image_path",
    "case",
    "polyp_type",
    "label",
    "label_id",
    "caption",
    "source_csv",
]

MANIFEST_COLUMNS = [
    "sample_id",
    "label",
    "case",
    "image_path",
    "resolved_image_path",
    "linked_image_path",
    "caption_gt",
]

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
        description=(
            "Prepara 2-fold estratificado y ejecuta parse/train/test por fold "
            "usando train.py, parse_colono*.py y test.py."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted([*MODEL_ALIAS.keys(), MODEL_ALL_KEY]),
        help="Modelo: biomedclip, vit, resnet101, resnet o all.",
    )
    parser.add_argument("--negative_csv", default=DEFAULT_NEGATIVE_CSV)
    parser.add_argument("--positive_csv", default=DEFAULT_POSITIVE_CSV)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--gpu",
        default=None,
        help="GPU para CUDA_VISIBLE_DEVICES (ej: 0, 1, x).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Frecuencia de checkpoints.",
    )
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Solo prepara archivos y estructura sin ejecutar parse/train/test.",
    )
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


def remove_existing_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_rows(
    positive_rows: Sequence[Dict[str, str]],
    negative_rows: Sequence[Dict[str, str]],
    positive_csv_path: Path,
    negative_csv_path: Path,
) -> List[Dict[str, str]]:
    merged: List[Dict[str, str]] = []
    sample_id = 0

    for row in positive_rows:
        merged.append(
            {
                "sample_id": sample_id,
                "image_path": (row.get("image_path") or "").strip(),
                "case": (row.get("case") or "").strip(),
                "polyp_type": (row.get("polyp_type") or "").strip(),
                "label": "positive",
                "label_id": 1,
                "caption": (row.get("caption") or "").strip(),
                "source_csv": to_repo_relative(positive_csv_path),
            }
        )
        sample_id += 1

    for row in negative_rows:
        merged.append(
            {
                "sample_id": sample_id,
                "image_path": (row.get("image_path") or "").strip(),
                "case": (row.get("case") or "").strip(),
                "polyp_type": "",
                "label": "negative",
                "label_id": 0,
                "caption": (row.get("caption") or "").strip(),
                "source_csv": to_repo_relative(negative_csv_path),
            }
        )
        sample_id += 1

    return merged


def label_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counter = Counter(str(row.get("label", "")) for row in rows)
    return {"positive": counter.get("positive", 0), "negative": counter.get("negative", 0)}


def stratified_two_fold(
    rows: Sequence[Dict[str, str]],
    seed: int,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    by_label: Dict[str, List[Dict[str, str]]] = {"positive": [], "negative": []}
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(seed)
    val_by_fold: Dict[int, List[Dict[str, str]]] = {1: [], 2: []}
    for label_rows in by_label.values():
        shuffled = list(label_rows)
        rng.shuffle(shuffled)
        split_index = len(shuffled) // 2
        val_by_fold[1].extend(shuffled[:split_index])
        val_by_fold[2].extend(shuffled[split_index:])

    all_rows = list(rows)
    output: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = []
    for fold_id in (1, 2):
        val_rows = sorted(val_by_fold[fold_id], key=lambda x: int(x["sample_id"]))
        val_ids = {int(r["sample_id"]) for r in val_rows}
        train_rows = sorted(
            [r for r in all_rows if int(r["sample_id"]) not in val_ids],
            key=lambda x: int(x["sample_id"]),
        )
        output.append((train_rows, val_rows))
    return output


def resolve_image_path(raw_path: str) -> Path:
    clean = (raw_path or "").strip().replace("\\", "/").strip('"').strip("'")
    if not clean:
        return REPO_ROOT / "__missing__"
    path = Path(clean)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def build_env(gpu: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    return env


def run_step(
    cmd: List[str],
    env: Dict[str, str],
    log_path: Path,
    cwd: Path = REPO_ROOT,
) -> None:
    pretty_cmd = shlex.join(cmd)
    print(f"\n[RUN] {pretty_cmd}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as logf:
        logf.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {pretty_cmd}\n")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def create_val_image_manifest(
    val_rows: Sequence[Dict[str, str]],
    val_images_dir: Path,
) -> Tuple[List[Dict[str, str]], int]:
    val_images_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []
    missing_count = 0

    for row in val_rows:
        src_path = resolve_image_path(str(row["image_path"]))
        if not src_path.exists():
            missing_count += 1
            continue

        dst_name = f"{int(row['sample_id']):06d}_{src_path.name}"
        dst_path = (val_images_dir / dst_name).resolve()
        if not dst_path.exists():
            try:
                dst_path.symlink_to(src_path)
            except Exception:
                # fallback sin symlink
                dst_path.write_bytes(src_path.read_bytes())

        manifest.append(
            {
                "sample_id": str(row["sample_id"]),
                "label": str(row["label"]),
                "case": str(row["case"]),
                "image_path": str(row["image_path"]),
                "resolved_image_path": src_path.as_posix(),
                "linked_image_path": dst_path.as_posix(),
                "caption_gt": str(row.get("caption", "")),
            }
        )

    return manifest, missing_count


def read_predictions_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def merge_predictions_with_manifest(
    fold_name: str,
    pred_rows: Sequence[Dict[str, str]],
    manifest_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    manifest_map = {row["linked_image_path"]: row for row in manifest_rows}
    merged: List[Dict[str, str]] = []
    for pred in pred_rows:
        linked = (pred.get("image_path") or "").strip()
        meta = manifest_map.get(linked, None)
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
                "linked_image_path": linked,
            }
        )
    return merged


def ensure_clip_embeddings_snapshot(parsed_pkl_path: Path, out_path: Path) -> Dict[str, object]:
    import torch

    if not parsed_pkl_path.exists():
        raise FileNotFoundError(f"No existe PKL de embeddings: {parsed_pkl_path}")

    with parsed_pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "clip_embedding" not in data:
        raise ValueError(f"Estructura invalida en {parsed_pkl_path}: falta 'clip_embedding'.")

    clip_embedding = data["clip_embedding"]
    if not torch.is_tensor(clip_embedding):
        clip_embedding = torch.as_tensor(clip_embedding)
    clip_embedding = clip_embedding.detach().cpu()

    payload = {
        "clip_embedding": clip_embedding,
        "captions": data.get("captions", []),
        "source_pkl": to_repo_relative(parsed_pkl_path),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    return {
        "path": to_repo_relative(out_path),
        "num_embeddings": int(clip_embedding.shape[0]) if clip_embedding.ndim >= 1 else 0,
        "embedding_dim": int(clip_embedding.shape[-1]) if clip_embedding.ndim >= 2 else 0,
    }

def build_run_commands(
    model: str,
    model_root: Path,
    weight_path: Path,
    gpu: Optional[str],
    args: argparse.Namespace,
) -> str:
    env_prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu else ""
    py = shlex.quote(sys.executable)
    parse_biomed = shlex.quote(str(PARSE_BIOMED_SCRIPT))
    parse_openai = shlex.quote(str(PARSE_OPENAI_SCRIPT))
    train_script = shlex.quote(str(TRAIN_SCRIPT))
    test_script = shlex.quote(str(TEST_SCRIPT))
    validation_script = shlex.quote(str(VALIDATION_SCRIPT))
    graficas_script = shlex.quote(str(GRAFICAS_SCRIPT))

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f"# Modelo: {model}",
        f"# GPU: {gpu if gpu else 'no_forzada'}",
        "# Comandos de referencia equivalentes al flujo automático.",
        "# Nota: la validacion/val_loss se calcula via scripts/validation.py.",
        "",
    ]

    model_cfg = MODEL_CONFIG[model]
    for fold_id in (1, 2):
        fold_name = f"fold_{fold_id}"
        fold_dir = model_root / "folds" / fold_name
        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        train_pkl = fold_dir / "data" / "train.pkl"
        val_pkl = fold_dir / "data" / "val.pkl"
        train_dir = fold_dir / "train"
        pred_csv = fold_dir / "inference" / "val_predictions_raw.csv"
        val_images = fold_dir / "inference" / "val_images_run_manual"
        train_prefix = f"positive_vs_negative_{fold_name}"

        lines.append(f"# ===== {fold_name} =====")
        if model == "biomedclip":
            lines.append(
                f"{env_prefix}{py} {parse_biomed} --csv_path {shlex.quote(str(train_csv))} "
                f"--images_root . --out_path {shlex.quote(str(train_pkl))} "
                f"--weights_path {shlex.quote(str(weight_path))}"
            )
            lines.append(
                f"{env_prefix}{py} {parse_biomed} --csv_path {shlex.quote(str(val_csv))} "
                f"--images_root . --out_path {shlex.quote(str(val_pkl))} "
                f"--weights_path {shlex.quote(str(weight_path))}"
            )
        else:
            lines.append(
                f"{env_prefix}{py} {parse_openai} --csv_path {shlex.quote(str(train_csv))} "
                f"--images_root . --out_path {shlex.quote(str(train_pkl))} "
                f"--clip_model_type {shlex.quote(model_cfg['clip_model_type'])}"
            )
            lines.append(
                f"{env_prefix}{py} {parse_openai} --csv_path {shlex.quote(str(val_csv))} "
                f"--images_root . --out_path {shlex.quote(str(val_pkl))} "
                f"--clip_model_type {shlex.quote(model_cfg['clip_model_type'])}"
            )

        lines.append(
            f"{env_prefix}{py} {train_script} "
            f"--data {shlex.quote(str(train_pkl))} "
            f"--out_dir {shlex.quote(str(train_dir))} "
            f"--prefix {shlex.quote(train_prefix)} "
            f"--epochs {args.epochs} --save_every {args.save_every} --bs {args.bs} "
            f"--prefix_length {args.prefix_length} --prefix_length_clip {args.prefix_length_clip} "
            f"--mapping_type {shlex.quote(args.mapping_type)} --num_layers {args.num_layers} "
            "--only_prefix --normalize_prefix"
        )

        if model == "biomedclip":
            lines.append(
                f"{env_prefix}{py} {test_script} "
                f"--images_root {shlex.quote(str(val_images))} "
                f"--checkpoint <CHECKPOINT_{fold_name.upper()}> "
                f"--output_csv {shlex.quote(str(pred_csv))} "
                f"--prefix_length {args.prefix_length} "
                f"--mapping_type {shlex.quote(args.mapping_type)} "
                f"--num_layers {args.num_layers} --beam_search "
                "--encoder biomedclip "
                f"--biomedclip_model_id {shlex.quote(model_cfg['biomedclip_model_id'])}"
            )
        else:
            lines.append(
                f"{env_prefix}{py} {test_script} "
                f"--images_root {shlex.quote(str(val_images))} "
                f"--checkpoint <CHECKPOINT_{fold_name.upper()}> "
                f"--output_csv {shlex.quote(str(pred_csv))} "
                f"--prefix_length {args.prefix_length} "
                f"--mapping_type {shlex.quote(args.mapping_type)} "
                f"--num_layers {args.num_layers} --beam_search "
                f"--encoder {shlex.quote(model_cfg['test_encoder'])} "
                f"--openai_clip_name {shlex.quote(model_cfg['openai_clip_name'])}"
            )
        lines.append("")

    lines.append("# ===== Post-proceso =====")
    lines.append(
        f"{env_prefix}{py} {validation_script} "
        f"--model {shlex.quote(model)} "
        f"--output_root {shlex.quote(str(model_root.parent))} "
        f"--start_epoch 0 --end_epoch {max(0, int(args.epochs) - 1)}"
    )
    lines.append(
        f"{env_prefix}{py} {graficas_script} "
        f"--input_root {shlex.quote(str(model_root.parent))} "
        f"--output_root {shlex.quote(str(model_root.parent / 'plots'))} "
        f"--models {shlex.quote(model_root.name)}"
    )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()

    if args.model == MODEL_ALL_KEY:
        models_to_run = list(MODEL_ALL_ORDER)
        output_root = resolve_repo_path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        log_path = output_root / "pipeline_all.log"
        remove_existing_path(log_path)

        env = build_env(args.gpu)
        python_exec = sys.executable
        script_path = Path(__file__).resolve()
        print(f"Modo --model all activado. Se ejecutaran: {', '.join(models_to_run)}")

        for requested_model in models_to_run:
            print(f"\n========== Ejecutando modelo {requested_model} ==========")
            cmd = [
                python_exec,
                str(script_path),
                "--model",
                requested_model,
                "--negative_csv",
                str(args.negative_csv),
                "--positive_csv",
                str(args.positive_csv),
                "--output_root",
                str(args.output_root),
                "--seed",
                str(args.seed),
                "--epochs",
                str(args.epochs),
                "--save_every",
                str(args.save_every),
                "--bs",
                str(args.bs),
                "--prefix_length",
                str(args.prefix_length),
                "--prefix_length_clip",
                str(args.prefix_length_clip),
                "--mapping_type",
                str(args.mapping_type),
                "--num_layers",
                str(args.num_layers),
            ]

            if args.gpu is not None:
                cmd.extend(["--gpu", str(args.gpu)])
            if args.prepare_only:
                cmd.append("--prepare_only")

            run_step(cmd, env, log_path)

        if not args.prepare_only:
            plot_cmd = [
                python_exec,
                str(GRAFICAS_SCRIPT),
                "--input_root",
                str(output_root),
                "--output_root",
                str(output_root / "plots"),
                "--models",
                "biomedclip",
                "vit",
                "resnet",
            ]
            run_step(plot_cmd, env, log_path)

        print(
            f"\nEjecucion completa para --model all. Log global: {to_repo_relative(log_path)}"
        )
        return

    canonical_model = MODEL_ALIAS[args.model]
    output_subdir = MODEL_OUTPUT_DIR[args.model]
    model_cfg = MODEL_CONFIG[canonical_model]

    positive_csv_path = resolve_repo_path(args.positive_csv)
    negative_csv_path = resolve_repo_path(args.negative_csv)
    output_root = resolve_repo_path(args.output_root)
    model_root = output_root / output_subdir
    weight_path = resolve_repo_path(model_cfg["weights_path"])

    for required in [positive_csv_path, negative_csv_path, TRAIN_SCRIPT, TEST_SCRIPT, VALIDATION_SCRIPT, GRAFICAS_SCRIPT]:
        if not required.exists():
            raise FileNotFoundError(f"No existe archivo requerido: {required}")
    if canonical_model == "biomedclip" and not PARSE_BIOMED_SCRIPT.exists():
        raise FileNotFoundError(f"No existe: {PARSE_BIOMED_SCRIPT}")
    if canonical_model in {"vit", "resnet101"} and not PARSE_OPENAI_SCRIPT.exists():
        raise FileNotFoundError(f"No existe: {PARSE_OPENAI_SCRIPT}")
    if not weight_path.exists():
        raise FileNotFoundError(f"No existe archivo de pesos: {weight_path}")

    if model_root.exists():
        print(f"Reemplazando salida previa en: {to_repo_relative(model_root)}")
        remove_existing_path(model_root)

    positive_rows = read_csv_rows(positive_csv_path)
    negative_rows = read_csv_rows(negative_csv_path)
    merged_rows = build_rows(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        positive_csv_path=positive_csv_path,
        negative_csv_path=negative_csv_path,
    )
    fold_splits = stratified_two_fold(merged_rows, args.seed)

    data_dir = model_root / "data"
    folds_dir = model_root / "folds"
    data_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)

    merged_csv_path = data_dir / "merged_labeled.csv"
    captions_reference_path = data_dir / "captions_reference.csv"
    captions_generated_path = data_dir / "captions_generated.csv"
    write_csv(merged_csv_path, merged_rows, ROW_COLUMNS)
    write_csv(
        captions_reference_path,
        merged_rows,
        ["sample_id", "image_path", "case", "label", "caption"],
    )

    fold_artifacts: Dict[str, Dict[str, str]] = {}
    summary_folds: List[Dict[str, object]] = []
    fold_rows_map: Dict[str, Tuple[List[Dict[str, str]], List[Dict[str, str]]]] = {}

    for idx, (train_rows, val_rows) in enumerate(fold_splits, start=1):
        fold_name = f"fold_{idx}"
        fold_dir = folds_dir / fold_name
        (fold_dir / "data").mkdir(parents=True, exist_ok=True)
        (fold_dir / "train").mkdir(parents=True, exist_ok=True)
        (fold_dir / "inference").mkdir(parents=True, exist_ok=True)
        (fold_dir / "embeddings").mkdir(parents=True, exist_ok=True)

        train_csv_path = fold_dir / "train.csv"
        val_csv_path = fold_dir / "val.csv"
        write_csv(train_csv_path, train_rows, ROW_COLUMNS)
        write_csv(val_csv_path, val_rows, ROW_COLUMNS)

        fold_artifacts[fold_name] = {
            "fold_dir": to_repo_relative(fold_dir),
            "train_csv": to_repo_relative(train_csv_path),
            "val_csv": to_repo_relative(val_csv_path),
            "train_pkl": to_repo_relative(fold_dir / "data" / "train.pkl"),
            "val_pkl": to_repo_relative(fold_dir / "data" / "val.pkl"),
            "train_dir": to_repo_relative(fold_dir / "train"),
            "inference_dir": to_repo_relative(fold_dir / "inference"),
            "val_predictions_raw_csv": to_repo_relative(fold_dir / "inference" / "val_predictions_raw.csv"),
            "val_predictions_csv": to_repo_relative(fold_dir / "inference" / "val_predictions.csv"),
            "val_manifest_csv": to_repo_relative(fold_dir / "inference" / "val_manifest.csv"),
            "train_clip_embeddings_pt": to_repo_relative(fold_dir / "embeddings" / "train_clip_embeddings.pt"),
            "val_clip_embeddings_pt": to_repo_relative(fold_dir / "embeddings" / "val_clip_embeddings.pt"),
            "val_pre_gpt_embeddings_pt": to_repo_relative(fold_dir / "embeddings" / "val_pre_gpt_embeddings.pt"),
            "val_post_gpt_embeddings_pt": to_repo_relative(fold_dir / "embeddings" / "val_post_gpt_embeddings.pt"),
            "val_loss_csv": to_repo_relative(fold_dir / "val_loss_per_epoch.csv"),
            "val_loss_summary_json": to_repo_relative(fold_dir / "inference" / "val_loss_summary.json"),
            "val_loss_per_epoch_csv": to_repo_relative(fold_dir / "val_loss_per_epoch.csv"),
        }
        fold_rows_map[fold_name] = (train_rows, val_rows)

        summary_folds.append(
            {
                "fold": fold_name,
                "train_samples": len(train_rows),
                "val_samples": len(val_rows),
                "train_label_distribution": label_counts(train_rows),
                "val_label_distribution": label_counts(val_rows),
                "artifacts": fold_artifacts[fold_name],
            }
        )

    run_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": canonical_model,
        "requested_model": args.model,
        "output_subdir": output_subdir,
        "gpu": args.gpu,
        "weights_path": to_repo_relative(weight_path),
        "seed": args.seed,
        "num_folds": 2,
        "inputs": {
            "negative_csv": to_repo_relative(negative_csv_path),
            "positive_csv": to_repo_relative(positive_csv_path),
        },
        "scripts": {
            "parse_colono_biomed": to_repo_relative(PARSE_BIOMED_SCRIPT),
            "parse_colono": to_repo_relative(PARSE_OPENAI_SCRIPT),
            "train": to_repo_relative(TRAIN_SCRIPT),
            "test": to_repo_relative(TEST_SCRIPT),
            "validation": to_repo_relative(VALIDATION_SCRIPT),
            "graficas": to_repo_relative(GRAFICAS_SCRIPT),
        },
        "training": {
            "epochs": args.epochs,
            "save_every": args.save_every,
            "bs": args.bs,
            "prefix_length": args.prefix_length,
            "prefix_length_clip": args.prefix_length_clip,
            "mapping_type": args.mapping_type,
            "num_layers": args.num_layers,
            "only_prefix": True,
            "normalize_prefix": True,
        },
        "outputs": {
            "output_root": to_repo_relative(output_root),
            "model_root": to_repo_relative(model_root),
            "merged_labeled_csv": to_repo_relative(merged_csv_path),
            "captions_reference_csv": to_repo_relative(captions_reference_path),
            "captions_generated_csv": to_repo_relative(captions_generated_path),
            "folds": fold_artifacts,
        },
        "dataset": {
            "total_samples": len(merged_rows),
            "label_distribution": label_counts(merged_rows),
        },
        "prepare_only": bool(args.prepare_only),
    }

    summary = {
        "model": canonical_model,
        "requested_model": args.model,
        "output_subdir": output_subdir,
        "gpu": args.gpu,
        "total_samples": len(merged_rows),
        "label_distribution": label_counts(merged_rows),
        "folds": summary_folds,
        "executed": not args.prepare_only,
    }

    run_config_path = model_root / "run_config.json"
    summary_path = model_root / "summary.json"
    commands_path = model_root / "run_commands.sh"
    log_path = model_root / "pipeline.log"
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    commands_path.write_text(
        build_run_commands(canonical_model, model_root, weight_path, args.gpu, args),
        encoding="utf-8",
    )

    print(f"Modelo seleccionado: {args.model} (interno: {canonical_model})")
    print(f"GPU: {args.gpu if args.gpu else 'no_forzada'}")
    print(f"Total muestras: {len(merged_rows)} | Distribucion: {label_counts(merged_rows)}")
    print(f"Estructura base guardada en: {to_repo_relative(model_root)}")

    if args.prepare_only:
        print("Modo prepare_only: no se ejecutaron parse/train/test.")
        return

    from validation import (  # noqa: WPS433
        compute_and_save_val_loss_per_epoch,
        compute_and_save_validation_artifacts,
        find_checkpoint_for_epoch,
    )

    env = build_env(args.gpu)
    python_exec = sys.executable
    all_generated_rows: List[Dict[str, str]] = []
    execution_report: List[Dict[str, object]] = []
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for fold_name in ("fold_1", "fold_2"):
        print(f"\n========== Ejecutando {fold_name} ==========")
        fold_dir = resolve_repo_path(fold_artifacts[fold_name]["fold_dir"])
        train_csv = resolve_repo_path(fold_artifacts[fold_name]["train_csv"])
        val_csv = resolve_repo_path(fold_artifacts[fold_name]["val_csv"])
        train_pkl = resolve_repo_path(fold_artifacts[fold_name]["train_pkl"])
        val_pkl = resolve_repo_path(fold_artifacts[fold_name]["val_pkl"])
        train_dir = resolve_repo_path(fold_artifacts[fold_name]["train_dir"])
        inference_dir = resolve_repo_path(fold_artifacts[fold_name]["inference_dir"])
        raw_pred_csv = resolve_repo_path(fold_artifacts[fold_name]["val_predictions_raw_csv"])
        merged_pred_csv = resolve_repo_path(fold_artifacts[fold_name]["val_predictions_csv"])
        manifest_csv = resolve_repo_path(fold_artifacts[fold_name]["val_manifest_csv"])
        train_clip_embeddings_pt = resolve_repo_path(fold_artifacts[fold_name]["train_clip_embeddings_pt"])
        val_clip_embeddings_pt = resolve_repo_path(fold_artifacts[fold_name]["val_clip_embeddings_pt"])
        val_pre_gpt_embeddings_pt = resolve_repo_path(fold_artifacts[fold_name]["val_pre_gpt_embeddings_pt"])
        val_post_gpt_embeddings_pt = resolve_repo_path(fold_artifacts[fold_name]["val_post_gpt_embeddings_pt"])
        val_loss_summary_json = resolve_repo_path(fold_artifacts[fold_name]["val_loss_summary_json"])
        val_loss_per_epoch_csv = resolve_repo_path(fold_artifacts[fold_name]["val_loss_per_epoch_csv"])

        parse_script = PARSE_BIOMED_SCRIPT if canonical_model == "biomedclip" else PARSE_OPENAI_SCRIPT
        if canonical_model == "biomedclip":
            parse_train_cmd = [
                python_exec,
                str(parse_script),
                "--csv_path",
                str(train_csv),
                "--images_root",
                ".",
                "--out_path",
                str(train_pkl),
                "--weights_path",
                str(weight_path),
            ]
            parse_val_cmd = [
                python_exec,
                str(parse_script),
                "--csv_path",
                str(val_csv),
                "--images_root",
                ".",
                "--out_path",
                str(val_pkl),
                "--weights_path",
                str(weight_path),
            ]
        else:
            clip_model_type = str(model_cfg["clip_model_type"])
            parse_train_cmd = [
                python_exec,
                str(parse_script),
                "--csv_path",
                str(train_csv),
                "--images_root",
                ".",
                "--out_path",
                str(train_pkl),
                "--clip_model_type",
                clip_model_type,
            ]
            parse_val_cmd = [
                python_exec,
                str(parse_script),
                "--csv_path",
                str(val_csv),
                "--images_root",
                ".",
                "--out_path",
                str(val_pkl),
                "--clip_model_type",
                clip_model_type,
            ]

        train_prefix = f"positive_vs_negative_{fold_name}"
        train_cmd = [
            python_exec,
            str(TRAIN_SCRIPT),
            "--data",
            str(train_pkl),
            "--out_dir",
            str(train_dir),
            "--prefix",
            train_prefix,
            "--epochs",
            str(args.epochs),
            "--save_every",
            str(args.save_every),
            "--prefix_length",
            str(args.prefix_length),
            "--prefix_length_clip",
            str(args.prefix_length_clip),
            "--bs",
            str(args.bs),
            "--mapping_type",
            str(args.mapping_type),
            "--num_layers",
            str(args.num_layers),
            "--only_prefix",
            "--normalize_prefix",
        ]

        run_step(parse_train_cmd, env, log_path)
        run_step(parse_val_cmd, env, log_path)
        train_clip_info = ensure_clip_embeddings_snapshot(train_pkl, train_clip_embeddings_pt)
        val_clip_info = ensure_clip_embeddings_snapshot(val_pkl, val_clip_embeddings_pt)
        run_step(train_cmd, env, log_path)

        val_loss_per_epoch_rows = compute_and_save_val_loss_per_epoch(
            val_pkl=val_pkl,
            train_dir=train_dir,
            checkpoint_prefix=train_prefix,
            start_epoch=0,
            end_epoch=max(0, int(args.epochs) - 1),
            out_csv_path=val_loss_per_epoch_csv,
            prefix_length=args.prefix_length,
            mapping_type=str(args.mapping_type),
            num_layers=args.num_layers,
            batch_size=args.bs,
            allow_missing_checkpoints=False,
        )
        if not val_loss_per_epoch_rows:
            raise RuntimeError(
                f"No se evaluo ninguna epoca para {fold_name}; revisa checkpoints en {train_dir}."
            )

        checkpoint_path = find_checkpoint_for_epoch(train_dir, train_prefix, args.epochs - 1)
        val_eval_summary = compute_and_save_validation_artifacts(
            val_pkl=val_pkl,
            checkpoint_path=checkpoint_path,
            val_loss_per_epoch_csv_path=val_loss_per_epoch_csv,
            val_loss_summary_path=val_loss_summary_json,
            pre_gpt_embeddings_path=val_pre_gpt_embeddings_pt,
            post_gpt_embeddings_path=val_post_gpt_embeddings_pt,
            prefix_length=args.prefix_length,
            mapping_type=str(args.mapping_type),
            num_layers=args.num_layers,
            batch_size=args.bs,
        )

        _, val_rows = fold_rows_map[fold_name]
        val_images_dir = inference_dir / f"val_images_{run_tag}"
        manifest_rows, missing_count = create_val_image_manifest(val_rows, val_images_dir)
        write_csv(manifest_csv, manifest_rows, MANIFEST_COLUMNS)

        test_cmd = [
            python_exec,
            str(TEST_SCRIPT),
            "--images_root",
            str(val_images_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--output_csv",
            str(raw_pred_csv),
            "--prefix_length",
            str(args.prefix_length),
            "--mapping_type",
            str(args.mapping_type),
            "--num_layers",
            str(args.num_layers),
            "--beam_search",
        ]

        if canonical_model == "biomedclip":
            test_cmd.extend(
                [
                    "--encoder",
                    "biomedclip",
                    "--biomedclip_model_id",
                    str(model_cfg["biomedclip_model_id"]),
                ]
            )
        else:
            test_cmd.extend(
                [
                    "--encoder",
                    str(model_cfg["test_encoder"]),
                    "--openai_clip_name",
                    str(model_cfg["openai_clip_name"]),
                ]
            )

        run_step(test_cmd, env, log_path)

        pred_rows = read_predictions_csv(raw_pred_csv)
        merged_pred_rows = merge_predictions_with_manifest(fold_name, pred_rows, manifest_rows)
        write_csv(merged_pred_csv, merged_pred_rows, VAL_PRED_COLUMNS)
        all_generated_rows.extend(merged_pred_rows)

        execution_report.append(
            {
                "fold": fold_name,
                "checkpoint": to_repo_relative(checkpoint_path),
                "val_manifest_rows": len(manifest_rows),
                "val_missing_images": missing_count,
                "raw_predictions_rows": len(pred_rows),
                "matched_predictions_rows": len(merged_pred_rows),
                "val_images_dir": to_repo_relative(val_images_dir),
                "mean_val_loss": val_eval_summary["mean_val_loss"],
                "val_loss_batches": val_eval_summary["val_batches"],
                "val_loss_per_epoch_csv": to_repo_relative(val_loss_per_epoch_csv),
                "val_loss_per_epoch_points": len(val_loss_per_epoch_rows),
                "train_clip_embeddings": train_clip_info,
                "val_clip_embeddings": val_clip_info,
                "pre_gpt_embeddings_shape": val_eval_summary["pre_gpt_embeddings_shape"],
                "post_gpt_embeddings_shape": val_eval_summary["post_gpt_embeddings_shape"],
            }
        )

        print(
            f"{fold_name}: ckpt={to_repo_relative(checkpoint_path)} | "
            f"manifest={len(manifest_rows)} | missing={missing_count} | "
            f"pred_raw={len(pred_rows)} | pred_matched={len(merged_pred_rows)} | "
            f"val_loss={val_eval_summary['mean_val_loss']:.6f} | "
            f"val_epochs={len(val_loss_per_epoch_rows)}"
        )

    write_csv(captions_generated_path, all_generated_rows, VAL_PRED_COLUMNS)
    run_config["execution_report"] = execution_report
    summary["execution_report"] = execution_report
    summary["generated_captions_rows"] = len(all_generated_rows)
    run_config_path.write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plot_cmd = [
        python_exec,
        str(GRAFICAS_SCRIPT),
        "--input_root",
        str(output_root),
        "--output_root",
        str(output_root / "plots"),
        "--models",
        output_subdir,
    ]
    run_step(plot_cmd, env, log_path)

    print(f"\nPipeline completo. Captions generados guardados en: {to_repo_relative(captions_generated_path)}")


if __name__ == "__main__":
    main()

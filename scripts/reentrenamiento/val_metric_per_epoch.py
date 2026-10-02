#!/usr/bin/env python3
"""F1 de deteccion (y deteccion por lesion) por EPOCA sobre el fold de validacion.

Que resuelve
------------
`reportes/reentrenamiento.md` §2: hoy el checkpoint se elige por `val_loss`
(`scripts/inferiencia.py`, `scripts/revalidate_epoch.py`), y ese criterio esta
mal para esta tarea. `val_loss` mide perplejidad de generacion de texto, no si
el modelo detecta el polipo: un modelo subentrenado que siempre dice
"...no polyps." (la mitad del dataset) tiene buena perplejidad y pesima
deteccion. Verificado en SUN: elegir por `val_loss` hundio el recall de
resnet101 (0.547 -> 0.313) y vit (0.561 -> 0.298).

Este script produce la pieza que faltaba: `val_task_metric_per_epoch.csv`, con
F1 frame-level y `lesion_detection_rate_50pct` para CADA checkpoint guardado.
Con eso, `scripts/revalidate_epoch.py --checkpoint_policy f1` puede elegir el
checkpoint que maximiza deteccion en vez del que minimiza perplejidad.

NO re-ejecuta `test.py`. Los embeddings CLIP del fold de validacion ya estan
cacheados en `weights/sun_<experimento>/<modelo>/fold_N/val.pkl` desde el entrenamiento, asi que no hay
que volver a pasar 5738 imagenes por el encoder por cada epoca. Solo se corre
el mapper + GPT-2, en batch y con generacion greedy.

  ADVERTENCIA METODOLOGICA 1: la generacion aqui es GREEDY y en batch; las
  tablas finales (`scripts/evaluate_fold_models.py`) se calculan con BEAM
  SEARCH via `test.py`. Los numeros de este CSV sirven para ORDENAR epocas
  entre si, no para reportarse como resultado. Si se quiere fidelidad exacta
  al pipeline de reporte, usar `--beam_size 5` (mucho mas lento).

  ADVERTENCIA METODOLOGICA 2 (§2 del reporte): elegir el checkpoint mirando el
  fold de validacion es seleccionar sobre el propio conjunto de evaluacion.
  Con 2 folds y sin split de desarrollo aparte, este sesgo no se elimina, solo
  se declara. Debe quedar escrito en el reporte final.

Metricas reutilizadas tal cual de `scripts/evaluate_fold_models.py`
(`binary_metrics`, `lesion_level_metrics`), para que el criterio de seleccion y
las tablas finales midan exactamente lo mismo.

Uso
---
    python scripts/reentrenamiento/val_metric_per_epoch.py \
        --fold_root fold/2fold_hp --gpu 0

Salidas (por modelo/fold):
    <fold_root>/<modelo>/folds/fold_N/val_task_metric_per_epoch.csv
    <fold_root>/val_task_metric_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS_ROOT = REPO_ROOT / "weights"


def fold_weights_dir(model_root: Path, fold_name: str) -> Path:
    """Los .pkl de embeddings viven en weights/sun_<experimento>/<modelo>/<fold>/."""
    return WEIGHTS_ROOT / f"sun_{model_root.parent.name}" / model_root.name / fold_name


SCRIPTS_DIR = REPO_ROOT / "scripts"
for _extra_path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import GPT2Tokenizer  # noqa: E402

from evaluate_fold_models import (  # noqa: E402
    binary_metrics,
    lesion_level_metrics,
    normalize_text,
)

DEFAULT_FOLD_ROOT = "fold/2fold"
DEFAULT_MODELS = ["biomedclip", "resnet", "vit"]
DEFAULT_FOLDS = ["fold_1", "fold_2"]

METRIC_CHOICES = ["f1", "recall", "accuracy", "lesion_detection_rate_50pct", "lesion_detection_rate_any"]

OUTPUT_CSV_NAME = "val_task_metric_per_epoch.csv"

CSV_COLUMNS = [
    "epoch",
    "checkpoint",
    "n_samples",
    "tp",
    "tn",
    "fp",
    "fn",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "n_lesions",
    "n_detected_50pct",
    "lesion_detection_rate_any",
    "lesion_detection_rate_50pct",
    "decoding",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula F1 de deteccion por epoca sobre el fold de validacion (SUN)."
    )
    parser.add_argument(
        "--fold_root",
        default=DEFAULT_FOLD_ROOT,
        help="Raiz del entrenamiento (la misma que --output_root de scripts/2fold_models.py).",
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Subcarpetas de modelo.")
    parser.add_argument("--folds", nargs="+", default=DEFAULT_FOLDS)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES, por ejemplo 0.")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--end_epoch", type=int, default=None, help="Por defecto, la ultima epoca disponible.")
    parser.add_argument(
        "--metric",
        default="f1",
        choices=METRIC_CHOICES,
        help="Metrica que se reporta como 'mejor epoca' en el resumen.",
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraccion de frames a evaluar por caso, equiespaciados (1.0 = todos). "
            "Se toma la MISMA fraccion en cada caso para no alterar el balance "
            "positivo/negativo (los casos negativos tienen ~570 frames y los "
            "positivos 76). Solo afecta la SELECCION, no las tablas finales."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=48, help="Tokens maximos a generar por caption.")
    parser.add_argument(
        "--beam_size",
        type=int,
        default=0,
        help="0 = greedy en batch (rapido, por defecto). >1 = beam search por muestra (lento).",
    )
    parser.add_argument("--stop_token", default=".")
    # Se leen de run_config.json si existe; estos flags solo sirven para forzarlos.
    parser.add_argument("--prefix_length", type=int, default=None)
    parser.add_argument("--prefix_length_clip", type=int, default=None)
    parser.add_argument("--mapping_type", default=None, choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Recalcula aunque el CSV ya exista.")
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


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def list_checkpoints(train_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe directorio de checkpoints: {train_dir}")
    found: List[Tuple[int, Path]] = []
    for ckpt in train_dir.glob(f"{prefix}-*.pt"):
        match = re.match(rf"^{re.escape(prefix)}-(\d+)\.pt$", ckpt.name)
        if match:
            found.append((int(match.group(1)), ckpt.resolve()))
    if not found:
        raise FileNotFoundError(f"No hay checkpoints con prefijo {prefix} en {train_dir}")
    return sorted(found, key=lambda item: item[0])


def load_training_config(model_root: Path) -> Dict[str, Any]:
    """Lee los hiperparametros con los que se entreno, para construir el mapper igual."""
    config_path = model_root / "run_config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - archivo corrupto
        print(f"[WARN] No se pudo leer {to_repo_relative(config_path)}: {exc}")
        return {}
    training = payload.get("training")
    return training if isinstance(training, dict) else {}


def resolve_model_hparams(args: argparse.Namespace, training: Dict[str, Any]) -> Dict[str, Any]:
    prefix_length = args.prefix_length if args.prefix_length is not None else int(training.get("prefix_length", 10))
    prefix_length_clip = (
        args.prefix_length_clip
        if args.prefix_length_clip is not None
        else int(training.get("prefix_length_clip", prefix_length))
    )
    mapping_type = args.mapping_type or str(training.get("mapping_type", "transformer"))
    num_layers = args.num_layers if args.num_layers is not None else int(training.get("num_layers", 8))
    return {
        "prefix_length": int(prefix_length),
        "prefix_length_clip": int(prefix_length_clip),
        "mapping_type": mapping_type,
        "num_layers": int(num_layers),
    }


def load_val_prefixes(val_pkl: Path) -> Tuple[torch.Tensor, List[str]]:
    """Devuelve los prefijos CLIP del fold de validacion y su caption de referencia.

    Los embeddings ya estan cacheados por parse_colono*.py: reusarlos evita
    volver a pasar todas las imagenes por el encoder en cada epoca.
    """
    with val_pkl.open("rb") as handle:
        payload = pickle.load(handle)

    embeddings = payload["clip_embedding"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    embeddings = embeddings.float()

    captions_raw = payload["captions"]
    order = [int(item["clip_embedding"]) for item in captions_raw]
    prefixes = embeddings[order]
    # Mismo normalize_prefix=True que usa el entrenamiento (train.ClipCocoDataset).
    prefixes = prefixes / prefixes.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    captions = [str(item["caption"]) for item in captions_raw]
    return prefixes, captions


def align_with_val_csv(val_csv: Path, captions: Sequence[str]) -> Optional[List[Dict[str, str]]]:
    """Empareja las filas de val.csv con las muestras del .pkl (para metricas por lesion).

    parse_colono*.py recorre el CSV en orden y omite las imagenes que no pudo
    abrir, asi que la correspondencia posicional solo es valida si no se perdio
    ninguna. Se verifica comparando las captions una a una: si algo no cuadra,
    se devuelve None y se reportan solo las metricas frame-level.
    """
    rows = read_csv_rows(val_csv)
    if not rows:
        print(f"[WARN] No existe o esta vacio {to_repo_relative(val_csv)}: sin metricas por lesion.")
        return None
    if len(rows) != len(captions):
        print(
            f"[WARN] {to_repo_relative(val_csv)} tiene {len(rows)} filas y el .pkl {len(captions)} "
            "muestras (alguna imagen fallo al codificarse): sin metricas por lesion."
        )
        return None
    for idx, (row, caption) in enumerate(zip(rows, captions)):
        if normalize_text(row.get("caption")) != normalize_text(caption):
            print(
                f"[WARN] Desalineacion entre val.csv y val.pkl en la fila {idx}: "
                "sin metricas por lesion."
            )
            return None
    return rows


def subsample_by_case(meta_rows: Sequence[Dict[str, str]], fraction: float) -> List[int]:
    """Indices de una fraccion de frames equiespaciados DENTRO de cada caso.

    Se aplica la misma fraccion a todos los casos (no un numero fijo de frames)
    porque los casos negativos tienen ~570 frames y los positivos 76: un tope
    fijo por caso convertiria un fold 50/50 en uno ~88% positivo y distorsionaria
    precision, especificidad y por tanto el F1 que se usa para seleccionar.
    """
    by_case: Dict[str, List[int]] = {}
    for idx, row in enumerate(meta_rows):
        by_case.setdefault(str(row.get("case", "")), []).append(idx)

    keep: List[int] = []
    for indices in by_case.values():
        n_keep = max(1, int(round(len(indices) * fraction)))
        if n_keep >= len(indices):
            keep.extend(indices)
            continue
        step = len(indices) / float(n_keep)
        keep.extend(indices[int(i * step)] for i in range(n_keep))
    return sorted(set(keep))


def build_model(hparams: Dict[str, Any], prefix_size: int, checkpoint_path: Path, device: torch.device):
    from train import ClipCaptionPrefix, MappingType  # noqa: WPS433

    mapping_enum = MappingType.Transformer if hparams["mapping_type"] == "transformer" else MappingType.MLP
    model = ClipCaptionPrefix(
        prefix_length=hparams["prefix_length"],
        clip_length=hparams["prefix_length_clip"],
        prefix_size=prefix_size,
        num_layers=hparams["num_layers"],
        mapping_type=mapping_enum,
    )
    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def generate_greedy_batch(
    model,
    prefix_embeds: torch.Tensor,
    stop_token_id: int,
    max_len: int,
) -> torch.Tensor:
    """Greedy decoding en batch sobre los prefijos ya proyectados. Devuelve [B, T]."""
    outputs = model.gpt(inputs_embeds=prefix_embeds, use_cache=True, return_dict=True)
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1)

    generated = [next_token]
    finished = next_token.eq(stop_token_id)

    for _ in range(max_len - 1):
        if bool(finished.all()):
            break
        token_embed = model.gpt.transformer.wte(next_token).unsqueeze(1)
        outputs = model.gpt(
            inputs_embeds=token_embed,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        # Las secuencias ya cerradas se rellenan con el stop token y se recortan al decodificar.
        next_token = torch.where(finished, torch.full_like(next_token, stop_token_id), next_token)
        generated.append(next_token)
        finished = finished | next_token.eq(stop_token_id)

    return torch.stack(generated, dim=1)


def decode_until_stop(tokenizer: GPT2Tokenizer, token_ids: Sequence[int], stop_token_id: int) -> str:
    cut = list(token_ids)
    for position, token_id in enumerate(cut):
        if token_id == stop_token_id:
            cut = cut[: position + 1]
            break
    return tokenizer.decode(cut).strip()


@torch.no_grad()
def generate_captions(
    model,
    tokenizer: GPT2Tokenizer,
    prefixes: torch.Tensor,
    hparams: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    max_len: int,
    stop_token_id: int,
    beam_size: int,
) -> List[str]:
    prefix_length = hparams["prefix_length"]
    captions: List[str] = []

    if beam_size and beam_size > 1:
        from validation import generate_beam_with_ids  # noqa: WPS433

        for start in tqdm(range(prefixes.shape[0]), desc="beam", leave=False):
            prefix = prefixes[start : start + 1].to(device)
            embed = model.clip_project(prefix).view(1, prefix_length, model.gpt_embedding_size)
            text, _ = generate_beam_with_ids(
                model, tokenizer, beam_size=beam_size, embed=embed, entry_length=max_len
            )
            captions.append(text.strip())
        return captions

    total_batches = (prefixes.shape[0] + batch_size - 1) // batch_size
    for start in tqdm(range(0, prefixes.shape[0], batch_size), total=total_batches, desc="greedy", leave=False):
        batch = prefixes[start : start + batch_size].to(device)
        embeds = model.clip_project(batch).view(-1, prefix_length, model.gpt_embedding_size)
        tokens = generate_greedy_batch(model, embeds, stop_token_id, max_len)
        for row in tokens.cpu().tolist():
            captions.append(decode_until_stop(tokenizer, row, stop_token_id))
    return captions


def build_metric_rows(
    generated: Sequence[str],
    gt_captions: Sequence[str],
    meta_rows: Optional[Sequence[Dict[str, str]]],
    fold_name: str,
) -> List[Dict[str, Any]]:
    """Filas con el mismo shape que espera evaluate_fold_models."""
    rows: List[Dict[str, Any]] = []
    for idx, (pred, gt) in enumerate(zip(generated, gt_captions)):
        meta = meta_rows[idx] if meta_rows is not None else {}
        rows.append(
            {
                "fold": fold_name,
                "sample_id": meta.get("sample_id", str(idx)),
                "label": meta.get("label", ""),
                "case": meta.get("case", ""),
                "image_path": meta.get("image_path", ""),
                "caption_gt": gt,
                "generated_caption": pred,
            }
        )
    return rows


def evaluate_fold(
    args: argparse.Namespace,
    model_root: Path,
    fold_name: str,
    hparams: Dict[str, Any],
    tokenizer: GPT2Tokenizer,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    fold_dir = model_root / "folds" / fold_name
    val_pkl = fold_weights_dir(model_root, fold_name) / "val.pkl"
    val_csv = fold_dir / "val.csv"
    train_dir = fold_dir / "train"
    checkpoint_prefix = f"positive_vs_negative_{fold_name}"
    out_csv = fold_dir / OUTPUT_CSV_NAME

    if not val_pkl.exists():
        print(f"[SKIP] {model_root.name}/{fold_name}: no existe {to_repo_relative(val_pkl)}")
        return None
    if out_csv.exists() and not args.overwrite:
        print(f"[SKIP] {model_root.name}/{fold_name}: ya existe {to_repo_relative(out_csv)} (usa --overwrite)")
        return None

    checkpoints = list_checkpoints(train_dir, checkpoint_prefix)
    end_epoch = args.end_epoch if args.end_epoch is not None else checkpoints[-1][0]
    selected = [(epoch, path) for epoch, path in checkpoints if args.start_epoch <= epoch <= end_epoch]
    if not selected:
        print(f"[SKIP] {model_root.name}/{fold_name}: sin checkpoints en el rango pedido.")
        return None

    prefixes, gt_captions = load_val_prefixes(val_pkl)
    meta_rows = align_with_val_csv(val_csv, gt_captions)

    if args.sample_fraction < 1.0:
        if meta_rows is None:
            print("[WARN] --sample_fraction necesita val.csv alineado; se usan todos los frames.")
        else:
            keep = subsample_by_case(meta_rows, args.sample_fraction)
            prefixes = prefixes[keep]
            gt_captions = [gt_captions[i] for i in keep]
            meta_rows = [meta_rows[i] for i in keep]
            print(
                f"  submuestreo: {len(keep)} frames "
                f"({args.sample_fraction:.0%} de cada caso, balance preservado)"
            )

    prefix_size = int(prefixes.shape[-1])
    stop_token_id = tokenizer.encode(args.stop_token)[0]
    decoding = f"beam{args.beam_size}" if args.beam_size and args.beam_size > 1 else "greedy"

    rows: List[Dict[str, Any]] = []
    for epoch, checkpoint_path in selected:
        model = build_model(hparams, prefix_size, checkpoint_path, device)
        generated = generate_captions(
            model=model,
            tokenizer=tokenizer,
            prefixes=prefixes,
            hparams=hparams,
            device=device,
            batch_size=args.batch_size,
            max_len=args.max_len,
            stop_token_id=stop_token_id,
            beam_size=args.beam_size,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        metric_rows = build_metric_rows(generated, gt_captions, meta_rows, fold_name)
        binary = binary_metrics(metric_rows)
        lesion = lesion_level_metrics(metric_rows) if meta_rows is not None else {}

        row = {
            "epoch": epoch,
            "checkpoint": to_repo_relative(checkpoint_path),
            "n_samples": binary["num_rows_used"],
            "tp": binary["tp"],
            "tn": binary["tn"],
            "fp": binary["fp"],
            "fn": binary["fn"],
            "accuracy": binary["accuracy"],
            "precision": binary["precision"],
            "recall": binary["recall"],
            "f1": binary["f1"],
            "specificity": binary["specificity"],
            "n_lesions": lesion.get("n_lesions", ""),
            "n_detected_50pct": lesion.get("n_detected_50pct", ""),
            "lesion_detection_rate_any": lesion.get("lesion_detection_rate_any", ""),
            "lesion_detection_rate_50pct": lesion.get("lesion_detection_rate_50pct", ""),
            "decoding": decoding,
        }
        rows.append(row)
        line = (
            f"  epoca {epoch:03d}: f1={binary['f1']:.4f} recall={binary['recall']:.4f} "
            f"prec={binary['precision']:.4f} acc={binary['accuracy']:.4f}"
        )
        if "lesion_detection_rate_50pct" in lesion:
            line += f" det50={lesion['lesion_detection_rate_50pct']:.4f}"
        print(line)

    write_csv(out_csv, rows, CSV_COLUMNS)

    def metric_value(row: Dict[str, Any]) -> Optional[float]:
        try:
            raw = row.get(args.metric, "")
            return None if raw == "" or raw is None else float(raw)
        except (TypeError, ValueError):
            return None

    scored = [(row, metric_value(row)) for row in rows]
    usable = [(row, value) for row, value in scored if value is not None]
    if not usable:
        # Sin este guard, max() sobre una columna vacia elegiria la primera epoca
        # en silencio. Pasa si se pide una metrica por lesion y val.csv no alineo.
        raise ValueError(
            f"La columna '{args.metric}' quedo vacia en {to_repo_relative(out_csv)}. "
            "Las metricas por lesion necesitan val.csv alineado con val.pkl; "
            "usa --metric f1 (frame-level) o revisa los avisos de desalineacion."
        )

    best_row, best_value = max(usable, key=lambda item: item[1])
    print(
        f"  -> mejor epoca por {args.metric}: {best_row['epoch']} "
        f"({args.metric}={best_value:.4f}) | {to_repo_relative(out_csv)}"
    )

    return {
        "fold": fold_name,
        "csv": to_repo_relative(out_csv),
        "epochs_evaluated": len(rows),
        "metric": args.metric,
        "best_epoch": best_row["epoch"],
        "best_metric_value": best_value,
        "best_checkpoint": best_row["checkpoint"],
        "decoding": decoding,
        "lesion_metrics_available": meta_rows is not None,
    }


def main() -> None:
    args = parse_args()

    if args.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    fold_root = resolve_repo_path(args.fold_root)
    if not fold_root.exists():
        raise FileNotFoundError(f"No existe fold_root: {fold_root}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"Dispositivo: {device} | fold_root: {to_repo_relative(fold_root)}")

    report: List[Dict[str, Any]] = []
    for subdir in args.models:
        model_root = fold_root / subdir
        if not model_root.exists():
            print(f"[SKIP] no existe {to_repo_relative(model_root)}")
            continue

        training = load_training_config(model_root)
        hparams = resolve_model_hparams(args, training)
        print(f"\n===== {subdir} ===== hiperparametros del mapper: {hparams}")

        for fold_name in args.folds:
            print(f"\n[{subdir}/{fold_name}]")
            result = evaluate_fold(args, model_root, fold_name, hparams, tokenizer, device)
            if result is not None:
                report.append({"model_dir": subdir, **result})

    if not report:
        print("\nNo se genero ninguna metrica.")
        return

    summary_path = fold_root / "val_task_metric_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "fold_root": to_repo_relative(fold_root),
                "selection_metric": args.metric,
                "note": (
                    "Metricas de SELECCION (decoding greedy sobre embeddings cacheados). "
                    "Las tablas del reporte se calculan con beam search via test.py."
                ),
                "runs": report,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nResumen: {to_repo_relative(summary_path)}")
    print(
        "Ahora ejecuta:\n"
        f"  python scripts/revalidate_epoch.py --source_root {to_repo_relative(fold_root)} "
        f"--output_root {to_repo_relative(fold_root)}_f1 --checkpoint_policy f1 --f1_metric {args.metric}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Genera graficas de perdida train/val por epoca para cada modelo en fold/2fold."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - entorno
    print(f"[ERROR] No se pudo importar matplotlib: {exc}")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "fold" / "2fold"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "plots" / "losses"
DEFAULT_MODELS = ("biomedclip", "vit", "resnet")

TRAIN_COLUMNS = ("train_loss", "training_loss", "loss", "mean_loss")
VAL_COLUMNS = ("val_loss", "valid_loss", "validation_loss", "dev_loss")
EPOCH_COLUMNS = ("epoch", "epoca", "epochs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea una grafica por modelo con perdida de entrenamiento y "
            "validacion por epoca desde fold/2fold."
        )
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=f"Raiz de entrada (default: {DEFAULT_INPUT_ROOT})",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Modelos a procesar (default: biomedclip vit resnet)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="DPI para PNG")
    parser.add_argument(
        "--strict-val",
        action="store_true",
        help="Falla si no se encuentra val_loss para un modelo.",
    )
    return parser.parse_args()


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _find_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    normalized = {name.strip().lower(): name for name in fieldnames if name}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def load_loss_csv(path: Path) -> Dict[int, Dict[str, float]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            return {}

        epoch_col = _find_column(fieldnames, EPOCH_COLUMNS)
        train_col = _find_column(fieldnames, TRAIN_COLUMNS)
        val_col = _find_column(fieldnames, VAL_COLUMNS)

        # Fallback comun: archivo con columnas epoch y mean_loss.
        if train_col is None and _find_column(fieldnames, ("mean_loss",)) is not None:
            train_col = _find_column(fieldnames, ("mean_loss",))

        if epoch_col is None:
            epoch_col = fieldnames[0]

        loss_by_epoch: Dict[int, Dict[str, float]] = {}
        for row in reader:
            epoch = _safe_int(row.get(epoch_col))
            if epoch is None:
                continue

            train_value = _safe_float(row.get(train_col)) if train_col else None
            val_value = _safe_float(row.get(val_col)) if val_col else None

            if train_value is None and val_value is None:
                continue

            payload = loss_by_epoch.setdefault(epoch, {})
            if train_value is not None:
                payload["train_loss"] = train_value
            if val_value is not None:
                payload["val_loss"] = val_value

        return loss_by_epoch


def load_fold_losses(fold_dir: Path) -> Tuple[Dict[int, float], Dict[int, float]]:
    train_dir = fold_dir / "train"
    candidate_csvs: List[Path] = []
    if train_dir.exists():
        train_epoch_csvs = sorted(train_dir.glob("*loss*epoch*.csv"))
        candidate_csvs.extend(train_epoch_csvs if train_epoch_csvs else sorted(train_dir.glob("*.csv")))

    # Algunos pipelines guardan val_loss por epoca en el directorio del fold.
    candidate_csvs.extend(sorted(fold_dir.glob("*val*loss*epoch*.csv")))

    # Deduplicar conservando orden.
    unique_csvs: List[Path] = []
    seen: set = set()
    for csv_path in candidate_csvs:
        key = str(csv_path.resolve()) if csv_path.exists() else str(csv_path)
        if key in seen:
            continue
        seen.add(key)
        unique_csvs.append(csv_path)

    if not unique_csvs:
        return {}, {}

    train_losses: Dict[int, float] = {}
    val_losses: Dict[int, float] = {}

    for csv_path in unique_csvs:
        data = load_loss_csv(csv_path)
        if not data:
            continue
        for epoch, payload in data.items():
            if "train_loss" in payload:
                train_losses[epoch] = payload["train_loss"]
            if "val_loss" in payload:
                val_losses[epoch] = payload["val_loss"]

    return train_losses, val_losses


def _extract_epoch_from_checkpoint(checkpoint: object) -> Optional[int]:
    text = str(checkpoint or "").strip()
    if not text:
        return None

    name = Path(text).name
    match = re.search(r"-(\d+)\.pt$", name)
    if match:
        return _safe_int(match.group(1))

    match = re.search(r"(\d+)$", name)
    if match:
        return _safe_int(match.group(1))
    return None


def load_val_from_model_summary(model_root: Path) -> Dict[str, Dict[int, float]]:
    summary_path = model_root / "summary.json"
    if not summary_path.exists():
        return {}

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    execution_report = payload.get("execution_report")
    if not isinstance(execution_report, list):
        return {}

    val_by_fold: Dict[str, Dict[int, float]] = {}
    for row in execution_report:
        if not isinstance(row, dict):
            continue

        fold_name = str(row.get("fold", "")).strip()
        mean_val_loss = _safe_float(row.get("mean_val_loss"))
        if not fold_name or mean_val_loss is None:
            continue

        epoch = _extract_epoch_from_checkpoint(row.get("checkpoint"))
        if epoch is None:
            fold_train, _ = load_fold_losses(model_root / "folds" / fold_name)
            if fold_train:
                epoch = max(fold_train)
        if epoch is None:
            continue

        val_by_fold.setdefault(fold_name, {})[epoch] = mean_val_loss

    return val_by_fold


def aggregate_mean(series_list: List[Dict[int, float]]) -> Dict[int, float]:
    collector: Dict[int, List[float]] = {}
    for series in series_list:
        for epoch, value in series.items():
            collector.setdefault(epoch, []).append(value)

    aggregated: Dict[int, float] = {}
    for epoch, values in collector.items():
        aggregated[epoch] = float(np.mean(values))
    return aggregated


def save_aggregate_csv(path: Path, train_mean: Dict[int, float], val_mean: Dict[int, float]) -> None:
    all_epochs = sorted(set(train_mean) | set(val_mean))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        for epoch in all_epochs:
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_mean.get(epoch, ""),
                    "val_loss": val_mean.get(epoch, ""),
                }
            )


def _plot_fold_lines(ax: plt.Axes, folds: Dict[str, Dict[int, float]], color: str, label_prefix: str) -> None:
    for fold_name, fold_data in sorted(folds.items()):
        if not fold_data:
            continue
        epochs = sorted(fold_data)
        values = [fold_data[e] for e in epochs]
        ax.plot(
            epochs,
            values,
            linestyle="--",
            linewidth=1.0,
            alpha=0.35,
            color=color,
            label=f"{label_prefix} ({fold_name})",
        )


def plot_model_losses(
    model_name: str,
    output_path: Path,
    train_folds: Dict[str, Dict[int, float]],
    val_folds: Dict[str, Dict[int, float]],
    dpi: int,
) -> None:
    train_mean = aggregate_mean(list(train_folds.values()))
    val_mean = aggregate_mean(list(val_folds.values()))

    if not train_mean:
        raise RuntimeError(f"{model_name}: no se encontraron perdidas de entrenamiento por epoca.")

    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    _plot_fold_lines(ax, train_folds, color="#1f77b4", label_prefix="Train")
    if val_mean:
        _plot_fold_lines(ax, val_folds, color="#d62728", label_prefix="Val")

    train_epochs = sorted(train_mean)
    ax.plot(
        train_epochs,
        [train_mean[e] for e in train_epochs],
        color="#1f77b4",
        linewidth=2.4,
        marker="o",
        markersize=3.6,
        label="Train (promedio folds)",
    )

    if val_mean:
        val_epochs = sorted(val_mean)
        ax.plot(
            val_epochs,
            [val_mean[e] for e in val_epochs],
            color="#d62728",
            linewidth=2.4,
            marker="o",
            markersize=3.6,
            label="Validacion (promedio folds)",
        )
        if len(val_epochs) == 1:
            ax.text(
                0.5,
                0.06,
                f"val_loss disponible solo para la epoca {val_epochs[0]} (checkpoint evaluado).",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
                bbox={"boxstyle": "round,pad=0.25", "fc": "#f7f7f7", "ec": "#cccccc"},
            )
        elif len(val_epochs) < len(train_epochs):
            ax.text(
                0.5,
                0.06,
                "val_loss disponible en subconjunto de epocas (segun checkpoints guardados).",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color="#555555",
                bbox={"boxstyle": "round,pad=0.25", "fc": "#f7f7f7", "ec": "#cccccc"},
            )
    else:
        ax.text(
            0.5,
            0.06,
            "No se encontro val_loss en los CSV de entrenamiento.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#f7f7f7", "ec": "#cccccc"},
        )

    ax.set_title(f"{model_name.upper()} - Perdida por epoca")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.75)

    handles, labels = ax.get_legend_handles_labels()
    unique: Dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(list(unique.values()), list(unique.keys()), fontsize=8, loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def process_model(model_root: Path, output_root: Path, dpi: int, strict_val: bool) -> Tuple[Path, Path]:
    model_name = model_root.name
    fold_dirs = sorted((model_root / "folds").glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"{model_name}: no se encontraron folds en {model_root / 'folds'}")

    train_folds: Dict[str, Dict[int, float]] = {}
    val_folds: Dict[str, Dict[int, float]] = {}

    for fold_dir in fold_dirs:
        train_losses, val_losses = load_fold_losses(fold_dir)
        if train_losses:
            train_folds[fold_dir.name] = train_losses
        if val_losses:
            val_folds[fold_dir.name] = val_losses

    # Fallback para 2fold: val_loss guardado en summary.json (solo checkpoint evaluado).
    summary_val_folds = load_val_from_model_summary(model_root)
    for fold_name, fold_vals in summary_val_folds.items():
        destination = val_folds.setdefault(fold_name, {})
        for epoch, value in fold_vals.items():
            destination.setdefault(epoch, value)

    if not train_folds:
        raise RuntimeError(f"{model_name}: no se pudieron leer perdidas de entrenamiento.")
    if strict_val:
        missing_val_folds = sorted(set(train_folds) - set(val_folds))
        if missing_val_folds:
            missing_csv = ", ".join(missing_val_folds)
            raise RuntimeError(
                f"{model_name}: no se encontro val_loss para folds: {missing_csv}. "
                "Revisa CSV por epoca o summary.json con mean_val_loss."
            )
    plot_path = output_root / f"{model_name}_train_val_loss.png"
    csv_path = output_root / f"{model_name}_train_val_loss.csv"

    plot_model_losses(
        model_name=model_name,
        output_path=plot_path,
        train_folds=train_folds,
        val_folds=val_folds,
        dpi=dpi,
    )
    save_aggregate_csv(
        path=csv_path,
        train_mean=aggregate_mean(list(train_folds.values())),
        val_mean=aggregate_mean(list(val_folds.values())),
    )
    return plot_path, csv_path


def main() -> int:
    args = parse_args()
    input_root = args.input_root
    output_root = args.output_root

    if not input_root.exists():
        print(f"[ERROR] No existe input_root: {input_root}")
        return 1

    failures: List[str] = []
    for model in args.models:
        model_root = input_root / model
        if not model_root.exists():
            failures.append(f"{model}: no existe {model_root}")
            continue
        try:
            plot_path, csv_path = process_model(
                model_root=model_root,
                output_root=output_root,
                dpi=args.dpi,
                strict_val=args.strict_val,
            )
            print(f"[OK] {model}: grafica -> {plot_path}")
            print(f"[OK] {model}: resumen CSV -> {csv_path}")
        except Exception as exc:
            failures.append(f"{model}: {exc}")

    if failures:
        print("[WARN] Algunos modelos no se pudieron procesar:")
        for item in failures:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

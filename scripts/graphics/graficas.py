#!/usr/bin/env python3
"""Genera graficas agregadas para experimentos 2-fold.

Salida:
1) Una sola imagen con matrices de confusion de todos los modelos.
2) Una sola imagen con PCA por frame de todos los modelos.
3) Una sola imagen con loss de entrenamiento/validacion por epoca.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover - entorno
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - entorno
    print(f"[ERROR] No se pudo importar matplotlib: {exc}")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "fold" / "2fold"
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "plots"
DEFAULT_MODELS = ("biomedclip", "resnet", "vit")

LABELS_ORDER = ("negative", "positive")
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS_ORDER)}

# Reglas para inferir etiqueta desde generated_caption.
NEGATIVE_PATTERNS = (
    re.compile(r"\bno\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+any\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bno\s+evidence\s+of\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bnormal\s+colonoscopy\b", flags=re.IGNORECASE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea imagenes combinadas (confusion y PCA) para "
            "biomedclip/vit/resnet desde fold/2fold."
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
        help=f"Raiz de salida (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Modelos a procesar (default: biomedclip vit resnet)",
    )
    parser.add_argument("--dpi", type=int, default=220, help="DPI para PNG")
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def normalize_label(raw_label: str) -> Optional[str]:
    text = (raw_label or "").strip().lower()
    if not text:
        return None
    if text in {"positive", "pos", "1", "polyp", "with_polyp"}:
        return "positive"
    if text in {"negative", "neg", "0", "no_polyp", "no-polyp"}:
        return "negative"
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return None


def infer_pred_label_from_caption(generated_caption: str) -> Optional[str]:
    text = (generated_caption or "").strip().lower()
    if not text:
        return None
    for pattern in NEGATIVE_PATTERNS:
        if pattern.search(text):
            return "negative"
    return "positive"


def compute_aggregated_confusion(pred_rows: Iterable[Dict[str, str]]) -> Dict[str, Any]:
    cm = np.zeros((2, 2), dtype=np.int64)
    used_rows = 0
    skipped_rows = 0

    for row in pred_rows:
        true_label = normalize_label(row.get("label", ""))
        pred_label = infer_pred_label_from_caption(row.get("generated_caption", ""))
        if true_label not in LABEL_TO_INDEX or pred_label not in LABEL_TO_INDEX:
            skipped_rows += 1
            continue
        cm[LABEL_TO_INDEX[true_label], LABEL_TO_INDEX[pred_label]] += 1
        used_rows += 1

    return {"cm": cm, "used_rows": used_rows, "skipped_rows": skipped_rows}


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: Optional[float]) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _row_to_meta(row: Dict[str, str]) -> Dict[str, str]:
    return {
        "sample_id": str(row.get("sample_id", "")),
        "case": str(row.get("case", "unknown") or "unknown"),
        "label": str(row.get("label", "unknown") or "unknown"),
        "image_path": str(row.get("image_path", "")),
    }


def load_val_embeddings_and_meta(val_pkl_path: Path, val_csv_path: Path) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    if torch is None:
        raise RuntimeError(
            "No se pudo importar torch para abrir val.pkl. "
            f"Error: {TORCH_IMPORT_ERROR}"
        )

    with val_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        val_rows = list(csv.DictReader(handle))

    with val_pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "clip_embedding" not in data:
        raise ValueError(f"Estructura invalida en {val_pkl_path}; falta 'clip_embedding'.")

    raw_embedding = data["clip_embedding"]
    captions = data.get("captions", [])

    if isinstance(raw_embedding, torch.Tensor):
        embeddings = raw_embedding.detach().cpu().numpy()
    else:
        embeddings = np.asarray(raw_embedding)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings invalidos en {val_pkl_path}; shape={embeddings.shape}")

    n = embeddings.shape[0]
    meta = [_row_to_meta({}) for _ in range(n)]

    # Caso ideal: val.csv y embeddings alineados.
    if len(val_rows) == n:
        return embeddings, [_row_to_meta(row) for row in val_rows]

    # Fallback: mapear indices de captions.
    if isinstance(captions, list):
        for idx, cap in enumerate(captions):
            if not isinstance(cap, dict):
                continue
            emb_idx = _safe_int(cap.get("clip_embedding"), idx)
            row_idx = _safe_int(cap.get("image_id"), idx)

            if not (0 <= emb_idx < n):
                emb_idx = idx
            if not (0 <= emb_idx < n):
                continue

            if 0 <= row_idx < len(val_rows):
                meta[emb_idx] = _row_to_meta(val_rows[row_idx])
            elif 0 <= idx < len(val_rows):
                meta[emb_idx] = _row_to_meta(val_rows[idx])

    for idx in range(n):
        if meta[idx].get("case", "unknown") == "unknown" and idx < len(val_rows):
            meta[idx] = _row_to_meta(val_rows[idx])

    return embeddings, meta


def collect_model_embeddings(model_root: Path) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    fold_dirs = sorted((model_root / "folds").glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No se encontraron folds en {model_root / 'folds'}")

    chunks: List[np.ndarray] = []
    all_meta: List[Dict[str, str]] = []

    for fold_dir in fold_dirs:
        val_pkl = fold_dir / "data" / "val.pkl"
        val_csv = fold_dir / "val.csv"
        if not val_pkl.exists() or not val_csv.exists():
            print(f"[WARN] {model_root.name}/{fold_dir.name}: falta val.pkl o val.csv")
            continue

        fold_embeddings, fold_meta = load_val_embeddings_and_meta(val_pkl, val_csv)
        chunks.append(fold_embeddings)
        for local_idx, item in enumerate(fold_meta):
            row = dict(item)
            row["fold"] = fold_dir.name
            row["frame_idx_in_fold"] = str(local_idx)
            all_meta.append(row)

    if not chunks:
        raise RuntimeError(f"No hubo embeddings validos para {model_root.name}")

    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.shape[0] != len(all_meta):
        raise RuntimeError(
            f"Desalineacion embeddings/meta en {model_root.name}: "
            f"{embeddings.shape[0]} vs {len(all_meta)}"
        )
    return embeddings, all_meta


def pca_2d(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"PCA espera matriz 2D, pero recibio shape={x.shape}")
    if x.shape[0] == 0:
        raise ValueError("No hay embeddings para PCA.")

    x_centered = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 1:
        points = np.hstack([x_centered, np.zeros((1, 1), dtype=np.float64)])
        return points[:, :2], np.array([1.0, 0.0], dtype=np.float64)

    _, singular_values, vt = np.linalg.svd(x_centered, full_matrices=False)
    projected = x_centered @ vt.T
    if projected.shape[1] < 2:
        projected = np.hstack([projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)])

    variances = (singular_values ** 2) / max(x.shape[0] - 1, 1)
    total_var = float(variances.sum()) if variances.size else 0.0
    if total_var <= 0:
        explained = np.array([0.0, 0.0], dtype=np.float64)
    else:
        r1 = float(variances[0] / total_var) if variances.size >= 1 else 0.0
        r2 = float(variances[1] / total_var) if variances.size >= 2 else 0.0
        explained = np.array([r1, r2], dtype=np.float64)

    return projected[:, :2], explained


def gather_prediction_rows(model_root: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in sorted(model_root.glob("folds/fold_*/inference/val_predictions.csv")):
        if path.exists():
            rows.extend(read_csv_rows(path))
    return rows


def _read_loss_series(path: Path, value_key: str) -> Dict[int, float]:
    series: Dict[int, float] = {}
    for row in read_csv_rows(path):
        epoch = _safe_int(row.get("epoch"), -1)
        value = _safe_float(row.get(value_key), None)
        if epoch < 0 or value is None:
            continue
        series[epoch] = value
    return series


def collect_model_loss_data(model_root: Path) -> Dict[str, Any]:
    fold_dirs = sorted((model_root / "folds").glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No se encontraron folds en {model_root / 'folds'}")

    fold_curves: List[Dict[str, Any]] = []
    train_values_by_epoch: Dict[int, List[float]] = {}
    val_values_by_epoch: Dict[int, List[float]] = {}

    for fold_dir in fold_dirs:
        val_loss_csv = fold_dir / "val_loss_per_epoch.csv"
        train_csv_candidates = sorted((fold_dir / "train").glob("*_loss_per_epoch.csv"))
        train_loss_csv = train_csv_candidates[0] if train_csv_candidates else None

        if train_loss_csv is None or not val_loss_csv.exists():
            print(f"[WARN] {model_root.name}/{fold_dir.name}: falta train_loss o val_loss por epoca")
            continue

        train_series = _read_loss_series(train_loss_csv, "mean_loss")
        val_series = _read_loss_series(val_loss_csv, "val_loss")
        common_epochs = sorted(set(train_series).intersection(val_series))

        if not common_epochs:
            print(f"[WARN] {model_root.name}/{fold_dir.name}: sin epocas comunes train/val")
            continue

        epochs = np.asarray(common_epochs, dtype=np.int64)
        train_loss = np.asarray([train_series[e] for e in common_epochs], dtype=np.float64)
        val_loss = np.asarray([val_series[e] for e in common_epochs], dtype=np.float64)
        fold_curves.append(
            {
                "fold": fold_dir.name,
                "epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        for e in common_epochs:
            train_values_by_epoch.setdefault(e, []).append(train_series[e])
            val_values_by_epoch.setdefault(e, []).append(val_series[e])

    if not fold_curves:
        raise RuntimeError(f"No hubo curvas de loss validas para {model_root.name}")

    epochs_common = sorted(set(train_values_by_epoch).intersection(val_values_by_epoch))
    if not epochs_common:
        raise RuntimeError(f"No hubo epocas comunes para agregar curvas de loss en {model_root.name}")

    epochs = np.asarray(epochs_common, dtype=np.int64)
    train_mean = np.asarray([np.mean(train_values_by_epoch[e]) for e in epochs_common], dtype=np.float64)
    val_mean = np.asarray([np.mean(val_values_by_epoch[e]) for e in epochs_common], dtype=np.float64)
    train_std = np.asarray([np.std(train_values_by_epoch[e]) for e in epochs_common], dtype=np.float64)
    val_std = np.asarray([np.std(val_values_by_epoch[e]) for e in epochs_common], dtype=np.float64)

    return {
        "epochs": epochs,
        "train_mean": train_mean,
        "val_mean": val_mean,
        "train_std": train_std,
        "val_std": val_std,
        "fold_curves": fold_curves,
        "folds_used": len(fold_curves),
        "folds_total": len(fold_dirs),
    }


def _build_model_data(
    model_name: str,
    model_root: Path,
    output_root: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model_out = output_root / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    pred_rows = gather_prediction_rows(model_root)
    confusion = compute_aggregated_confusion(pred_rows)
    cm = confusion["cm"]

    embeddings, meta = collect_model_embeddings(model_root)
    points_2d, explained = pca_2d(embeddings)
    loss_data = collect_model_loss_data(model_root)

    pca_rows: List[Dict[str, Any]] = []
    for idx, (item, point) in enumerate(zip(meta, points_2d)):
        pca_rows.append(
            {
                "global_idx": idx,
                "fold": item.get("fold", ""),
                "frame_idx_in_fold": item.get("frame_idx_in_fold", ""),
                "sample_id": item.get("sample_id", ""),
                "case": item.get("case", ""),
                "label": item.get("label", ""),
                "image_path": item.get("image_path", ""),
                "pc1": float(point[0]),
                "pc2": float(point[1]),
            }
        )
    pca_csv = model_out / "pca_points_pre_gpt2.csv"
    write_csv(
        pca_csv,
        pca_rows,
        fieldnames=[
            "global_idx",
            "fold",
            "frame_idx_in_fold",
            "sample_id",
            "case",
            "label",
            "image_path",
            "pc1",
            "pc2",
        ],
    )

    plot_payload = {
        "model": model_name,
        "cm": cm,
        "cm_used_rows": int(confusion["used_rows"]),
        "points_2d": points_2d,
        "explained": explained,
        "meta": meta,
        "loss_epochs": loss_data["epochs"],
        "loss_train_mean": loss_data["train_mean"],
        "loss_val_mean": loss_data["val_mean"],
        "loss_train_std": loss_data["train_std"],
        "loss_val_std": loss_data["val_std"],
        "loss_fold_curves": loss_data["fold_curves"],
        "loss_folds_used": int(loss_data["folds_used"]),
        "loss_folds_total": int(loss_data["folds_total"]),
    }

    summary_payload = {
        "model": model_name,
        "status": "ok",
        "output_dir": str(model_out),
        "prediction_rows_total": len(pred_rows),
        "prediction_rows_used_in_cm": int(confusion["used_rows"]),
        "prediction_rows_skipped_in_cm": int(confusion["skipped_rows"]),
        "confusion_matrix_counts": {
            "true_negative_pred_negative": int(cm[0, 0]),
            "true_negative_pred_positive": int(cm[0, 1]),
            "true_positive_pred_negative": int(cm[1, 0]),
            "true_positive_pred_positive": int(cm[1, 1]),
        },
        "n_frames_for_embeddings": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "pca_explained_variance_ratio": {
            "pc1": float(explained[0]),
            "pc2": float(explained[1]),
        },
        "loss_curve_summary": {
            "folds_used": int(loss_data["folds_used"]),
            "folds_total": int(loss_data["folds_total"]),
            "n_epochs": int(len(loss_data["epochs"])),
            "final_epoch": int(loss_data["epochs"][-1]) if len(loss_data["epochs"]) else None,
            "final_train_loss_mean": (
                float(loss_data["train_mean"][-1]) if len(loss_data["train_mean"]) else None
            ),
            "final_val_loss_mean": (
                float(loss_data["val_mean"][-1]) if len(loss_data["val_mean"]) else None
            ),
        },
        "exports": {
            "pca_points_csv": str(pca_csv),
        },
    }
    return plot_payload, summary_payload


def _as_axes_list(axes: Any) -> List[Any]:
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    return [axes]


def plot_confusion_matrices_combined(
    plot_items: Sequence[Dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> None:
    if not plot_items:
        return

    n = len(plot_items)
    # Para 3 modelos, forzamos layout 2x2 y usamos la 4ta celda para la barra.
    if n == 3:
        nrows, ncols = 2, 2
    else:
        ncols = max(1, int(np.ceil(np.sqrt(n))))
        nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.8 * nrows), squeeze=False)
    ax_list = _as_axes_list(axes)
    vmax = max(max(int(item["cm"].max()), 1) for item in plot_items)
    images = []

    for ax, item in zip(ax_list, plot_items):
        cm = item["cm"]
        used_rows = int(item["cm_used_rows"])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        images.append(im)

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = int(cm[i, j])
                ratio = cm_norm[i, j]
                text = f"{count}\n({ratio:.1%})"
                color = "white" if count > vmax * 0.5 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

        ax.set_xticks(range(len(LABELS_ORDER)))
        ax.set_yticks(range(len(LABELS_ORDER)))
        ax.set_xticklabels([x.capitalize() for x in LABELS_ORDER], fontsize=9)
        ax.set_yticklabels([x.capitalize() for x in LABELS_ORDER], fontsize=9)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Etiqueta real")
        ax.set_title(f"{item['model']} (N={used_rows})")

    fig.suptitle("Matrices de confusion agregadas por modelo", fontsize=14, y=1.02)
    extra_axes = ax_list[n:]
    if extra_axes:
        colorbar_slot_ax = extra_axes[0]
        colorbar_slot_ax.axis("off")
        for ax in extra_axes[1:]:
            ax.axis("off")
        # Dibujar una barra vertical angosta dentro de la 4ta celda.
        cax = colorbar_slot_ax.inset_axes([0.42, 0.08, 0.16, 0.84])
        cbar = fig.colorbar(images[0], cax=cax, orientation="vertical")
        cbar.ax.set_title("Conteo", fontsize=9, pad=6)
    else:
        fig.colorbar(images[0], ax=ax_list[:n], fraction=0.025, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_pca_combined(
    plot_items: Sequence[Dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> None:
    if not plot_items:
        return

    n = len(plot_items)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 5.1), squeeze=False)
    ax_list = _as_axes_list(axes)

    color_map = {
        "negative": "#2a9d8f",
        "positive": "#e76f51",
        "unknown": "#7f7f7f",
    }

    for ax, item in zip(ax_list, plot_items):
        points = np.asarray(item["points_2d"])
        meta = item["meta"]
        explained = item["explained"]
        labels = [normalize_label(x.get("label", "")) or "unknown" for x in meta]

        for label in ("negative", "positive", "unknown"):
            idx = [i for i, val in enumerate(labels) if val == label]
            if not idx:
                continue
            pts = points[idx]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=8,
                alpha=0.5,
                c=color_map[label],
                label=f"{label} (n={len(idx)})",
                edgecolors="none",
            )

        ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}% var)")
        ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}% var)")
        ax.set_title(item["model"])
        ax.grid(alpha=0.22, linestyle="--", linewidth=0.6)
        ax.legend(loc="best", fontsize=8, frameon=True)

    fig.suptitle("PCA por frame (tokens/embeddings pre-GPT-2) por modelo", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves_grid_combined(
    plot_items: Sequence[Dict[str, Any]],
    out_path: Path,
    dpi: int,
) -> None:
    if not plot_items:
        return

    n = len(plot_items)
    ncols = max(1, int(np.ceil(np.sqrt(n))))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.7 * nrows), squeeze=False)
    ax_list = _as_axes_list(axes)

    train_color = "#1d3557"
    val_color = "#d62828"

    for ax, item in zip(ax_list, plot_items):
        epochs = np.asarray(item["loss_epochs"])
        train_mean = np.asarray(item["loss_train_mean"])
        val_mean = np.asarray(item["loss_val_mean"])
        train_std = np.asarray(item["loss_train_std"])
        val_std = np.asarray(item["loss_val_std"])
        fold_curves = item.get("loss_fold_curves", [])

        for fold_curve in fold_curves:
            fold_epochs = np.asarray(fold_curve.get("epochs", []))
            fold_train = np.asarray(fold_curve.get("train_loss", []))
            fold_val = np.asarray(fold_curve.get("val_loss", []))
            if fold_epochs.size > 0 and fold_train.size == fold_epochs.size:
                ax.plot(fold_epochs, fold_train, color=train_color, alpha=0.20, linewidth=1.0)
            if fold_epochs.size > 0 and fold_val.size == fold_epochs.size:
                ax.plot(fold_epochs, fold_val, color=val_color, alpha=0.20, linewidth=1.0)

        if epochs.size > 0:
            ax.plot(epochs, train_mean, color=train_color, linewidth=2.3, marker="o", markersize=3.5, label="Train")
            ax.plot(epochs, val_mean, color=val_color, linewidth=2.3, marker="o", markersize=3.5, label="Validation")

            if train_std.size == epochs.size:
                ax.fill_between(
                    epochs,
                    train_mean - train_std,
                    train_mean + train_std,
                    color=train_color,
                    alpha=0.12,
                    linewidth=0,
                )
            if val_std.size == epochs.size:
                ax.fill_between(
                    epochs,
                    val_mean - val_std,
                    val_mean + val_std,
                    color=val_color,
                    alpha=0.12,
                    linewidth=0,
                )
        else:
            ax.text(0.5, 0.5, "Sin datos de loss", ha="center", va="center", transform=ax.transAxes, fontsize=10)

        ax.set_title(f"{item['model']} (folds={int(item['loss_folds_used'])})")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
        ax.legend(loc="best", fontsize=8, frameon=True)

    for ax in ax_list[n:]:
        ax.axis("off")

    fig.suptitle("Loss de entrenamiento y validacion por epoca (agregada por modelo)", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] input_root={input_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] models={args.models}")

    if not input_root.exists():
        raise FileNotFoundError(f"No existe input_root: {input_root}")

    summary: Dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "models_requested": list(args.models),
        "models_processed": {},
        "combined_plots": {},
    }

    plot_items: List[Dict[str, Any]] = []
    for model_name in args.models:
        model_root = input_root / model_name
        if not model_root.exists():
            print(f"[WARN] No existe carpeta para modelo '{model_name}': {model_root}")
            summary["models_processed"][model_name] = {
                "model": model_name,
                "status": "missing_model_dir",
                "model_root": str(model_root),
            }
            continue

        print(f"[INFO] Procesando modelo: {model_name}")
        try:
            plot_payload, summary_payload = _build_model_data(
                model_name=model_name,
                model_root=model_root,
                output_root=output_root,
            )
            plot_items.append(plot_payload)
            summary["models_processed"][model_name] = summary_payload
            print(f"[OK] {model_name}: datos listos.")
        except Exception as exc:
            print(f"[ERROR] {model_name}: {exc}")
            summary["models_processed"][model_name] = {
                "model": model_name,
                "status": "error",
                "error": str(exc),
            }

    if plot_items:
        confusion_path = output_root / "confusion_matrices_all_models.png"
        pca_path = output_root / "pca_frames_all_models.png"
        loss_path = output_root / "train_val_loss_all_models.png"

        plot_confusion_matrices_combined(plot_items, confusion_path, args.dpi)
        plot_pca_combined(plot_items, pca_path, args.dpi)
        plot_loss_curves_grid_combined(plot_items, loss_path, args.dpi)

        summary["combined_plots"] = {
            "confusion_matrices": str(confusion_path),
            "pca_frames": str(pca_path),
            "train_val_loss": str(loss_path),
        }
    else:
        summary["combined_plots"] = {
            "status": "no_valid_models_processed"
        }

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[DONE] Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()

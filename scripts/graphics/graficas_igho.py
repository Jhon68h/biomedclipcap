#!/usr/bin/env python3
"""Genera graficas exploratorias del dataset IGHO.

Produce exactamente dos figuras, alineadas con lo que ya existe para el
dataset `sun` en `plots_multidataset/sun/` (ver `scripts/graphics/
complementary_graph.py`) y con los resultados obtenidos en la investigacion
de `reporte_resultados.md`:

1. `lesion_size_distribution.png`
   Igual formato que `plots_multidataset/sun/lesion_size_distribution.png`:
   histograma tipo "step" del tamano de lesion (mm) real (ground truth) vs.
   el tamano extraido de las captions generadas por cada modelo
   (biomedclip, resnet, vit).
   - Ground truth: columna `size` de `igho/igho_dataset.csv` (una lesion
     por video, 8 videos -> 8 valores). No se calcula un area a partir de
     width/height porque esas columnas no existen en el dataset.
   - Predicho: se extrae con la misma expresion regular que usa
     `scripts/graphics/complementary_graph.py::extract_size` (patron
     `"<numero> mm"`) sobre `generated_caption` de TODOS los
     `igho/videos/video_*/{model}/fold_*/predictions.csv`.

2. `confusion_matrix.png`
   Matriz de confusion (positivo/negativo, frame-level) por modelo, una al
   lado de la otra. En vez de recalcular la clasificacion binaria desde las
   predicciones crudas (lo que duplicaria y podria desincronizarse de la
   logica de `igho/metrics/igho_metrics.py`), se construye a partir de los
   totales YA CALCULADOS por ese script y guardados en
   `igho/metrics/promedio_todos_los_videos.csv` (columnas `total_tp`,
   `total_fp`, `total_fn`, `total_tn`, agregadas sobre los 8 videos y los 2
   folds). Esto mantiene la matriz consistente con las cifras de accuracy/
   recall ya reportadas en `reporte_resultados.md`.

No se ejecuta ningun modelo ni se decodifica ningun video: solo se leen
CSV ya existentes en el repositorio.

Salida exclusiva en:
    plots_multidataset/igho/
No se toca ningun otro dataset (p. ej. plots_multidataset/sun/).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_CSV = REPO_ROOT / "igho" / "igho_dataset.csv"
DEFAULT_VIDEOS_DIR = REPO_ROOT / "igho" / "videos"
DEFAULT_METRICS_CSV = REPO_ROOT / "igho" / "metrics" / "promedio_todos_los_videos.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "plots_multidataset" / "igho"
DEFAULT_DPI = 220
DEFAULT_MODELS = ("biomedclip", "resnet", "vit")

COL_SIZE = "size"
LABELS_ORDER = ("negative", "positive")

SIZE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*mm", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera lesion_size_distribution.png y confusion_matrix.png para IGHO, "
            "en el mismo formato usado para el dataset sun."
        )
    )
    parser.add_argument("--dataset_csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--videos_dir", type=Path, default=DEFAULT_VIDEOS_DIR)
    parser.add_argument("--metrics_csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser.parse_args()


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: object) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None


def extract_size(text: object) -> Optional[float]:
    match = SIZE_PATTERN.search(str(text or ""))
    return float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# 1) lesion_size_distribution.png
# ---------------------------------------------------------------------------

def load_ground_truth_sizes(dataset_csv: Path) -> List[float]:
    if not dataset_csv.exists():
        raise FileNotFoundError(f"No se encontro el dataset IGHO en: {dataset_csv}")
    rows = load_csv_rows(dataset_csv)
    sizes = [safe_float(row.get(COL_SIZE)) for row in rows]
    return [s for s in sizes if s is not None]


def load_predicted_sizes(videos_dir: Path, model: str) -> List[float]:
    prediction_files = sorted(videos_dir.glob(f"video_*/{model}/fold_*/predictions.csv"))
    sizes: List[float] = []
    for path in prediction_files:
        for row in load_csv_rows(path):
            size = extract_size(row.get("generated_caption"))
            if size is not None:
                sizes.append(size)
    return sizes


def plot_lesion_size_distribution(
    gt_sizes: Sequence[float],
    predicted_sizes_by_model: Dict[str, List[float]],
    out_path: Path,
    dpi: int,
) -> None:
    all_sizes = list(gt_sizes)
    for sizes in predicted_sizes_by_model.values():
        all_sizes.extend(sizes)

    if not all_sizes:
        print(f"[graficas_igho] Omitida '{out_path.name}': no hay tamanos (ni GT ni predichos).")
        return

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    bins = np.linspace(0, max(all_sizes) + 1, 20)

    # Nota: el ground truth son solo 8 valores (un tamano por video/lesion),
    # mientras que cada modelo aporta un tamano por CADA frame generado
    # (decenas de miles). En conteos absolutos (`ax.hist` sin density) la
    # curva del ground truth queda aplastada cerca de 0 frente a las curvas
    # de prediccion y no se distingue visualmente. Por eso se grafica
    # `density=True` en todas las curvas: cada una pasa a representar la
    # proporcion relativa de sus propios tamanos (area = 1), lo que permite
    # comparar la FORMA de las distribuciones real vs. predicha pese a la
    # enorme diferencia de n entre el ground truth y las predicciones.
    if gt_sizes:
        ax.hist(
            gt_sizes,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.4,
            color="black",
            label=f"Ground truth (n={len(gt_sizes)})",
            zorder=5,
        )

    colors = plt.cm.tab10(np.linspace(0, 1, len(predicted_sizes_by_model) + 1))
    for idx, (model, sizes) in enumerate(predicted_sizes_by_model.items()):
        if not sizes:
            print(f"[graficas_igho] Aviso: '{model}' no aporto tamanos predichos (todas las captions sin 'mm').")
            continue
        ax.hist(
            sizes,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            color=colors[idx],
            label=f"{model} (n={len(sizes)})",
        )

    ax.set_xlabel("Tamano (mm)", fontsize=10)
    ax.set_ylabel("Densidad (area = 1 por curva)", fontsize=10)
    ax.set_title("igho", fontsize=12)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle("Distribucion de tamanos de lesion: real vs predicho", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficas_igho] Escrito: {out_path.relative_to(REPO_ROOT)} (GT n={len(gt_sizes)})")


# ---------------------------------------------------------------------------
# 2) confusion_matrix.png
# ---------------------------------------------------------------------------

def load_confusion_totals(metrics_csv: Path, models: Sequence[str]) -> Dict[str, np.ndarray]:
    """Lee tp/fp/fn/tn agregados (8 videos x 2 folds) por modelo.

    Fuente: `igho/metrics/promedio_todos_los_videos.csv`, generado por
    `igho/metrics/igho_metrics.py`. Las filas de interes tienen `model`
    con sufijo `_promedio` (p. ej. `biomedclip_promedio`). Los valores
    pueden ser no enteros (medias entre folds sumadas entre videos) y se
    redondean solo para poder dibujarlos en la matriz.
    """
    if not metrics_csv.exists():
        raise FileNotFoundError(f"No se encontro {metrics_csv}")

    rows_by_model: Dict[str, Dict[str, str]] = {}
    for row in load_csv_rows(metrics_csv):
        model_key = row.get("model", "")
        for model in models:
            if model_key == f"{model}_promedio":
                rows_by_model[model] = row

    matrices: Dict[str, np.ndarray] = {}
    for model in models:
        row = rows_by_model.get(model)
        if row is None:
            print(f"[graficas_igho] Aviso: no hay fila '{model}_promedio' en {metrics_csv.name}.")
            continue
        tp = safe_float(row.get("total_tp")) or 0.0
        fp = safe_float(row.get("total_fp")) or 0.0
        fn = safe_float(row.get("total_fn")) or 0.0
        tn = safe_float(row.get("total_tn")) or 0.0
        # Orden de LABELS_ORDER = (negative, positive):
        # fila 0 = negativo real -> [tn, fp]; fila 1 = positivo real -> [fn, tp]
        cm = np.array([[tn, fp], [fn, tp]])
        matrices[model] = np.rint(cm).astype(int)
    return matrices


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    def safe_div(num: float, den: float) -> float:
        return num / den if den else 0.0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "n": tp + tn + fp + fn,
    }


DEFAULT_CM_TITLE = "Matriz de confusion por modelo — igho (agregado de 8 videos x 2 folds)"


def plot_confusion_matrix(
    matrices: Dict[str, np.ndarray],
    out_path: Path,
    dpi: int,
    title: str = DEFAULT_CM_TITLE,
) -> None:
    if not matrices:
        print(f"[graficas_igho] Omitida '{out_path.name}': no hay matrices de confusion disponibles.")
        return

    models = list(matrices.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.6), squeeze=False)
    axes = axes[0]

    vmax = max(int(cm.max()) for cm in matrices.values())

    for ax, model in zip(axes, models):
        cm = matrices[model]
        metrics = metrics_from_confusion(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[l.capitalize() for l in LABELS_ORDER])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", im_kw={"vmin": 0, "vmax": vmax})
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Etiqueta real" if model == models[0] else "")
        ax.set_title(
            f"{model}\nN={metrics['n']:,}, Acc={metrics['accuracy']:.1%}, "
            f"Prec={metrics['precision']:.1%}, Rec={metrics['recall']:.1%}",
            fontsize=9,
        )

    fig.suptitle(
        title,
        fontsize=13,
        y=1.05,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficas_igho] Escrito: {out_path.relative_to(REPO_ROOT)}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_sizes = load_ground_truth_sizes(args.dataset_csv)
    print(f"[graficas_igho] Ground truth: {len(gt_sizes)} tamanos de lesion desde {args.dataset_csv.relative_to(REPO_ROOT)}")

    predicted_sizes_by_model = {}
    for model in args.models:
        sizes = load_predicted_sizes(args.videos_dir, model)
        predicted_sizes_by_model[model] = sizes
        print(f"[graficas_igho] {model}: {len(sizes)} tamanos predichos extraidos de las captions.")

    plot_lesion_size_distribution(
        gt_sizes,
        predicted_sizes_by_model,
        args.output_dir / "lesion_size_distribution.png",
        args.dpi,
    )

    matrices = load_confusion_totals(args.metrics_csv, args.models)
    plot_confusion_matrix(matrices, args.output_dir / "confusion_matrix.png", args.dpi)

    print(f"[graficas_igho] Listo. Salida en: {args.output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

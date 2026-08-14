#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

LABELS_ORDER = ("negative", "positive")
DEFAULT_MODELS = ("biomedclip", "resnet", "vit")

NEGATIVE_PATTERNS = (
    re.compile(r"\bno\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bno\s+visible\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bno\s+evidence\s+of\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+any\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bnormal\s+colonoscopy\b", flags=re.IGNORECASE),
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "specificity")


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = value.replace("<|endoftext|>", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def is_negative_caption(text: Any) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in NEGATIVE_PATTERNS)


def infer_binary_label_from_caption(text: Any) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    return "negative" if is_negative_caption(normalized) else "positive"


def normalize_label(raw_label: Any) -> Optional[str]:
    text = normalize_text(raw_label)
    if not text:
        return None
    if text in {"positive", "pos", "1", "polyp", "with_polyp", "with-polyp"}:
        return "positive"
    if text in {"negative", "neg", "0", "no_polyp", "no-polyp"}:
        return "negative"
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return None


def extract_size(text: Any) -> Optional[float]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*mm", normalize_text(text))
    return float(match.group(1)) if match else None


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def get_first(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def extract_case_from_path(path: Any) -> str:
    filename = str(path or "").replace("\\", "/").split("/")[-1]
    if "_img_" in filename:
        return filename.split("_img_")[0]
    stem = Path(filename).stem
    return stem or "unknown"


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def fold_from_path(path: Path) -> str:
    for part in path.parts:
        if re.match(r"^fold_\d+$", part):
            return part
    return "_all"


def collect_prediction_files(model_dir: Path) -> List[Path]:
    direct_files = sorted((model_dir / "folds").glob("fold_*/inference/val_predictions.csv"))
    if direct_files:
        return direct_files

    fold_pred_files = sorted(model_dir.glob("fold_*/predictions.csv"))
    if fold_pred_files:
        return fold_pred_files

    aggregated = model_dir / "data" / "captions_generated.csv"
    if aggregated.exists():
        return [aggregated]

    legacy_files = sorted(model_dir.glob("fold/*/inference/val_predictions.csv"))
    if legacy_files:
        return legacy_files

    flat_files = sorted(model_dir.glob("*predictions*.csv"))
    return flat_files


def canonicalize_row(raw: Dict[str, str], dataset: str, model: str, source_file: Path) -> Dict[str, Any]:
    fold_value = get_first(raw, ["fold"], default="") or fold_from_path(source_file)
    image_path = get_first(raw, ["image_path"])
    case_value = get_first(raw, ["case"]) or extract_case_from_path(image_path)

    return {
        "dataset": dataset,
        "model": model,
        "fold": fold_value,
        "sample_id": get_first(raw, ["sample_id"]),
        "case": case_value,
        "image_path": image_path,
        "label": get_first(raw, ["label"]),
        "caption_gt": get_first(raw, ["caption_gt", "gt_caption", "caption"]),
        "generated_caption": get_first(raw, ["generated_caption", "pred_caption"]),
    }


def load_dataset_rows(dataset: str, dataset_root: Path, models: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model in models:
        model_dir = dataset_root / model
        if not model_dir.exists():
            continue
        for source_file in collect_prediction_files(model_dir):
            for raw_row in read_csv_rows(source_file):
                rows.append(canonicalize_row(raw_row, dataset, model, source_file))
    return rows


def true_label(row: Dict[str, Any]) -> Optional[str]:
    label = infer_binary_label_from_caption(row.get("caption_gt"))
    if label is None:
        label = normalize_label(row.get("label"))
    return label


def pred_label(row: Dict[str, Any]) -> Optional[str]:
    return infer_binary_label_from_caption(row.get("generated_caption"))


def compute_confusion(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, int]:
    y_true: List[str] = []
    y_pred: List[str] = []
    for row in rows:
        yt = true_label(row)
        yp = pred_label(row)
        if yt not in LABELS_ORDER or yp not in LABELS_ORDER:
            continue
        y_true.append(yt)
        y_pred.append(yp)

    if y_true:
        cm = confusion_matrix(y_true, y_pred, labels=list(LABELS_ORDER))
    else:
        cm = np.zeros((2, 2), dtype=np.int64)
    return np.asarray(cm, dtype=np.int64), len(y_true)


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": safe_div(tn, tn + fp),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def group_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row.get(k, "")) for k in keys)
        groups.setdefault(key, []).append(row)
    return groups


def plot_confusion_grid(
    all_rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str],
    models: Sequence[str],
    out_path: Path,
    dpi: int,
) -> None:
    grouped = group_rows(all_rows, ["dataset", "model"])
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(4.6 * len(models), 4.2 * len(datasets)), squeeze=False)

    vmax = 1
    for key, rows in grouped.items():
        cm, _ = compute_confusion(rows)
        vmax = max(vmax, int(cm.max()))

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            rows = grouped.get((dataset, model), [])
            cm, used = compute_confusion(rows)
            metrics = metrics_from_confusion(cm)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[l.capitalize() for l in LABELS_ORDER])
            disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", im_kw={"vmin": 0, "vmax": vmax})
            ax.set_xlabel("Prediccion")
            ax.set_ylabel("Etiqueta real" if j == 0 else "")
            ax.set_title(
                f"{dataset} / {model}\nN={used}, Acc={metrics['accuracy']:.1%}, "
                f"Prec={metrics['precision']:.1%}, Rec={metrics['recall']:.1%}",
                fontsize=9,
            )

    fig.suptitle("Matrices de confusion por dataset y modelo", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def compute_dataset_stats(all_rows: Sequence[Dict[str, Any]], datasets: Sequence[str], models: Sequence[str]) -> List[Dict[str, Any]]:
    stats_rows: List[Dict[str, Any]] = []
    for dataset in datasets:
        reference_model = None
        for model in models:
            if any(r["dataset"] == dataset and r["model"] == model for r in all_rows):
                reference_model = model
                break
        if reference_model is None:
            continue

        ref_rows = [r for r in all_rows if r["dataset"] == dataset and r["model"] == reference_model]
        by_fold = group_rows(ref_rows, ["fold"])

        for fold_key, rows in sorted(by_fold.items()):
            fold_name = fold_key[0]
            labels = [true_label(r) for r in rows]
            n_pos = sum(1 for l in labels if l == "positive")
            n_neg = sum(1 for l in labels if l == "negative")
            n_cases = len({r["case"] for r in rows if r.get("case")})
            stats_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold_name,
                    "reference_model": reference_model,
                    "n_frames_total": len(rows),
                    "n_frames_positive": n_pos,
                    "n_frames_negative": n_neg,
                    "n_cases": n_cases,
                }
            )

        labels_all = [true_label(r) for r in ref_rows]
        stats_rows.append(
            {
                "dataset": dataset,
                "fold": "_all",
                "reference_model": reference_model,
                "n_frames_total": len(ref_rows),
                "n_frames_positive": sum(1 for l in labels_all if l == "positive"),
                "n_frames_negative": sum(1 for l in labels_all if l == "negative"),
                "n_cases": len({r["case"] for r in ref_rows if r.get("case")}),
            }
        )
    return stats_rows


def plot_dataset_stats(stats_rows: Sequence[Dict[str, Any]], datasets: Sequence[str], out_path: Path, dpi: int) -> None:
    totals = [r for r in stats_rows if r["fold"] == "_all"]
    totals_by_dataset = {r["dataset"]: r for r in totals}

    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 5))
    x = np.arange(len(datasets))
    width = 0.5

    pos_counts = [totals_by_dataset.get(d, {}).get("n_frames_positive", 0) for d in datasets]
    neg_counts = [totals_by_dataset.get(d, {}).get("n_frames_negative", 0) for d in datasets]

    ax.bar(x, neg_counts, width, label="Negative", color="#2a9d8f")
    ax.bar(x, pos_counts, width, bottom=neg_counts, label="Positive", color="#e76f51")

    for i, d in enumerate(datasets):
        total = pos_counts[i] + neg_counts[i]
        if total > 0:
            ratio = pos_counts[i] / total
            ax.text(x[i], total + max(totals_by_dataset.get(d, {}).get("n_frames_total", 1), 1) * 0.02,
                     f"{ratio:.1%} pos", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Numero de frames")
    ax.set_title("Distribucion real de clases por dataset")
    ax.legend()
    ax.grid(alpha=0.25, axis="y", linestyle="--", linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_generalization_gap(
    all_rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str],
    models: Sequence[str],
    out_path: Path,
    dpi: int,
) -> None:
    grouped = group_rows(all_rows, ["dataset", "model"])
    metric_values = {"accuracy": {}, "f1": {}}

    for dataset in datasets:
        for model in models:
            rows = grouped.get((dataset, model), [])
            cm, _ = compute_confusion(rows)
            metrics = metrics_from_confusion(cm)
            metric_values["accuracy"].setdefault(dataset, {})[model] = metrics["accuracy"]
            metric_values["f1"].setdefault(dataset, {})[model] = metrics["f1"]

    fig, axes = plt.subplots(1, 2, figsize=(6.5 * 2, 5))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(models), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, metric_name in zip(axes, ("accuracy", "f1")):
        for i, model in enumerate(models):
            values = [metric_values[metric_name].get(d, {}).get(model, 0.0) for d in datasets]
            ax.bar(x + i * width - width * (len(models) - 1) / 2, values, width, label=model, color=colors[i])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"{metric_name.capitalize()} por dataset y modelo")
        ax.grid(alpha=0.25, axis="y", linestyle="--", linewidth=0.6)
        ax.legend(fontsize=8)

    fig.suptitle("Brecha de generalizacion entre datasets", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_fold_variability(
    all_rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str],
    models: Sequence[str],
    out_path: Path,
    dpi: int,
) -> None:
    grouped = group_rows(all_rows, ["dataset", "model"])
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(4.2 * len(models), 4.0 * len(datasets)), squeeze=False)

    for i, dataset in enumerate(datasets):
        for j, model in enumerate(models):
            ax = axes[i][j]
            rows = grouped.get((dataset, model), [])
            by_fold = group_rows(rows, ["fold"])

            per_metric_values = {m: [] for m in METRIC_NAMES}
            for fold_key, fold_rows in by_fold.items():
                cm, _ = compute_confusion(fold_rows)
                metrics = metrics_from_confusion(cm)
                for m in METRIC_NAMES:
                    per_metric_values[m].append(metrics[m])

            data = [per_metric_values[m] for m in METRIC_NAMES]
            bp = ax.boxplot(data, labels=[m[:4] for m in METRIC_NAMES], showmeans=True, widths=0.5)
            for m_idx, values in enumerate(data):
                xs = np.random.normal(m_idx + 1, 0.04, size=len(values))
                ax.scatter(xs, values, color="#264653", s=18, zorder=3)

            ax.set_ylim(0, 1)
            ax.set_title(f"{dataset} / {model}", fontsize=10)
            ax.grid(alpha=0.25, axis="y", linestyle="--", linewidth=0.6)

    fig.suptitle("Variabilidad de metricas entre folds", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_size_distributions(
    all_rows: Sequence[Dict[str, Any]],
    datasets: Sequence[str],
    models: Sequence[str],
    out_path: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 4.5), squeeze=False)
    axes = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(models) + 1))

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        dataset_rows = [r for r in all_rows if r["dataset"] == dataset]

        reference_model = None
        for model in models:
            if any(r["model"] == model for r in dataset_rows):
                reference_model = model
                break

        gt_sizes: List[float] = []
        seen_images = set()
        if reference_model is not None:
            for row in dataset_rows:
                if row["model"] != reference_model:
                    continue
                key = row.get("image_path") or row.get("sample_id")
                if key in seen_images:
                    continue
                seen_images.add(key)
                size = extract_size(row.get("caption_gt"))
                if size is not None:
                    gt_sizes.append(size)

        all_sizes = list(gt_sizes)
        for model in models:
            pred_sizes = [
                extract_size(row.get("generated_caption"))
                for row in dataset_rows
                if row["model"] == model
            ]
            all_sizes.extend([s for s in pred_sizes if s is not None])

        if not all_sizes:
            ax.text(0.5, 0.5, "Sin datos de tamano", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset)
            continue

        bins = np.linspace(0, max(all_sizes) + 1, 20)

        if gt_sizes:
            ax.hist(gt_sizes, bins=bins, histtype="step", linewidth=2.2, color="black", label="Ground truth")

        for m_idx, model in enumerate(models):
            pred_sizes = [
                extract_size(row.get("generated_caption"))
                for row in dataset_rows
                if row["model"] == model
            ]
            pred_sizes = [s for s in pred_sizes if s is not None]
            if pred_sizes:
                ax.hist(pred_sizes, bins=bins, histtype="step", linewidth=1.6, color=colors[m_idx], label=model)

        ax.set_xlabel("Tamano (mm)")
        ax.set_ylabel("Frecuencia")
        ax.set_title(dataset)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle("Distribucion de tamanos de lesion: real vs predicho", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_dataset_arg(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"Formato invalido, use nombre=ruta: {value}")
    name, path_str = value.split("=", 1)
    return name.strip(), Path(path_str).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graficas comparativas multi-dataset para ClipCap.")
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help="nombre=ruta, repetible. Ej: --dataset sun=fold/2fold --dataset igho=igho/eval --dataset real_colon=real_colon/eval",
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--output_root", type=Path, default=Path("plots_multidataset"))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_names = [name for name, _ in args.datasets]
    all_rows: List[Dict[str, Any]] = []

    for name, root in args.datasets:
        if not root.exists():
            print(f"[WARN] No existe la ruta del dataset '{name}': {root}")
            continue
        rows = load_dataset_rows(name, root, args.models)
        print(f"[INFO] {name}: {len(rows)} filas cargadas desde {root}")
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No se cargaron filas de ningun dataset/modelo.")

    plot_confusion_grid(all_rows, dataset_names, args.models, output_root / "confusion_matrices_by_dataset.png", args.dpi)

    stats_rows = compute_dataset_stats(all_rows, dataset_names, args.models)
    write_csv(
        output_root / "dataset_stats.csv",
        stats_rows,
        ["dataset", "fold", "reference_model", "n_frames_total", "n_frames_positive", "n_frames_negative", "n_cases"],
    )
    plot_dataset_stats(stats_rows, dataset_names, output_root / "dataset_class_balance.png", args.dpi)

    plot_generalization_gap(all_rows, dataset_names, args.models, output_root / "generalization_gap.png", args.dpi)

    plot_fold_variability(all_rows, dataset_names, args.models, output_root / "fold_variability.png", args.dpi)

    plot_size_distributions(all_rows, dataset_names, args.models, output_root / "lesion_size_distribution.png", args.dpi)

    summary = {
        "output_root": str(output_root),
        "datasets": dataset_names,
        "models": args.models,
        "n_rows_total": len(all_rows),
        "outputs": {
            "confusion_matrices": "confusion_matrices_by_dataset.png",
            "dataset_stats_csv": "dataset_stats.csv",
            "dataset_class_balance": "dataset_class_balance.png",
            "generalization_gap": "generalization_gap.png",
            "fold_variability": "fold_variability.png",
            "lesion_size_distribution": "lesion_size_distribution.png",
        },
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"[DONE] Resultados en: {output_root}")


if __name__ == "__main__":
    main()
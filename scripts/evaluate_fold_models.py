from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FOLD_ROOT = REPO_ROOT / "fold" / "5kfold"

TABLE_I_FILENAME = "table_i_frame_level_metrics.csv"
TABLE_II_FILENAME = "table_ii_clinical_report_generation_metrics.csv"

CANONICAL_NEGATIVE_CAPTION = "this is a colonoscopy frame from a patient with no polyps."

MORPH = [
    "flat elevated mucosal",
    "subpedunculated",
    "pedunculated",
    "sessile",
    "flat elevated",
    "mucosal",
    "elevated",
    "flat",
]

LESIONS = ["adenoma", "hyperplastic", "polyp"]

LOCATIONS = [
    "sigmoid colon",
    "descending colon",
    "transverse colon",
    "ascending colon",
    "rectum",
    "cecum",
    "colon",
]

STOP_PHRASES = [
    "this is a colonoscopy frame from a",
    "this is a colonoscopy frame from",
    "a patient with a",
    "patient with a",
    "patient with",
    "a patient",
]

NEGATIVE_PATTERNS = [
    re.compile(r"\bno\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bno\s+visible\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bno\s+evidence\s+of\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bwithout\s+any\s+polyps?\b", flags=re.IGNORECASE),
    re.compile(r"\bnormal\s+colonoscopy\b", flags=re.IGNORECASE),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def normalize_text(text: Any) -> str:
    value = str(text or "")
    value = value.replace("<|endoftext|>", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip().lower()


def clean_caption(text: Any) -> str:
    value = normalize_text(text)
    for phrase in STOP_PHRASES:
        value = value.replace(phrase, "")
    return " ".join(value.split())


def tokenize(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def safe_int(value: Any, default: int = -1) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except Exception:
        return default


def get_first(row: Dict[str, Any], keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


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


def is_negative_caption(text: Any) -> bool:
    normalized = normalize_text(text)
    if not normalized:
        return False
    if normalized == CANONICAL_NEGATIVE_CAPTION:
        return True
    return any(pattern.search(normalized) for pattern in NEGATIVE_PATTERNS)


def infer_binary_label_from_caption(text: Any) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    if is_negative_caption(normalized):
        return "negative"
    return "positive"


def extract_attr(text: Any, vocab: Sequence[str]) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    for value in vocab:
        if value in normalized:
            return value
    return None


def extract_lesion(text: Any) -> Optional[str]:
    normalized = normalize_text(text)
    if not normalized:
        return None
    if is_negative_caption(normalized):
        return None
    for value in LESIONS:
        pattern = rf"\b{re.escape(value)}s?\b"
        if re.search(pattern, normalized):
            return value
    return None


def extract_size(text: Any) -> Optional[int]:
    match = re.search(r"(\d+)\s*mm", normalize_text(text))
    return int(match.group(1)) if match else None


def extract_case_from_path(path: Any) -> str:
    filename = str(path or "").replace("\\", "/").split("/")[-1]
    if "_img_" in filename:
        return filename.split("_img_")[0]
    stem = Path(filename).stem
    return stem or "unknown"


def ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def corpus_modified_precision(rows: Sequence[Dict[str, Any]], n: int) -> float:
    total = 0
    clipped = 0
    for row in rows:
        ref_tokens = tokenize(row["gt_caption_clean"])
        pred_tokens = tokenize(row["pred_caption_clean"])
        pred_counts = Counter(ngrams(pred_tokens, n))
        ref_counts = Counter(ngrams(ref_tokens, n))
        total += sum(pred_counts.values())
        for gram, count in pred_counts.items():
            clipped += min(count, ref_counts.get(gram, 0))
    return safe_div(clipped, total)


def evaluate_clinical(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []
    for row in rows:
        gt_caption = row["gt_caption"]
        pred_caption = row["pred_caption"]

        gt_type = extract_lesion(gt_caption)
        pred_type = extract_lesion(pred_caption)
        gt_location = extract_attr(gt_caption, LOCATIONS)
        pred_location = extract_attr(pred_caption, LOCATIONS)
        gt_morph = extract_attr(gt_caption, MORPH)
        pred_morph = extract_attr(pred_caption, MORPH)
        gt_size = extract_size(gt_caption)
        pred_size = extract_size(pred_caption)

        evaluated.append(
            {
                "gt_type": gt_type,
                "pred_type": pred_type,
                "type_correct": int(gt_type == pred_type) if gt_type is not None else None,
                "gt_location": gt_location,
                "pred_location": pred_location,
                "location_correct": int(gt_location == pred_location) if gt_location is not None else None,
                "gt_morph": gt_morph,
                "pred_morph": pred_morph,
                "morph_correct": int(gt_morph == pred_morph) if gt_morph is not None else None,
                "gt_size": gt_size,
                "pred_size": pred_size,
                "size_error": abs(gt_size - pred_size) if (gt_size is not None and pred_size is not None) else None,
            }
        )
    return evaluated


def mean_over_present(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[float, int]:
    values = [row[key] for row in rows if row[key] is not None]
    return safe_div(sum(values), len(values)), len(values)


def mean_size_error(rows: Sequence[Dict[str, Any]]) -> Tuple[float, int]:
    values = [row["size_error"] for row in rows if row["size_error"] is not None]
    return safe_div(sum(values), len(values)), len(values)


def population_std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def split_rows_by_fold(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    fold_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        fold = normalize_text(row.get("fold"))
        if not fold:
            fold = "_all"
        fold_groups.setdefault(fold, []).append(row)
    return [fold_groups[name] for name in sorted(fold_groups)]


def std_across_fold_metrics(
    fold_metrics: Sequence[Dict[str, Any]], metric_names: Sequence[str]
) -> Dict[str, float]:
    return {
        metric_name: population_std([float(metrics.get(metric_name, 0.0)) for metrics in fold_metrics])
        for metric_name in metric_names
    }


def canonicalize_prediction_row(row: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    fold_value = get_first(row, ["fold"], default="")
    if not fold_value:
        fold_value = source_file.parent.parent.name if source_file.parent.parent.name.startswith("fold_") else ""

    image_path = get_first(row, ["image_path"])
    case_value = get_first(row, ["case"])
    if not case_value:
        case_value = extract_case_from_path(image_path)

    return {
        "fold": fold_value,
        "sample_id": get_first(row, ["sample_id"]),
        "label": get_first(row, ["label"]),
        "case": case_value,
        "image_path": image_path,
        "caption_gt": get_first(row, ["caption_gt", "gt_caption", "caption"]),
        "generated_caption": get_first(row, ["generated_caption", "pred_caption"]),
        "linked_image_path": get_first(row, ["linked_image_path"]),
        "_source_file": source_file.as_posix(),
    }


def fold_sort_key(path: Path) -> Tuple[int, str]:
    fold_name = path.parent.parent.name if path.parent.parent.name.startswith("fold_") else path.parent.name
    match = re.search(r"(\d+)", fold_name)
    return (int(match.group(1)) if match else 10**9, path.as_posix())


def collect_prediction_files(model_dir: Path) -> List[Path]:
    direct_files = sorted((model_dir / "folds").glob("fold_*/inference/val_predictions.csv"), key=fold_sort_key)
    if direct_files:
        return direct_files

    aggregated = model_dir / "data" / "captions_generated.csv"
    if aggregated.exists():
        return [aggregated]

    legacy_files = sorted(model_dir.glob("fold/*/inference/val_predictions.csv"))
    if legacy_files:
        return legacy_files

    return []


def load_prediction_rows(model_dir: Path) -> Tuple[List[Dict[str, Any]], List[Path]]:
    source_files = collect_prediction_files(model_dir)
    if not source_files:
        return [], []

    rows: List[Dict[str, Any]] = []
    for source_file in source_files:
        for raw_row in read_csv(source_file):
            rows.append(canonicalize_prediction_row(raw_row, source_file))

    rows.sort(
        key=lambda row: (
            row.get("fold", ""),
            safe_int(row.get("sample_id"), 10**9),
            row.get("case", ""),
            row.get("image_path", ""),
        )
    )
    return rows, source_files


def resolve_model_name(model_dir: Path) -> str:
    for candidate in ("summary.json", "run_config.json"):
        path = model_dir / candidate
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for key in ("model", "requested_model", "output_subdir"):
            value = payload.get(key)
            if value:
                return str(value)
    return model_dir.name


def discover_model_dirs(fold_root: Path) -> List[Path]:
    candidates = [
        path
        for path in fold_root.iterdir()
        if path.is_dir() and path.name != "plots" and collect_prediction_files(path)
    ]
    return sorted(candidates, key=lambda path: resolve_model_name(path))


def binary_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = tn = fp = fn = 0
    used = 0

    for row in rows:
        y_true = infer_binary_label_from_caption(row.get("caption_gt"))
        if y_true is None:
            y_true = normalize_label(row.get("label"))
        y_pred = infer_binary_label_from_caption(row.get("generated_caption"))

        if y_true not in {"positive", "negative"} or y_pred not in {"positive", "negative"}:
            continue

        used += 1
        true_positive = y_true == "positive"
        pred_positive = y_pred == "positive"

        if true_positive and pred_positive:
            tp += 1
        elif (not true_positive) and (not pred_positive):
            tn += 1
        elif (not true_positive) and pred_positive:
            fp += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "num_rows_used": used,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": safe_div(tn, tn + fp),
    }


def report_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prepared_rows = [
        {
            **row,
            "gt_caption": row["caption_gt"],
            "pred_caption": row["generated_caption"],
            "gt_caption_clean": clean_caption(row["caption_gt"]),
            "pred_caption_clean": clean_caption(row["generated_caption"]),
        }
        for row in rows
    ]

    clinical_rows = evaluate_clinical(prepared_rows)
    malignancy_accuracy, malignancy_support = mean_over_present(clinical_rows, "type_correct")
    location_accuracy, location_support = mean_over_present(clinical_rows, "location_correct")
    paris_accuracy, paris_support = mean_over_present(clinical_rows, "morph_correct")
    size_mae_mm, size_support = mean_size_error(clinical_rows)

    return {
        "num_rows_used": len(prepared_rows),
        "bleu_1": corpus_modified_precision(prepared_rows, 1),
        "bleu_4": corpus_modified_precision(prepared_rows, 4),
        "malignancy_accuracy": malignancy_accuracy,
        "location_accuracy": location_accuracy,
        "paris_accuracy": paris_accuracy,
        "size_mae_mm": size_mae_mm,
        "supports": {
            "malignancy": malignancy_support,
            "location": location_support,
            "paris": paris_support,
            "size_mae_mm": size_support,
        },
    }


def evaluate_model(model_dir: Path) -> Optional[Dict[str, Any]]:
    rows, source_files = load_prediction_rows(model_dir)
    if not rows:
        return None

    model_name = resolve_model_name(model_dir)
    folds = sorted({row.get("fold", "") for row in rows if row.get("fold", "")})
    fold_rows = split_rows_by_fold(rows)

    binary = binary_metrics(rows)
    report = report_metrics(rows)
    binary_std = std_across_fold_metrics(
        [binary_metrics(items) for items in fold_rows],
        ["accuracy", "precision", "recall", "f1", "specificity"],
    )
    report_std = std_across_fold_metrics(
        [report_metrics(items) for items in fold_rows],
        [
            "bleu_1",
            "bleu_4",
            "malignancy_accuracy",
            "location_accuracy",
            "paris_accuracy",
            "size_mae_mm",
        ],
    )

    return {
        "model": model_name,
        "model_dir": model_dir.name,
        "source_files": [path.as_posix() for path in source_files],
        "num_rows": len(rows),
        "num_folds": len(folds) if folds else len(source_files),
        "binary": binary,
        "report": report,
        "binary_std": binary_std,
        "report_std": report_std,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Table I and Table II metrics from fold/5kfold prediction CSVs."
    )
    parser.add_argument(
        "--fold_root",
        default=str(DEFAULT_FOLD_ROOT),
        help=f"Root with model results (default: {DEFAULT_FOLD_ROOT})",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model folder names or canonical model names to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_root = Path(args.fold_root).resolve()
    if not fold_root.exists():
        raise FileNotFoundError(f"fold_root not found: {fold_root}")

    model_dirs = discover_model_dirs(fold_root)
    if args.models:
        requested = {normalize_text(model) for model in args.models}
        filtered: List[Path] = []
        for model_dir in model_dirs:
            model_name = resolve_model_name(model_dir)
            if normalize_text(model_dir.name) in requested or normalize_text(model_name) in requested:
                filtered.append(model_dir)
        model_dirs = filtered

    if not model_dirs:
        raise FileNotFoundError(f"No model result directories found under {fold_root}")

    table_i_rows: List[Dict[str, Any]] = []
    table_ii_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"fold_root": fold_root.as_posix(), "models": []}

    for model_dir in model_dirs:
        result = evaluate_model(model_dir)
        if result is None:
            continue

        summary["models"].append(result)

        binary = result["binary"]
        report = result["report"]
        binary_std = result["binary_std"]
        report_std = result["report_std"]

        table_i_rows.append(
            {
                "model": result["model"],
                "accuracy": binary["accuracy"],
                "accuracy_std": binary_std["accuracy"],
                "precision": binary["precision"],
                "precision_std": binary_std["precision"],
                "recall": binary["recall"],
                "recall_std": binary_std["recall"],
                "f1": binary["f1"],
                "f1_std": binary_std["f1"],
                "specificity": binary["specificity"],
                "specificity_std": binary_std["specificity"],
            }
        )
        table_ii_rows.append(
            {
                "model": result["model"],
                "bleu_1": report["bleu_1"],
                "bleu_1_std": report_std["bleu_1"],
                "bleu_4": report["bleu_4"],
                "bleu_4_std": report_std["bleu_4"],
                "malignancy_accuracy": report["malignancy_accuracy"],
                "malignancy_accuracy_std": report_std["malignancy_accuracy"],
                "location_accuracy": report["location_accuracy"],
                "location_accuracy_std": report_std["location_accuracy"],
                "paris_accuracy": report["paris_accuracy"],
                "paris_accuracy_std": report_std["paris_accuracy"],
                "size_mae_mm": report["size_mae_mm"],
                "size_mae_mm_std": report_std["size_mae_mm"],
            }
        )

        print(
            f"[{result['model']}] rows={result['num_rows']} folds={result['num_folds']} "
            f"accuracy={binary['accuracy']:.4f} bleu_1={report['bleu_1']:.4f}"
        )

    table_i_path = fold_root / TABLE_I_FILENAME
    table_ii_path = fold_root / TABLE_II_FILENAME
    write_csv(
        table_i_path,
        table_i_rows,
        [
            "model",
            "accuracy",
            "accuracy_std",
            "precision",
            "precision_std",
            "recall",
            "recall_std",
            "f1",
            "f1_std",
            "specificity",
            "specificity_std",
        ],
    )
    write_csv(
        table_ii_path,
        table_ii_rows,
        [
            "model",
            "bleu_1",
            "bleu_1_std",
            "bleu_4",
            "bleu_4_std",
            "malignancy_accuracy",
            "malignancy_accuracy_std",
            "location_accuracy",
            "location_accuracy_std",
            "paris_accuracy",
            "paris_accuracy_std",
            "size_mae_mm",
            "size_mae_mm_std",
        ],
    )

    write_json(fold_root / "evaluate_fold_models_summary.json", summary)

    print(f"Table I written to: {table_i_path.as_posix()}")
    print(f"Table II written to: {table_ii_path.as_posix()}")


if __name__ == "__main__":
    main()

#inferience_metrics.py
"""Compare lesion annotations vs. inference reports frame-by-frame.

Modified to iterate over:
  dataset_real_colon/real_colon_inference/video_NNN-NNN/{biomedclip,resnet,vit}/fold_{0,1}/predictions.csv

It averages the 2 folds per video/model, then averages across all videos
to produce a single summary CSV per model and a global summary CSV.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

INFERENCE_ROOT = BASE_DIR / "dataset_real_colon" / "real_colon_inference"
ANNOTATIONS_DIR = BASE_DIR / "dataset_real_colon" / "Anotaciones"
GT_CSV_PATH = BASE_DIR / "dataset_real_colon" / "Datos_real_colon-lesion_info.csv"
OUTPUT_DIR = BASE_DIR / "dataset_real_colon" / "real_colon_inference"

MODELS = ["biomedclip", "resnet", "vit"]
FOLDS = ["fold_1", "fold_2"]


POSITIVE_MARKS = {"This is a colonoscopy frame from a patient with a"}
NEGATIVE_MARKS = {"This is a colonoscopy frame from a patient with no"}

# ── Metrics keys ───────────────────────────────────────────────────────────────
DETECTION_METRIC_KEYS = [
    "evaluated_frames",
    "gt_frames_with_lesion",
    "gt_frames_without_lesion",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "specificity",
    "f1",
    "accuracy",
    "tp_size_match_rate",
    "tp_location_match_rate",
    "tp_histology_match_rate",
    "tp_all_fields_match_rate",
]

CLINICAL_METRIC_KEYS = [
    "GT Positive Frames",
    "Malignacy Acc.",
    "Loc. Acc.",
    "Size (mm)",
]

ALL_METRIC_KEYS = DETECTION_METRIC_KEYS + CLINICAL_METRIC_KEYS


# ── Helper utilities ───────────────────────────────────────────────────────────
def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def parse_int(text: str) -> Optional[int]:
    cleaned = normalize_spaces(text).replace(",", ".")
    if not cleaned:
        return None
    match = re.search(r"-?\d+", cleaned)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def parse_float(text: str) -> Optional[float]:
    cleaned = normalize_spaces(text).replace(",", ".")
    if not cleaned:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def is_positive_mark(mark: str) -> bool:
    lowered = normalize_spaces(mark).lower()
    if lowered in POSITIVE_MARKS:
        return True
    if lowered in NEGATIVE_MARKS:
        return False
    return lowered.startswith("s") and "no" not in lowered


def parse_frame_number(frame_id: str) -> Optional[int]:
    match = re.match(r"^\d{3}-\d{3}_(\d+)$", normalize_spaces(frame_id))
    if not match:
        return None
    return int(match.group(1))


def parse_video_id(frame_id: str) -> str:
    match = re.match(r"^(\d{3}-\d{3})_\d+$", normalize_spaces(frame_id))
    if match:
        return match.group(1)
    return ""


def frame_sort_key(frame_id: str) -> Tuple[str, int]:
    cleaned = normalize_spaces(frame_id)
    match = re.match(r"^(\d{3}-\d{3})_(\d+)$", cleaned)
    if not match:
        return cleaned, -1
    return match.group(1), int(match.group(2))


def normalize_site(site: str) -> str:
    lowered = normalize_spaces(site).lower().replace("_", " ")
    lowered = re.sub(r"[^a-z\s-]", "", lowered)
    lowered = normalize_spaces(lowered)
    if not lowered:
        return ""
    if "sigmoid" in lowered or lowered == "sigma":
        return "sigmoid colon"
    if "caecum" in lowered or "cecum" in lowered:
        return "cecum"
    if "ascending" in lowered:
        return "ascending"
    if "descending" in lowered:
        return "descending"
    if "transverse" in lowered or "trasnverse" in lowered:
        return "transverse"
    if "rectum" in lowered or "rectal" in lowered:
        return "rectum"
    if "hepatic flexure" in lowered:
        return "hepatic flexure"
    if "splenic flexure" in lowered:
        return "splenic flexure"
    return lowered


def normalize_histology(text: str) -> str:
    lowered = normalize_spaces(text).lower()
    if not lowered:
        return ""
    if "no polyp" in lowered or "not-polyp" in lowered or "not polyp" in lowered:
        return "no_polyp"
    if "adenoma" in lowered or lowered == "ad":
        return "adenoma"
    if "hyperplastic" in lowered or lowered == "hp":
        return "hp"
    if "ssl" in lowered or "sessile serrated lesion" in lowered:
        return "ssl"
    if lowered == "tsa" or "traditional serrated adenoma" in lowered:
        return "tsa"
    if "serrated" in lowered:
        return "serrated"
    if "polyp" in lowered:
        return "polyp"
    return lowered


# ── CSV I/O ────────────────────────────────────────────────────────────────────
def load_csv_rows(path: Path) -> List[Dict[str, str]]:
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


# ── Parsing ────────────────────────────────────────────────────────────────────
def parse_caption(caption: str) -> Dict[str, object]:
    text = normalize_spaces(caption)
    lower = text.lower()

    has_lesion = False
    if re.search(r"\bwith\s+no\b", lower) or "no polyps" in lower or "without" in lower:
        has_lesion = False
    elif re.search(r"\bwith\s+an?\b", lower) or re.search(r"\bwith\s+\w+\s+polyp\b", lower):
        has_lesion = True
    elif "polyp" in lower and "no polyp" not in lower:
        has_lesion = True

    size_mm: Optional[float] = None
    size_match = re.search(r"(\d+(?:\.\d+)?)\s*mm\b", lower)
    if size_match:
        size_mm = float(size_match.group(1))

    location_raw = ""
    location_match = re.search(r"located in the\s+([a-z][a-z\s_-]*?)(?:[.,;]|$)", lower)
    if location_match:
        location_raw = normalize_spaces(location_match.group(1))
    location_norm = normalize_site(location_raw)

    histology = normalize_histology(lower)
    if histology == "polyp":
        histology = ""

    return {
        "pred_has_lesion": has_lesion,
        "pred_size_mm": size_mm,
        "pred_location_raw": location_raw,
        "pred_location_norm": location_norm,
        "pred_histology_norm": histology,
    }


def parse_annotations(
    rows: Sequence[Dict[str, str]],
    video_id: str,
) -> Dict[str, Dict[str, object]]:
    frames: Dict[str, Dict[str, object]] = {}

    for row in rows:
        frame_num = parse_int(row.get("numero_frame", ""))
        if frame_num is None:
            continue

        frame_id = f"{video_id}_{frame_num}"
        marca = normalize_spaces(row.get("marca_lesion", "")).lower()
        unique_id = normalize_spaces(row.get("unique_id", ""))

        item = frames.setdefault(
            frame_id,
            {
                "frame_number": frame_num,
                "marca_values": set(),
                "has_lesion": False,
                "unique_ids": set(),
                "max_lesion_count": 0,
            },
        )

        item["frame_number"] = frame_num
        if marca:
            item["marca_values"].add(marca)
        if is_positive_mark(marca):
            item["has_lesion"] = True
            if unique_id:
                item["unique_ids"].add(unique_id)

        lesion_count = parse_int(row.get("lesion_count", ""))
        if lesion_count is not None:
            item["max_lesion_count"] = max(int(item["max_lesion_count"]), lesion_count)

    return frames


def parse_predictions(rows: Sequence[Dict[str, str]], video_id: str) -> Dict[str, Dict[str, object]]:
    """Parse predictions.csv with columns: image_path, generated_caption.

    Extract frame_id from image_path (e.g. .../002-003_0.jpg -> 002-003_0)
    and parse generated_caption the same way as reporte_medico.
    """
    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        image_path = normalize_spaces(row.get("image_path", ""))
        if not image_path:
            continue
        # Extract frame id from the filename: e.g. "002-003_0.jpg" -> "002-003_0"
        filename = Path(image_path).stem  # removes extension
        # The frame id should be like NNN-NNN_N
        frame = normalize_spaces(filename)
        caption = normalize_spaces(row.get("generated_caption", ""))
        parsed = parse_caption(caption)
        parsed["reporte_medico"] = caption
        result[frame] = parsed
    return result


def parse_inference(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for row in rows:
        frame = normalize_spaces(row.get("frame", ""))
        if not frame:
            continue
        caption = normalize_spaces(row.get("reporte_medico", ""))
        parsed = parse_caption(caption)
        parsed["reporte_medico"] = caption
        result[frame] = parsed
    return result


def parse_gt(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, object]]:
    gt_map: Dict[str, Dict[str, object]] = {}
    for row in rows:
        unique_id = normalize_spaces(row.get("unique_object_id", ""))
        if not unique_id:
            continue
        size_mm = parse_float(row.get("size [mm]", ""))
        site_raw = normalize_spaces(row.get("site", ""))
        histology_ext = normalize_spaces(row.get("histology_extended", ""))
        histology_class = normalize_spaces(row.get("histology_class", ""))
        hist_source = histology_ext if histology_ext else histology_class
        gt_map[unique_id] = {
            "size_mm": size_mm,
            "site_raw": site_raw,
            "site_norm": normalize_site(site_raw),
            "histology_extended": histology_ext,
            "histology_class": histology_class,
            "histology_norm": normalize_histology(hist_source),
        }
    return gt_map


def float_equal(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def stringify_sorted(values: Iterable[str]) -> str:
    cleaned = [normalize_spaces(v) for v in values if normalize_spaces(v)]
    return ";".join(sorted(set(cleaned)))


def split_semicolon_values(text: object) -> List[str]:
    cleaned = normalize_spaces(str(text))
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(";") if item.strip()]


def confusion_label(gt_has_lesion: bool, pred_has_lesion: bool) -> str:
    if gt_has_lesion and pred_has_lesion:
        return "TP"
    if (not gt_has_lesion) and pred_has_lesion:
        return "FP"
    if gt_has_lesion and (not pred_has_lesion):
        return "FN"
    return "TN"


# ── Core comparison logic ─────────────────────────────────────────────────────
def build_comparison_rows(
    annotations_by_frame: Dict[str, Dict[str, object]],
    inference_by_frame: Dict[str, Dict[str, object]],
    gt_by_unique_id: Dict[str, Dict[str, object]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    all_frames = sorted(
        set(annotations_by_frame.keys()) | set(inference_by_frame.keys()),
        key=frame_sort_key,
    )
    rows: List[Dict[str, object]] = []

    tp = fp = fn = tn = 0
    evaluated_rows = 0
    tp_size_match = tp_site_match = tp_hist_match = tp_all_match = 0
    gt_frames_with_lesion = 0
    gt_frames_without_lesion = 0

    for frame in all_frames:
        annotation = annotations_by_frame.get(frame)
        inference = inference_by_frame.get(frame)

        video_id = parse_video_id(frame)
        frame_number = parse_frame_number(frame)
        annotation_has_lesion = bool(annotation["has_lesion"]) if annotation else False
        annotation_marks = stringify_sorted(annotation["marca_values"]) if annotation else ""
        annotation_unique_ids: Set[str] = set(annotation["unique_ids"]) if annotation else set()
        lesion_count = int(annotation["max_lesion_count"]) if annotation else 0

        pred_has_lesion = bool(inference["pred_has_lesion"]) if inference else False
        pred_size_mm = inference.get("pred_size_mm") if inference else None
        pred_location_raw = str(inference.get("pred_location_raw", "")) if inference else ""
        pred_location_norm = str(inference.get("pred_location_norm", "")) if inference else ""
        pred_histology_norm = str(inference.get("pred_histology_norm", "")) if inference else ""
        reporte = str(inference.get("reporte_medico", "")) if inference else ""

        gt_entries = [gt_by_unique_id[uid] for uid in sorted(annotation_unique_ids) if uid in gt_by_unique_id]
        missing_gt_uids = sorted(uid for uid in annotation_unique_ids if uid not in gt_by_unique_id)

        gt_sizes = [entry["size_mm"] for entry in gt_entries if entry["size_mm"] is not None]
        gt_sites_raw = [str(entry["site_raw"]) for entry in gt_entries if entry["site_raw"]]
        gt_sites_norm = [str(entry["site_norm"]) for entry in gt_entries if entry["site_norm"]]
        gt_histology_norm = [str(entry["histology_norm"]) for entry in gt_entries if entry["histology_norm"]]

        size_match = pred_has_lesion and any(float_equal(pred_size_mm, gt_size) for gt_size in gt_sizes)
        location_match = pred_has_lesion and bool(pred_location_norm) and pred_location_norm in set(gt_sites_norm)
        histology_match = (
            pred_has_lesion and bool(pred_histology_norm) and pred_histology_norm in set(gt_histology_norm)
        )

        all_fields_match = False
        if pred_has_lesion and pred_size_mm is not None and pred_location_norm and pred_histology_norm:
            for entry in gt_entries:
                if (
                    float_equal(pred_size_mm, entry.get("size_mm"))
                    and pred_location_norm == entry.get("site_norm")
                    and pred_histology_norm == entry.get("histology_norm")
                ):
                    all_fields_match = True
                    break

        confusion = ""
        in_scope = annotation is not None
        if in_scope:
            if annotation_has_lesion:
                gt_frames_with_lesion += 1
            else:
                gt_frames_without_lesion += 1

            confusion = confusion_label(annotation_has_lesion, pred_has_lesion)
            evaluated_rows += 1
            if confusion == "TP":
                tp += 1
                tp_size_match += int(size_match)
                tp_site_match += int(location_match)
                tp_hist_match += int(histology_match)
                tp_all_match += int(all_fields_match)
            elif confusion == "FP":
                fp += 1
            elif confusion == "FN":
                fn += 1
            elif confusion == "TN":
                tn += 1
        else:
            confusion = "NO_ANNOTATION"

        error_reason_parts: List[str] = []
        if confusion == "FP":
            error_reason_parts.append("false_positive_presence")
        elif confusion == "FN":
            error_reason_parts.append("false_negative_presence")
        elif confusion == "TP":
            if pred_size_mm is None:
                error_reason_parts.append("missing_pred_size")
            elif not size_match:
                error_reason_parts.append("size_mismatch")
            if not pred_location_norm:
                error_reason_parts.append("missing_pred_location")
            elif not location_match:
                error_reason_parts.append("location_mismatch")
            if not pred_histology_norm:
                error_reason_parts.append("missing_pred_histology")
            elif not histology_match:
                error_reason_parts.append("histology_mismatch")
        elif confusion == "NO_ANNOTATION":
            error_reason_parts.append("frame_not_in_annotations")

        rows.append(
            {
                "video_id": video_id,
                "frame": frame,
                "frame_number": frame_number if frame_number is not None else "",
                "annotation_has_lesion": int(annotation_has_lesion) if in_scope else "",
                "annotation_marca_lesion": annotation_marks,
                "annotation_lesion_count": lesion_count if in_scope else "",
                "annotation_unique_ids": stringify_sorted(annotation_unique_ids),
                "gt_unique_ids_found": stringify_sorted([uid for uid in annotation_unique_ids if uid in gt_by_unique_id]),
                "gt_unique_ids_missing": stringify_sorted(missing_gt_uids),
                "gt_size_mm_values": ";".join(str(int(v)) if float(v).is_integer() else str(v) for v in sorted(set(gt_sizes))),
                "gt_site_values_raw": stringify_sorted(gt_sites_raw),
                "gt_site_values_norm": stringify_sorted(gt_sites_norm),
                "gt_histology_values_norm": stringify_sorted(gt_histology_norm),
                "pred_has_lesion": int(pred_has_lesion),
                "pred_size_mm": (
                    int(pred_size_mm) if isinstance(pred_size_mm, float) and pred_size_mm.is_integer() else pred_size_mm
                )
                if pred_size_mm is not None
                else "",
                "pred_location_raw": pred_location_raw,
                "pred_location_norm": pred_location_norm,
                "pred_histology_norm": pred_histology_norm,
                "confusion_label": confusion,
                "size_match": int(size_match),
                "location_match": int(location_match),
                "histology_match": int(histology_match),
                "all_fields_match": int(all_fields_match),
                "in_scope_for_metrics": int(in_scope),
                "error_reason": ";".join(error_reason_parts),
                "reporte_medico": reporte,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / evaluated_rows if evaluated_rows else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    summary = {
        "evaluated_frames": evaluated_rows,
        "gt_frames_with_lesion": gt_frames_with_lesion,
        "gt_frames_without_lesion": gt_frames_without_lesion,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "tp_size_match_rate": (tp_size_match / tp) if tp else 0.0,
        "tp_location_match_rate": (tp_site_match / tp) if tp else 0.0,
        "tp_histology_match_rate": (tp_hist_match / tp) if tp else 0.0,
        "tp_all_fields_match_rate": (tp_all_match / tp) if tp else 0.0,
    }
    return rows, summary


def build_clinical_metrics(
    comparison_rows: Sequence[Dict[str, object]],
    video_id: str,
) -> Dict[str, object]:
    gt_positive_rows = [
        row
        for row in comparison_rows
        if str(row.get("in_scope_for_metrics", "")) == "1"
        and str(row.get("annotation_has_lesion", "")) == "1"
    ]

    denominator = len(gt_positive_rows)
    malignancy_correct = 0
    location_correct = 0
    size_abs_errors: List[float] = []

    for row in gt_positive_rows:
        if str(row.get("histology_match", "")) == "1":
            malignancy_correct += 1
        if str(row.get("location_match", "")) == "1":
            location_correct += 1

        pred_size = parse_float(str(row.get("pred_size_mm", "")))
        gt_sizes = [parse_float(value) for value in split_semicolon_values(row.get("gt_size_mm_values", ""))]
        gt_sizes = [value for value in gt_sizes if value is not None]
        if pred_size is not None and gt_sizes:
            size_abs_errors.append(min(abs(pred_size - gt_size) for gt_size in gt_sizes))

    malignancy_acc = (malignancy_correct / denominator) if denominator else 0.0
    location_acc = (location_correct / denominator) if denominator else 0.0
    size_mae = (sum(size_abs_errors) / len(size_abs_errors)) if size_abs_errors else 0.0

    return {
        "GT Positive Frames": denominator,
        "Malignacy Acc.": round(malignancy_acc, 4),
        "Loc. Acc.": round(location_acc, 4),
        "Size (mm)": round(size_mae, 4),
    }


# ── Averaging utilities ───────────────────────────────────────────────────────
def average_metrics(metrics_list: List[Dict[str, object]]) -> Dict[str, object]:
    """Average a list of metric dictionaries.

    For each key, computes the arithmetic mean across all dicts that have
    a numeric value for that key.
    """
    if not metrics_list:
        return {}

    all_keys = []
    seen = set()
    for m in metrics_list:
        for k in m:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    averaged: Dict[str, object] = {}
    for key in all_keys:
        values = []
        for m in metrics_list:
            val = m.get(key)
            if val is not None and val != "":
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass
        if values:
            averaged[key] = round(sum(values) / len(values), 4)
        else:
            averaged[key] = ""
    return averaged


def std_metrics(metrics_list: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute standard deviation for each metric across a list of metric dicts.

    Returns a dict with the same keys but suffixed with '_std'.
    Uses population std (ddof=0) when n=1, sample std (ddof=1) when n>1.
    """
    if not metrics_list:
        return {}

    all_keys = []
    seen = set()
    for m in metrics_list:
        for k in m:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    std_dict: Dict[str, object] = {}
    for key in all_keys:
        values = []
        for m in metrics_list:
            val = m.get(key)
            if val is not None and val != "":
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    pass
        if len(values) >= 2:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std_dict[f"{key}_std"] = round(math.sqrt(variance), 4)
        elif len(values) == 1:
            std_dict[f"{key}_std"] = 0.0
        else:
            std_dict[f"{key}_std"] = ""
    return std_dict


def merge_detection_and_clinical(
    detection: Dict[str, object],
    clinical: Dict[str, object],
) -> Dict[str, object]:
    """Merge detection and clinical metrics into a single dict."""
    merged = {}
    merged.update(detection)
    merged.update(clinical)
    return merged


# ── Compute metrics for one fold ───────────────────────────────────────────────
def compute_fold_metrics(
    predictions_csv: Path,
    annotations_by_frame: Dict[str, Dict[str, object]],
    gt_by_unique_id: Dict[str, Dict[str, object]],
    video_id: str,
) -> Dict[str, object]:
    """Compute all metrics (detection + clinical) for a single fold."""
    pred_rows = load_csv_rows(predictions_csv)
    inference_by_frame = parse_predictions(pred_rows, video_id)

    comparison_rows, detection_summary = build_comparison_rows(
        annotations_by_frame=annotations_by_frame,
        inference_by_frame=inference_by_frame,
        gt_by_unique_id=gt_by_unique_id,
    )
    clinical = build_clinical_metrics(comparison_rows, video_id=video_id)

    return merge_detection_and_clinical(detection_summary, clinical)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load ground truth (shared across all videos)
    gt_rows = load_csv_rows(GT_CSV_PATH)
    gt_by_unique_id = parse_gt(gt_rows)

    # Discover video directories
    video_dirs = sorted([
        d for d in INFERENCE_ROOT.iterdir()
        if d.is_dir() and re.match(r"^video_\d{3}-\d{3}$", d.name)
    ])

    if not video_dirs:
        print(f"No se encontraron carpetas de video en: {INFERENCE_ROOT}")
        return

    print(f"Videos encontrados: {len(video_dirs)}")

    # per-model accumulator: model_name -> list of per-video averaged metrics
    model_all_video_metrics: Dict[str, List[Dict[str, object]]] = {m: [] for m in MODELS}

    for video_dir in video_dirs:
        # Extract video_id from folder name: "video_001-001" -> "001-001"
        video_id = video_dir.name.replace("video_", "")
        print(f"\n{'='*60}")
        print(f"Procesando video: {video_id}")

        # Load annotations for this video
        annotation_csv = ANNOTATIONS_DIR / f"lesiones_{video_id}.csv"
        if not annotation_csv.exists():
            print(f"  [WARN] No se encontró archivo de anotaciones: {annotation_csv}")
            print(f"  Saltando video {video_id}")
            continue

        annotation_rows = load_csv_rows(annotation_csv)
        annotations_by_frame = parse_annotations(annotation_rows, video_id=video_id)

        for model_name in MODELS:
            model_dir = video_dir / model_name
            if not model_dir.exists():
                print(f"  [WARN] No existe carpeta del modelo: {model_dir}")
                continue

            fold_metrics_list: List[Dict[str, object]] = []

            for fold_name in FOLDS:
                fold_dir = model_dir / fold_name
                predictions_csv = fold_dir / "predictions.csv"

                if not predictions_csv.exists():
                    print(f"  [WARN] No existe: {predictions_csv}")
                    continue

                fold_metrics = compute_fold_metrics(
                    predictions_csv=predictions_csv,
                    annotations_by_frame=annotations_by_frame,
                    gt_by_unique_id=gt_by_unique_id,
                    video_id=video_id,
                )
                fold_metrics_list.append(fold_metrics)
                print(f"  {model_name}/{fold_name}: OK (F1={fold_metrics.get('f1', 'N/A')})")

            if fold_metrics_list:
                # Average across folds for this video/model
                video_model_avg = average_metrics(fold_metrics_list)
                model_all_video_metrics[model_name].append(video_model_avg)
                print(
                    f"  {model_name} promedio folds: "
                    f"F1={video_model_avg.get('f1', 'N/A')}, "
                    f"Precision={video_model_avg.get('precision', 'N/A')}, "
                    f"Recall={video_model_avg.get('recall', 'N/A')}"
                )

    # ── Write final summary CSV (one row per model, averaged across all videos) ──
    print(f"\n{'='*60}")
    print("Generando resumen final...")

    summary_rows: List[Dict[str, object]] = []

    for model_name in MODELS:
        video_metrics = model_all_video_metrics[model_name]
        if not video_metrics:
            print(f"  [WARN] Sin datos para modelo: {model_name}")
            continue

        global_avg = average_metrics(video_metrics)
        global_std = std_metrics(video_metrics)
        global_avg["model"] = model_name
        global_avg["num_videos"] = len(video_metrics)
        # Merge std values into the same row
        global_avg.update(global_std)
        summary_rows.append(global_avg)

        print(
            f"  {model_name} (n={len(video_metrics)} videos): "
            f"F1={global_avg.get('f1', 'N/A')} ± {global_avg.get('f1_std', 'N/A')}, "
            f"Precision={global_avg.get('precision', 'N/A')} ± {global_avg.get('precision_std', 'N/A')}, "
            f"Recall={global_avg.get('recall', 'N/A')} ± {global_avg.get('recall_std', 'N/A')}, "
            f"Accuracy={global_avg.get('accuracy', 'N/A')} ± {global_avg.get('accuracy_std', 'N/A')}, "
            f"Malignacy Acc.={global_avg.get('Malignacy Acc.', 'N/A')} ± {global_avg.get('Malignacy Acc._std', 'N/A')}, "
            f"Loc. Acc.={global_avg.get('Loc. Acc.', 'N/A')} ± {global_avg.get('Loc. Acc._std', 'N/A')}, "
            f"Size (mm)={global_avg.get('Size (mm)', 'N/A')} ± {global_avg.get('Size (mm)_std', 'N/A')}"
        )

    # Write the final CSV — interleave mean and std columns
    output_fieldnames = ["model", "num_videos"]
    for key in ALL_METRIC_KEYS:
        output_fieldnames.append(key)
        output_fieldnames.append(f"{key}_std")
    output_path = OUTPUT_DIR / "metricas_promedio_global.csv"
    write_csv(output_path, summary_rows, output_fieldnames)

    print(f"\nCSV resumen global guardado en: {output_path}")


if __name__ == "__main__":
    main()

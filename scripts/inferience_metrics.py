#!/usr/bin/env python3
"""Compare lesion annotations vs. inference reports frame-by-frame."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_ANNOTATIONS_CSV = "dataset_real_colon/Anotaciones/lesiones_001-001.csv"
DEFAULT_INFERENCE_CSV = "outputs/real_colon_inference/video_001-001/frame_reporte.csv"
DEFAULT_GT_CSV = "dataset_real_colon/Datos real colon - lesion_info.csv"
DEFAULT_OUTPUT_CSV = "outputs/real_colon_inference/video_001-001/frame_comparacion_metricas.csv"
DEFAULT_SUMMARY_CSV = "outputs/real_colon_inference/video_001-001/metricas_evaluacion.csv"
DEFAULT_CLINICAL_METRICS_CSV = "outputs/real_colon_inference/video_001-001/metricas_generacion.csv"


POSITIVE_MARKS = {"si", "sí", "yes", "true", "1", "lesion", "lesión"}
NEGATIVE_MARKS = {"no", "false", "0"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compara anotaciones por frame con reportes de inferencia y genera "
            "un CSV con TP/FP/FN/TN y comparación de atributos."
        )
    )
    parser.add_argument("--annotations_csv", default=DEFAULT_ANNOTATIONS_CSV)
    parser.add_argument("--inference_csv", default=DEFAULT_INFERENCE_CSV)
    parser.add_argument("--gt_csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--output_csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--clinical_metrics_csv", default=DEFAULT_CLINICAL_METRICS_CSV)
    parser.add_argument(
        "--video_id",
        default=None,
        help=(
            "ID del video para construir frame_id (ej. 001-001). "
            "Si no se pasa, se infiere desde el nombre de annotations_csv."
        ),
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parent.parent / path).resolve()


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


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def infer_video_id(annotation_path: Path, annotation_rows: Sequence[Dict[str, str]]) -> str:
    by_name = re.search(r"lesiones_(\d{3}-\d{3})\.csv$", annotation_path.name)
    if by_name:
        return by_name.group(1)

    for row in annotation_rows:
        unique_id = normalize_spaces(row.get("unique_id", ""))
        match = re.match(r"^(\d{3}-\d{3})_\d+$", unique_id)
        if match:
            return match.group(1)

        xml_name = normalize_spaces(row.get("xml_file", ""))
        match = re.match(r"^(\d{3}-\d{3})_\d+\.xml$", xml_name)
        if match:
            return match.group(1)

    raise ValueError(
        "No se pudo inferir video_id. Pásalo explícitamente con --video_id (ej. 001-001)."
    )


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


def write_summary_csv(path: Path, summary: Dict[str, object]) -> None:
    fieldnames = [
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
    write_csv(path, [summary], fieldnames)


def build_clinical_metrics_rows(
    comparison_rows: Sequence[Dict[str, object]],
    video_id: str,
) -> Tuple[List[Dict[str, object]], bool]:
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
    paris_available = False

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

    malignancy_acc = (malignancy_correct / denominator * 100.0) if denominator else 0.0
    location_acc = (location_correct / denominator * 100.0) if denominator else 0.0
    size_mae = (sum(size_abs_errors) / len(size_abs_errors)) if size_abs_errors else ""

    row = {
        "video_id": video_id,
        "GT Positive Frames": denominator,
        "Malignacy Acc.": round(malignancy_acc / 100.0, 4),
        "Loc. Acc.": round(location_acc / 100.0, 4),
        "Size (mm)": round(size_mae, 4) if size_mae != "" else "",
    }
    if paris_available:
        row["Paris Acc."] = ""

    return [row], paris_available


def write_clinical_metrics_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    include_paris: bool,
) -> None:
    fieldnames = [
        "video_id",
        "GT Positive Frames",
        "Malignacy Acc.",
        "Loc. Acc.",
    ]
    if include_paris:
        fieldnames.append("Paris Acc.")
    fieldnames.append("Size (mm)")
    write_csv(path, rows, fieldnames)


def main() -> None:
    args = parse_args()

    annotations_csv = resolve_repo_path(args.annotations_csv)
    inference_csv = resolve_repo_path(args.inference_csv)
    gt_csv = resolve_repo_path(args.gt_csv)
    output_csv = resolve_repo_path(args.output_csv)
    summary_csv = resolve_repo_path(args.summary_csv)
    clinical_metrics_csv = resolve_repo_path(args.clinical_metrics_csv)

    annotation_rows = load_csv_rows(annotations_csv)
    inference_rows = load_csv_rows(inference_csv)
    gt_rows = load_csv_rows(gt_csv)

    video_id = str(args.video_id).strip() if args.video_id else infer_video_id(annotations_csv, annotation_rows)

    annotations_by_frame = parse_annotations(annotation_rows, video_id=video_id)
    inference_by_frame = parse_inference(inference_rows)
    gt_by_unique_id = parse_gt(gt_rows)

    comparison_rows, summary = build_comparison_rows(
        annotations_by_frame=annotations_by_frame,
        inference_by_frame=inference_by_frame,
        gt_by_unique_id=gt_by_unique_id,
    )
    clinical_metrics_rows, include_paris = build_clinical_metrics_rows(comparison_rows, video_id=video_id)

    comparison_columns = [
        "video_id",
        "frame",
        "frame_number",
        "annotation_has_lesion",
        "annotation_marca_lesion",
        "annotation_lesion_count",
        "annotation_unique_ids",
        "gt_unique_ids_found",
        "gt_unique_ids_missing",
        "gt_size_mm_values",
        "gt_site_values_raw",
        "gt_site_values_norm",
        "gt_histology_values_norm",
        "pred_has_lesion",
        "pred_size_mm",
        "pred_location_raw",
        "pred_location_norm",
        "pred_histology_norm",
        "confusion_label",
        "size_match",
        "location_match",
        "histology_match",
        "all_fields_match",
        "in_scope_for_metrics",
        "error_reason",
        "reporte_medico",
    ]
    write_csv(output_csv, comparison_rows, comparison_columns)
    write_summary_csv(summary_csv, summary)
    write_clinical_metrics_csv(clinical_metrics_csv, clinical_metrics_rows, include_paris)

    print(f"Video evaluado: {video_id}")
    print(f"Comparacion por frame: {output_csv}")
    print(f"Resumen de metricas: {summary_csv}")
    print(f"Metricas clinicas: {clinical_metrics_csv}")
    print(
        "GT (anotaciones) | "
        f"Frames con lesion={int(summary['gt_frames_with_lesion'])}, "
        f"Frames sin lesion={int(summary['gt_frames_without_lesion'])}"
    )
    print(
        "Confusion matrix | "
        f"TP={int(summary['tp'])}, FP={int(summary['fp'])}, "
        f"FN={int(summary['fn'])}, TN={int(summary['tn'])}"
    )


if __name__ == "__main__":
    main()

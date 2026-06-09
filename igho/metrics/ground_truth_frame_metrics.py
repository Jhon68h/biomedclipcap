#!/usr/bin/env python3
"""Evaluate IGHO frame reports using full-colonoscopy ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.inferience_metrics import (
    build_comparison_rows,
    normalize_histology,
    normalize_site,
    parse_float,
    parse_inference,
    write_csv,
)


DEFAULT_BASE_DIR = "igho"
DEFAULT_GT_CSV = "igho/ground_truth.csv"
DEFAULT_OUTPUT_DIR = "igho/metrics"
DEFAULT_CUT_OUTPUT_DIR = "igho/metrics_cut"
DEFAULT_FPS = 60
MODELS = ("biomedclip", "resnet", "vit")
FOLD_DIR_CANDIDATES = ("fold1", "fold_1")
VIDEO_DIR_RE = re.compile(r"^video_?\s*(?P<number>\d+)$")
FRAME_RE = re.compile(r"^(?:frame_)?(?P<number>\d+)$")


SUMMARY_COLUMNS = [
    "source_video",
    "video_dir",
    "model",
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
    "GT Positive Frames",
    "Malignacy Acc.",
    "Loc. Acc.",
    "Paris Acc.",
    "Size (mm)",
]

AVERAGE_COLUMNS = [
    "model",
    "videos_evaluated",
    "total_evaluated_frames",
    "total_gt_frames_with_lesion",
    "total_gt_frames_without_lesion",
    "total_tp",
    "total_fp",
    "total_fn",
    "total_tn",
    "avg_precision",
    "std_precision",
    "avg_recall",
    "std_recall",
    "avg_specificity",
    "std_specificity",
    "avg_f1",
    "std_f1",
    "avg_accuracy",
    "std_accuracy",
    "GT Positive Frames",
    "Malignacy Acc.",
    "Loc. Acc.",
    "Paris Acc.",
    "Size (mm)",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapta igho/ground_truth.csv al formato de scripts/inferience_metrics.py "
            "y genera metricas por video/modelo, incluyendo Paris."
        )
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--ground_truth_csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument(
        "--cut",
        action="store_true",
        help="Evalua carpetas video_N_cut y guarda por defecto en igho/metrics_cut.",
    )
    parser.add_argument(
        "--only_video",
        default=None,
        help="Evalua solo una carpeta especifica, por ejemplo video_1_cut o video_1.",
    )
    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def normalize_spaces(text: object) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, skipinitialspace=True)
        return [dict(row) for row in reader]


def normalized_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        normalize_spaces(key).lower(): normalize_spaces(value)
        for key, value in row.items()
        if key is not None
    }


def get_first(row: Dict[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = normalize_spaces(row.get(name, ""))
        if value:
            return value
    return ""


def normalize_video_dir(value: str) -> str:
    match = VIDEO_DIR_RE.fullmatch(normalize_spaces(value))
    if not match:
        return normalize_spaces(value)
    return f"video_{int(match.group('number'))}"


def time_to_seconds(time_str: str) -> int:
    cleaned = normalize_spaces(time_str)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", cleaned)
    if not match:
        raise ValueError(f"Tiempo invalido, se esperaba MM:SS: {time_str!r}")
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    if seconds >= 60:
        raise ValueError(f"Segundos invalidos en tiempo MM:SS: {time_str!r}")
    return minutes * 60 + seconds


def seconds_to_frame(seconds: int, fps: int = DEFAULT_FPS) -> int:
    return seconds * fps


def is_empty_time(value: str) -> bool:
    return normalize_spaces(value) in {"", "0", "00:00"}


def parse_time_pair(start_time: str, end_time: str, fps: int) -> Optional[Tuple[int, int]]:
    if is_empty_time(start_time) and is_empty_time(end_time):
        return None
    start_frame = seconds_to_frame(time_to_seconds(start_time), fps=fps)
    end_frame = seconds_to_frame(time_to_seconds(end_time), fps=fps)
    if end_frame < start_frame:
        raise ValueError(f"Rango con final antes de inicio: {start_time!r}, {end_time!r}")
    return start_frame, end_frame


def parse_report_frame_number(frame_value: str) -> Optional[int]:
    cleaned = normalize_spaces(frame_value)
    match = FRAME_RE.fullmatch(cleaned)
    if match:
        return int(match.group("number"))

    match = re.fullmatch(r"\d{3}-\d{3}_(?P<number>\d+)", cleaned)
    if match:
        return int(match.group("number"))
    return None


def extract_size_mm(row: Dict[str, str], informe: str) -> Optional[float]:
    value = get_first(row, ("size_mm", "size [mm]", "tamano_mm", "tamaño_mm", "size"))
    parsed = parse_float(value)
    if parsed is not None:
        return parsed

    match = re.search(r"(\d+(?:[.,]\d+)?)\s*mm\b", informe, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def extract_site(row: Dict[str, str], informe: str) -> str:
    value = get_first(row, ("site", "ubicacion", "ubicación", "location"))
    if value:
        return value

    lowered = informe.lower()
    if "sigmoide" in lowered or "sigmoid" in lowered:
        return "sigmoid colon"
    if "recto" in lowered or "rectum" in lowered:
        return "rectum"
    if "ciego" in lowered or "cecum" in lowered or "caecum" in lowered:
        return "cecum"
    if "ascendente" in lowered or "ascending" in lowered:
        return "ascending"
    if "descendente" in lowered or "descending" in lowered:
        return "descending"
    if "transverso" in lowered or "transverse" in lowered:
        return "transverse"
    return ""


def extract_histology(row: Dict[str, str], informe: str) -> str:
    value = get_first(row, ("histology", "histologia", "histología", "tipo_polipo", "tipo_pólipo"))
    if value:
        return value

    lowered = informe.lower()
    if "hiperplas" in lowered or "hyperplastic" in lowered:
        return "hyperplastic"
    if "adenoma" in lowered:
        return "adenoma"
    if "ssl" in lowered or "sessile serrated" in lowered:
        return "ssl"
    if "tsa" in lowered:
        return "tsa"
    if "serrated" in lowered or "serrado" in lowered:
        return "serrated"
    if "polipo" in lowered or "pólipo" in lowered or "polyp" in lowered:
        return "polyp"
    return ""


def normalize_paris(value: str) -> str:
    cleaned = normalize_spaces(value).upper()
    cleaned = cleaned.replace("O-", "0-").replace("O ", "0 ")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("PARIS", "")
    cleaned = cleaned.strip(":;-")
    if not cleaned:
        return ""
    if not cleaned.startswith("0"):
        cleaned = f"0-{cleaned}"
    elif len(cleaned) > 1 and cleaned[1] != "-":
        cleaned = f"0-{cleaned[1:]}"
    return cleaned


def extract_paris(text: str) -> str:
    lowered = normalize_spaces(text).lower()
    explicit = re.search(r"\bparis\s*[:\-]?\s*(0\s*[- ]?\s*(?:isp|ip|is|i{1,3})(?:[abc])?)", lowered)
    if explicit:
        return normalize_paris(explicit.group(1))

    if "subpedunculated" in lowered or "semi-pedunculated" in lowered:
        return "0-ISP"
    if "pedunculated" in lowered:
        return "0-IP"
    if "sessile" in lowered or "sesil" in lowered or "sésil" in lowered:
        return "0-IS"
    if "flat elevated" in lowered or "elevated" in lowered:
        return "0-IIA"
    if "depressed" in lowered:
        return "0-IIC"
    if "flat" in lowered:
        return "0-IIB"
    return ""


def build_ground_truth(csv_path: Path, fps: int) -> Dict[str, List[Dict[str, object]]]:
    gt_by_video: Dict[str, List[Dict[str, object]]] = {}
    counters: Dict[str, int] = {}

    for row_number, raw_row in enumerate(load_csv_rows(csv_path), start=2):
        row = normalized_row(raw_row)
        source_video = get_first(row, ("video", "source_video", "archivo", "filename", "file"))
        video_dir = normalize_video_dir(get_first(row, ("nombre", "video_dir", "video_folder", "carpeta")))
        informe = get_first(row, ("informe", "descripcion", "description"))
        start_time = get_first(row, ("inicio", "start", "start_time"))
        end_time = get_first(row, ("final", "fin", "end", "end_time"))
        frame_range = parse_time_pair(start_time, end_time, fps=fps)
        if frame_range is None:
            continue

        key = video_dir or source_video
        counters[key] = counters.get(key, 0) + 1
        polyp_id = get_first(row, ("polyp_id", "unique_object_id", "id"))
        if not polyp_id:
            polyp_id = f"{key}_polyp_{counters[key]}"

        paris = get_first(row, ("paris", "clasificacion_paris", "clasificación_paris")) or extract_paris(informe)
        histology = extract_histology(row, informe)
        site = extract_site(row, informe)
        size_mm = extract_size_mm(row, informe)

        gt_by_video.setdefault(key, []).append(
            {
                "source_video": source_video,
                "video_dir": video_dir,
                "polyp_id": polyp_id,
                "informe": informe,
                "start_frame": frame_range[0],
                "end_frame": frame_range[1],
                "size_mm": size_mm,
                "site_raw": site,
                "site_norm": normalize_site(site),
                "histology_raw": histology,
                "histology_norm": normalize_histology(histology),
                "paris_norm": normalize_paris(paris),
                "row_number": row_number,
            }
        )
    return gt_by_video


def video_sort_number(path: Path) -> int:
    match = re.fullmatch(r"video_(?P<number>\d+)(?:_cut)?", path.name)
    return int(match.group("number")) if match else -1


def find_video_dirs(base_dir: Path, use_cut: bool = False) -> List[Path]:
    pattern = r"video_\d+_cut" if use_cut else r"video_\d+"
    video_dirs = [path for path in base_dir.iterdir() if path.is_dir() and re.fullmatch(pattern, path.name)]
    return sorted(video_dirs, key=video_sort_number)


def base_video_dir_name(video_dir_name: str) -> str:
    return re.sub(r"_cut$", "", video_dir_name)


def find_frame_reports(video_dir: Path) -> Dict[str, Path]:
    reports: Dict[str, Path] = {}
    for model in MODELS:
        for fold_dir in FOLD_DIR_CANDIDATES:
            report_csv = video_dir / model / fold_dir / "frame_reporte.csv"
            if report_csv.exists():
                reports[model] = report_csv
                break
    return reports


def source_video_from_config(video_dir: Path) -> str:
    config_path = video_dir / "run_config.json"
    if not config_path.exists():
        return ""

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    images_root = normalize_spaces(config.get("images_root", ""))
    return Path(images_root).stem if images_root else ""


def build_annotations(
    inference_rows: Sequence[Dict[str, str]],
    polyps: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    annotations: Dict[str, Dict[str, object]] = {}

    for row in inference_rows:
        frame = normalize_spaces(row.get("frame", ""))
        frame_number = parse_report_frame_number(frame)
        if not frame or frame_number is None:
            continue

        matching_polyps = [
            polyp
            for polyp in polyps
            if int(polyp["start_frame"]) <= frame_number <= int(polyp["end_frame"])
        ]
        unique_ids = {str(polyp["polyp_id"]) for polyp in matching_polyps}
        has_lesion = bool(unique_ids)

        annotations[frame] = {
            "frame_number": frame_number,
            "marca_values": {"si"} if has_lesion else {"no"},
            "has_lesion": has_lesion,
            "unique_ids": unique_ids,
            "max_lesion_count": len(unique_ids),
        }

    return annotations


def build_gt_by_unique_id(polyps: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    gt: Dict[str, Dict[str, object]] = {}
    for polyp in polyps:
        gt[str(polyp["polyp_id"])] = {
            "size_mm": polyp.get("size_mm"),
            "site_raw": polyp.get("site_raw", ""),
            "site_norm": polyp.get("site_norm", ""),
            "histology_extended": polyp.get("histology_raw", ""),
            "histology_class": polyp.get("histology_raw", ""),
            "histology_norm": polyp.get("histology_norm", ""),
        }
    return gt


def split_ids(text: object) -> List[str]:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return []
    return [item for item in cleaned.split(";") if item]


def split_semicolon_values(text: object) -> List[str]:
    cleaned = normalize_spaces(text)
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(";") if item.strip()]


def add_named_clinical_metrics(
    summary: Dict[str, object],
    comparison_rows: Sequence[Dict[str, object]],
    polyps: Sequence[Dict[str, object]],
) -> None:
    paris_by_id = {
        str(polyp["polyp_id"]): str(polyp.get("paris_norm", ""))
        for polyp in polyps
        if str(polyp.get("paris_norm", ""))
    }

    gt_positive_rows = [
        row
        for row in comparison_rows
        if str(row.get("in_scope_for_metrics", "")) == "1"
        and str(row.get("annotation_has_lesion", "")) == "1"
    ]

    malignancy_correct = 0
    location_correct = 0
    paris_evaluated = 0
    paris_correct = 0
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

        gt_paris_values = {
            paris_by_id[uid]
            for uid in split_ids(row.get("annotation_unique_ids", ""))
            if uid in paris_by_id
        }
        if not gt_paris_values:
            continue

        paris_evaluated += 1
        pred_paris = extract_paris(str(row.get("reporte_medico", "")))
        if pred_paris and pred_paris in gt_paris_values:
            paris_correct += 1

    denominator = len(gt_positive_rows)
    summary["GT Positive Frames"] = denominator
    summary["Malignacy Acc."] = round((malignancy_correct / denominator), 4) if denominator else 0.0
    summary["Loc. Acc."] = round((location_correct / denominator), 4) if denominator else 0.0
    summary["Paris Acc."] = round((paris_correct / paris_evaluated), 4) if paris_evaluated else 0.0
    summary["Size (mm)"] = round((sum(size_abs_errors) / len(size_abs_errors)), 4) if size_abs_errors else ""


def compute_report_metrics(
    report_csv: Path,
    polyps: Sequence[Dict[str, object]],
    video_dir: str,
    source_video: str,
    model: str,
) -> Dict[str, object]:
    inference_rows = load_csv_rows(report_csv)
    comparison_rows, summary = build_comparison_rows(
        annotations_by_frame=build_annotations(inference_rows, polyps),
        inference_by_frame=parse_inference(inference_rows),
        gt_by_unique_id=build_gt_by_unique_id(polyps),
    )
    add_named_clinical_metrics(summary, comparison_rows, polyps)
    summary.update(
        {
            "source_video": source_video,
            "video_dir": video_dir,
            "model": model,
            "report_csv": str(report_csv),
        }
    )
    return summary


def average_metric(rows: Sequence[Dict[str, object]], metric: str) -> float:
    values = numeric_values(rows, metric)
    return sum(values) / len(values) if values else 0.0


def numeric_values(rows: Sequence[Dict[str, object]], metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = row.get(metric, "")
        if value == "":
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def std_metric(rows: Sequence[Dict[str, object]], metric: str) -> float:
    values = numeric_values(rows, metric)
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def build_average_rows(summary_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows_by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in summary_rows:
        rows_by_model.setdefault(str(row["model"]), []).append(row)

    output_rows: List[Dict[str, object]] = []
    for model in MODELS:
        rows = rows_by_model.get(model, [])
        if not rows:
            continue
        output_rows.append(
            {
                "model": model,
                "videos_evaluated": len(rows),
                "total_evaluated_frames": sum(int(row["evaluated_frames"]) for row in rows),
                "total_gt_frames_with_lesion": sum(int(row["gt_frames_with_lesion"]) for row in rows),
                "total_gt_frames_without_lesion": sum(int(row["gt_frames_without_lesion"]) for row in rows),
                "total_tp": sum(int(row["tp"]) for row in rows),
                "total_fp": sum(int(row["fp"]) for row in rows),
                "total_fn": sum(int(row["fn"]) for row in rows),
                "total_tn": sum(int(row["tn"]) for row in rows),
                "avg_precision": average_metric(rows, "precision"),
                "std_precision": std_metric(rows, "precision"),
                "avg_recall": average_metric(rows, "recall"),
                "std_recall": std_metric(rows, "recall"),
                "avg_specificity": average_metric(rows, "specificity"),
                "std_specificity": std_metric(rows, "specificity"),
                "avg_f1": average_metric(rows, "f1"),
                "std_f1": std_metric(rows, "f1"),
                "avg_accuracy": average_metric(rows, "accuracy"),
                "std_accuracy": std_metric(rows, "accuracy"),
                "GT Positive Frames": sum(int(row["GT Positive Frames"]) for row in rows),
                "Malignacy Acc.": average_metric(rows, "Malignacy Acc."),
                "Loc. Acc.": average_metric(rows, "Loc. Acc."),
                "Paris Acc.": average_metric(rows, "Paris Acc."),
                "Size (mm)": average_metric(rows, "Size (mm)"),
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    base_dir = resolve_repo_path(args.base_dir)
    gt_csv = resolve_repo_path(args.ground_truth_csv)
    output_dir = resolve_repo_path(
        args.output_dir or (DEFAULT_CUT_OUTPUT_DIR if args.cut else DEFAULT_OUTPUT_DIR)
    )

    gt_by_video = build_ground_truth(gt_csv, fps=args.fps)
    all_summary_rows: List[Dict[str, object]] = []

    for video_dir in find_video_dirs(base_dir, use_cut=args.cut):
        if args.only_video and video_dir.name != args.only_video:
            continue

        source_video = source_video_from_config(video_dir)
        gt_video_key = base_video_dir_name(video_dir.name)
        polyps = gt_by_video.get(gt_video_key) or gt_by_video.get(source_video, [])
        output_source_video = str(polyps[0].get("source_video", "")) if polyps else source_video
        video_rows: List[Dict[str, object]] = []

        for model, report_csv in find_frame_reports(video_dir).items():
            summary = compute_report_metrics(
                report_csv=report_csv,
                polyps=polyps,
                video_dir=video_dir.name,
                source_video=output_source_video,
                model=model,
            )
            video_rows.append(summary)
            all_summary_rows.append(summary)

        write_csv(output_dir / f"{video_dir.name}_metricas.csv", video_rows, SUMMARY_COLUMNS)

    write_csv(output_dir / "promedio_todos_los_videos.csv", build_average_rows(all_summary_rows), AVERAGE_COLUMNS)
    print(f"Metricas guardadas en: {output_dir}")


if __name__ == "__main__":
    main()

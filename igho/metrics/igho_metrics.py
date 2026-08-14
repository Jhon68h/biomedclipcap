"""Evaluate IGHO frame reports using full-colonoscopy ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    warnings.warn(
        "NLTK no esta instalado. El puntaje BLEU no se calculara. "
        "Instalalo ejecutando: pip install nltk"
    )

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


DEFAULT_BASE_DIR = "igho/videos"
DEFAULT_GT_CSV = "igho/igho_dataset.csv"
DEFAULT_OUTPUT_DIR = "igho/videos/metrics"
DEFAULT_CUT_OUTPUT_DIR = "igho/metrics_cut"
MODELS = ("biomedclip", "resnet", "vit")
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
    "BLEU-1", 
    "BLEU-4", 
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
    "BLEU-1", 
    "BLEU-4", 
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adapta igho/ground_truth.csv al formato de scripts/inferience_metrics.py "
            "y genera metricas por video/modelo, incluyendo Paris y BLEU."
        )
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--ground_truth_csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--cut",
        action="store_true",
        help="Evalua carpetas video_N_cut y guarda por defecto en igho/metrics_cut.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Evalua solo una carpeta especifica, por ejemplo video_1_cut o video_1.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Especifica manualmente el frame de inicio del ground truth (opcional).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Especifica manualmente el frame de fin del ground truth (opcional).",
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


def video_dir_from_source_video(value: str) -> str:
    cleaned = normalize_spaces(value)
    match = re.search(r"(?:^|_)(?P<number>\d+)$", cleaned)
    if not match:
        return ""
    return f"video_{int(match.group('number'))}"


def parse_report_frame_number(frame_value: str) -> Optional[int]:
    cleaned = normalize_spaces(frame_value)
    match = FRAME_RE.fullmatch(cleaned)
    if match:
        return int(match.group("number"))

    match = re.fullmatch(r"\d{3}-\d{3}_(?P<number>\d+)", cleaned)
    if match:
        return int(match.group("number"))
    return None

def extract_size_mm(row: Dict[str, str]) -> Optional[float]:
    value = get_first(row, ("size",))
    return parse_float(value)

def extract_site(row: Dict[str, str]) -> str:
    return get_first(row, ("lesion_location",)) or ""

def extract_histology(row: Dict[str, str]) -> str:
    diagnosis = normalize_spaces(get_first(row, ("diagnosis",))).upper()
    if re.search(r"\bNICE\s*III\b", diagnosis):
        return "cancer"
    if re.search(r"\bNICE\s*II\b", diagnosis):
        return "adenoma"
    if re.search(r"\bNICE\s*I\b", diagnosis):
        return "hyperplastic"
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

def extract_paris(value: object) -> str:
    if isinstance(value, dict):
        return normalize_paris(get_first(value, ("paris",)))

    lowered = normalize_spaces(value).lower()
    explicit = re.search(r"\bparis\s*[:\-]?\s*(0\s*[- ]?\s*(?:isp|ip|is|i{1,3}|ii[abc]?)(?:[abc])?)", lowered)
    if explicit:
        return normalize_paris(explicit.group(1))

    if "subpedunculated" in lowered or "semi-pedunculated" in lowered:
        return "0-ISP"
    if "pedunculated" in lowered:
        return "0-IP"
    if "sessile" in lowered or "sesil" in lowered or "sesil" in lowered:
        return "0-IS"
    if "flat elevated" in lowered or "elevated" in lowered:
        return "0-IIA"
    if "depressed" in lowered:
        return "0-IIC"
    if "flat" in lowered:
        return "0-IIB"
    return ""


def build_ground_truth(
    csv_path: Path, 
    manual_start_frame: Optional[int] = None,
    manual_end_frame: Optional[int] = None,
    target_video_key: Optional[str] = None,
) -> Dict[str, List[Dict[str, object]]]:
    gt_by_video: Dict[str, List[Dict[str, object]]] = {}
    counters: Dict[str, int] = {}

    for row_number, raw_row in enumerate(load_csv_rows(csv_path), start=2):
        row = normalized_row(raw_row)

        source_video = get_first(
            row,
            ("id", "video", "source_video", "archivo", "filename", "file"),
        )

        video_dir = normalize_video_dir(
            get_first(
                row,
                ("nombre", "video_dir", "video_folder", "carpeta"),
            )
        )
        if not video_dir:
            video_dir = video_dir_from_source_video(source_video)

        key = video_dir or source_video

        frame_range = None
        
        if (manual_start_frame is not None and manual_end_frame is not None
                and (target_video_key is None or key == target_video_key)):
            frame_range = (manual_start_frame, manual_end_frame)
        else:
            start_val = get_first(row, ("start", "start_frame", "frame_inicio", "inicio_frame"))
            end_val = get_first(row, ("end", "end_frame", "frame_fin", "fin_frame"))
            
            if start_val and end_val:
                try:
                    frame_range = (int(float(start_val)), int(float(end_val)))
                except ValueError:
                    pass

        if frame_range is None:
            continue

        histology = extract_histology(row) 
        counters[key] = counters.get(key, 0) + 1

        polyp_id = get_first(
            row,
            ("polyp_id", "unique_object_id", "lesion_id", "object_id"),
        )

        if not polyp_id:
            polyp_id = f"{key}_polyp_{counters[key]}"

        paris = get_first(
            row,
            ("paris", "clasificacion_paris", "clasificación_paris"),
        )

        site = extract_site(row)
        size_mm = extract_size_mm(row)
        report = get_first(row, ("report", "reporte", "reporte_medico"))

        gt_by_video.setdefault(key, []).append(
            {
                "source_video": source_video,
                "video_dir": video_dir,
                "polyp_id": polyp_id,
                "start_frame": frame_range[0],
                "end_frame": frame_range[1],
                "histology_raw": histology,
                "histology_norm": normalize_histology(histology),
                "size_mm": size_mm,
                "site_raw": site,
                "site_norm": normalize_site(site),
                "paris_norm": normalize_paris(paris),
                "report_gt": report,
                "row_number": row_number,
            }
        )
    return gt_by_video


def video_sort_number(path: Path) -> int:
    match = re.search(r"video_(?P<number>\d+)", path.name)
    return int(match.group("number")) if match else -1


def find_video_dirs(base_dir: Path, use_cut: bool = False) -> List[Path]:
    pattern = r"video_\d+.*_cut$" if use_cut else r"video_\d+.*"
    video_dirs = [path for path in base_dir.iterdir() if path.is_dir() and re.match(pattern, path.name)]
    return sorted(video_dirs, key=video_sort_number)


def base_video_dir_name(video_dir_name: str) -> str:
    match = re.search(r"video_(\d+)", video_dir_name)
    if match:
        return f"video_{match.group(1)}"
    return video_dir_name


def find_frame_reports(video_dir: Path) -> List[Tuple[str, str, Path]]:
    reports = []
    for model in MODELS:
        model_dir = video_dir / model
        if not model_dir.is_dir():
            continue
            
        for fold_dir in model_dir.iterdir():
            if fold_dir.is_dir() and "fold" in fold_dir.name.lower():
                for csv_name in ("frame_reporte.csv", "predictions.csv", "generated_captions.csv", "results.csv"):
                    report_csv = fold_dir / csv_name
                    if report_csv.exists():
                        reports.append((model, fold_dir.name, report_csv))
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
            "report_gt": polyp.get("report_gt", ""), 
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
    polyps_by_id = {str(p["polyp_id"]): p for p in polyps}
    
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
    
    bleu1_scores: List[float] = []
    bleu4_scores: List[float] = []

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
        
        if gt_paris_values:
            paris_evaluated += 1
            pred_paris = extract_paris(str(row.get("reporte_medico", "")))
            if pred_paris and pred_paris in gt_paris_values:
                paris_correct += 1
                
        # --- Cálculo de BLEU-1 y BLEU-4 Score ---
        pred_caption = str(row.get("reporte_medico", ""))
        gt_reports = [
            str(polyps_by_id[uid].get("report_gt", ""))
            for uid in split_ids(row.get("annotation_unique_ids", ""))
            if uid in polyps_by_id and polyps_by_id[uid].get("report_gt")
        ]

        if HAS_NLTK and pred_caption and gt_reports:
            candidate = pred_caption.lower().split()
            references = [ref.lower().split() for ref in gt_reports if ref]
            if references and candidate:
                try:
                    cc = SmoothingFunction().method1
                    b1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=cc)
                    b4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=cc)
                    
                    bleu1_scores.append(b1)
                    bleu4_scores.append(b4)
                except Exception:
                    pass

    denominator = len(gt_positive_rows)
    summary["GT Positive Frames"] = denominator
    summary["Malignacy Acc."] = round((malignancy_correct / denominator), 4) if denominator else 0.0
    summary["Loc. Acc."] = round((location_correct / denominator), 4) if denominator else 0.0
    summary["Paris Acc."] = round((paris_correct / paris_evaluated), 4) if paris_evaluated else 0.0
    summary["Size (mm)"] = round((sum(size_abs_errors) / len(size_abs_errors)), 4) if size_abs_errors else ""
    summary["BLEU-1"] = round((sum(bleu1_scores) / len(bleu1_scores)), 4) if bleu1_scores else ""
    summary["BLEU-4"] = round((sum(bleu4_scores) / len(bleu4_scores)), 4) if bleu4_scores else ""


def compute_report_metrics(
    report_csv: Path,
    polyps: Sequence[Dict[str, object]],
    video_dir: str,
    source_video: str,
    model: str,
    fold_name: str,
) -> Dict[str, object]:
    
    raw_rows = load_csv_rows(report_csv)
    inference_rows = []
    
    for row in raw_rows:
        img_path = row.get("image_path", "")
        caption = row.get("generated_caption", "")
        
        if img_path and caption:
            frame_name = Path(img_path).stem 
            inference_rows.append({
                "frame": frame_name,
                "reporte_medico": caption
            })
        else:
            inference_rows.append(row)

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
            "base_model": model,                   
            "model": f"{model}_{fold_name}",       
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
    
    for model_key in sorted(rows_by_model.keys()):
        rows = rows_by_model.get(model_key, [])
        if not rows:
            continue
            
        output_rows.append(
            {
                "model": model_key,
                "videos_evaluated": len(set(str(row.get("video_dir", "")) for row in rows)),
                "total_evaluated_frames": round(sum(float(row["evaluated_frames"]) for row in rows), 2),
                "total_gt_frames_with_lesion": round(sum(float(row["gt_frames_with_lesion"]) for row in rows), 2),
                "total_gt_frames_without_lesion": round(sum(float(row["gt_frames_without_lesion"]) for row in rows), 2),
                "total_tp": round(sum(float(row["tp"]) for row in rows), 2),
                "total_fp": round(sum(float(row["fp"]) for row in rows), 2),
                "total_fn": round(sum(float(row["fn"]) for row in rows), 2),
                "total_tn": round(sum(float(row["tn"]) for row in rows), 2),
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
                "GT Positive Frames": round(sum(float(row["GT Positive Frames"]) for row in rows), 2),
                "Malignacy Acc.": average_metric(rows, "Malignacy Acc."),
                "Loc. Acc.": average_metric(rows, "Loc. Acc."),
                "Paris Acc.": average_metric(rows, "Paris Acc."),
                "Size (mm)": average_metric(rows, "Size (mm)"),
                "BLEU-1": average_metric(rows, "BLEU-1"),
                "BLEU-4": average_metric(rows, "BLEU-4"),
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()
    
    if args.video:
        args.video = Path(args.video).name

    base_dir = resolve_repo_path(args.base_dir)
    gt_csv = resolve_repo_path(args.ground_truth_csv)
    output_dir = resolve_repo_path(
        args.output_dir or (DEFAULT_CUT_OUTPUT_DIR if args.cut else DEFAULT_OUTPUT_DIR)
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)

    target_key = base_video_dir_name(args.video) if args.video else None
    gt_by_video = build_ground_truth(
        gt_csv, 
        manual_start_frame=args.start,
        manual_end_frame=args.end,
        target_video_key=target_key,
    )
    
    all_video_averages: List[Dict[str, object]] = []

    for video_dir in find_video_dirs(base_dir, use_cut=args.cut):
        if args.video and video_dir.name != args.video:
            continue

        source_video = source_video_from_config(video_dir)
        gt_video_key = base_video_dir_name(video_dir.name)
        polyps = gt_by_video.get(gt_video_key) or gt_by_video.get(source_video, [])
        output_source_video = str(polyps[0].get("source_video", "")) if polyps else source_video
        
        video_rows_folds: List[Dict[str, object]] = []

        for model, fold_name, report_csv in find_frame_reports(video_dir):
            summary = compute_report_metrics(
                report_csv=report_csv,
                polyps=polyps,
                video_dir=video_dir.name,
                source_video=output_source_video,
                model=model,
                fold_name=fold_name,
            )
            video_rows_folds.append(summary)

        video_rows_final: List[Dict[str, object]] = []
        by_base_model = {}
        for row in video_rows_folds:
            by_base_model.setdefault(str(row["base_model"]), []).append(row)
            video_rows_final.append(row)

        for base_model, rows in by_base_model.items():
            if len(rows) > 0:
                avg_row = {
                    "source_video": rows[0]["source_video"],
                    "video_dir": rows[0]["video_dir"],
                    "model": f"{base_model}_promedio", 
                }
                
                for col in SUMMARY_COLUMNS:
                    if col in ["source_video", "video_dir", "model"]: 
                        continue
                        
                    vals = [float(r[col]) for r in rows if r.get(col) not in ("", None)]
                    if vals:
                        avg_row[col] = round(sum(vals) / len(vals), 4)
                    else:
                        avg_row[col] = ""
                
                if len(rows) > 1:
                    video_rows_final.append(avg_row)
                    
                all_video_averages.append(avg_row)

        if video_rows_final:
            write_csv(output_dir / f"{video_dir.name}_metricas.csv", video_rows_final, SUMMARY_COLUMNS)

    if all_video_averages:
        write_csv(output_dir / "promedio_todos_los_videos.csv", build_average_rows(all_video_averages), AVERAGE_COLUMNS)
    print(f"Metricas guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
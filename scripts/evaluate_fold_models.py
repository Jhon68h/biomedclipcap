import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path


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
    r"\bno\s+polyps?\b",
    r"\bno\s+visible\s+polyps?\b",
    r"\bno\s+evidence\s+of\s+polyps?\b",
    r"\bwithout\s+polyps?\b",
]


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("<|endoftext|>", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def clean_caption(text: str) -> str:
    text = normalize_text(text)
    for phrase in STOP_PHRASES:
        text = text.replace(phrase, "")
    return " ".join(text.split())


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def extract_attr(text: str, vocab):
    text = normalize_text(text)
    for value in vocab:
        if value in text:
            return value
    return None


def extract_lesion(text: str):
    text = normalize_text(text)
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text):
            return "no_polyp"
    for value in LESIONS:
        pattern = rf"\b{re.escape(value)}s?\b"
        if re.search(pattern, text):
            return value
    return None


def extract_size(text: str):
    match = re.search(r"(\d+)\s*mm", normalize_text(text))
    return int(match.group(1)) if match else None


def safe_div(num: float, den: float):
    return num / den if den else 0.0


def ngrams(tokens, n: int):
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def modified_precision(ref_tokens, pred_tokens, n: int):
    pred_counts = Counter(ngrams(pred_tokens, n))
    if not pred_counts:
        return 0.0
    ref_counts = Counter(ngrams(ref_tokens, n))
    clipped = 0
    total = 0
    for gram, count in pred_counts.items():
        clipped += min(count, ref_counts.get(gram, 0))
        total += count
    return safe_div(clipped, total)


def brevity_penalty(ref_len: int, pred_len: int):
    if pred_len == 0:
        return 0.0
    if pred_len > ref_len:
        return 1.0
    return math.exp(1.0 - (ref_len / pred_len))


def bleu_precision_mean(rows, n: int):
    if not rows:
        return 0.0
    values = []
    for row in rows:
        ref_tokens = tokenize(row["gt_caption_clean"])
        pred_tokens = tokenize(row["pred_caption_clean"])
        values.append(modified_precision(ref_tokens, pred_tokens, n))
    return sum(values) / len(values)


def corpus_bleu4(rows):
    if not rows:
        return 0.0
    precision_logs = []
    total_ref_len = 0
    total_pred_len = 0
    for n in range(1, 5):
        p = max(bleu_precision_mean(rows, n), 1e-12)
        precision_logs.append(math.log(p))
    for row in rows:
        total_ref_len += len(tokenize(row["gt_caption_clean"]))
        total_pred_len += len(tokenize(row["pred_caption_clean"]))
    bp = brevity_penalty(total_ref_len, total_pred_len)
    return bp * math.exp(sum(precision_logs) / 4.0)


def classification_metrics(rows):
    tp = tn = fp = fn = 0
    for row in rows:
        y_true = row["label_true"] == "polyp"
        y_pred = row["label_pred"] == "polyp"
        if y_true and y_pred:
            tp += 1
        elif (not y_true) and (not y_pred):
            tn += 1
        elif (not y_true) and y_pred:
            fp += 1
        else:
            fn += 1

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "counts": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
        "specificity": safe_div(tn, tn + fp),
    }


def evaluate_clinical(rows):
    evaluated = []
    for row in rows:
        gt_caption = row["caption_gt"]
        pred_caption = row["caption_pred"]

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
                "image_path": row["image_path"],
                "case": row["case"],
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
                "size_correct": int(gt_size == pred_size) if gt_size is not None else None,
                "size_error": abs(gt_size - pred_size) if (gt_size is not None and pred_size is not None) else None,
            }
        )
    return evaluated


def mean_over_present(rows, key: str):
    values = [row[key] for row in rows if row[key] is not None]
    return safe_div(sum(values), len(values)), len(values)


def mean_size_error(rows):
    values = [row["size_error"] for row in rows if row["size_error"] is not None]
    return safe_div(sum(values), len(values)), len(values)


def generation_metrics(rows):
    prepared = []
    for row in rows:
        prepared.append(
            {
                **row,
                "gt_caption_clean": clean_caption(row["caption_gt"]),
                "pred_caption_clean": clean_caption(row["caption_pred"]),
            }
        )

    clinical_rows = evaluate_clinical(prepared)

    malignacy_accuracy, malignacy_support = mean_over_present(clinical_rows, "type_correct")
    loc_accuracy, loc_support = mean_over_present(clinical_rows, "location_correct")
    paris_accuracy, paris_support = mean_over_present(clinical_rows, "morph_correct")
    size_accuracy, size_support = mean_over_present(clinical_rows, "size_correct")
    size_mae_mm, size_mae_support = mean_size_error(clinical_rows)

    return {
        "bleu1": bleu_precision_mean(prepared, 1),
        "bleu4": corpus_bleu4(prepared),
        "malignacy_accuracy": malignacy_accuracy,
        "malignancy_accuracy": malignacy_accuracy,
        "loc_accuracy": loc_accuracy,
        "paris_accuracy": paris_accuracy,
        "size_accuracy": size_accuracy,
        "size_mae_mm": size_mae_mm,
        "supports": {
            "malignacy": malignacy_support,
            "location": loc_support,
            "paris": paris_support,
            "size_accuracy": size_support,
            "size_mae": size_mae_support,
        },
        "clinical_preview": clinical_rows[:20],
    }


def evaluate_fold_prediction_csv(path: Path):
    rows = read_csv(path)
    return {
        "fold": path.parents[1].name,
        "num_rows": len(rows),
        "source_csv": str(path),
        "classification": classification_metrics(rows),
        "generation": generation_metrics(rows),
    }


def aggregate_model(model_dir: Path, task: str):
    prediction_csvs = sorted((model_dir / task).glob("fold/*/inference/val_predictions.csv"))
    if not prediction_csvs:
        return None

    per_fold = [evaluate_fold_prediction_csv(path) for path in prediction_csvs]

    all_rows = []
    for path in prediction_csvs:
        all_rows.extend(read_csv(path))

    return {
        "model": model_dir.name,
        "task": task,
        "num_folds": len(per_fold),
        "num_rows": len(all_rows),
        "classification": classification_metrics(all_rows),
        "generation": generation_metrics(all_rows),
        "per_fold": per_fold,
    }


def comparison_row(summary):
    cls = summary["classification"]
    gen = summary["generation"]
    return {
        "model": summary["model"],
        "task": summary["task"],
        "num_rows": summary["num_rows"],
        "accuracy": cls["accuracy"],
        "precision": cls["precision"],
        "recall": cls["recall"],
        "f1": cls["f1"],
        "specificity": cls["specificity"],
        "bleu1": gen["bleu1"],
        "bleu4": gen["bleu4"],
        "malignacy_accuracy": gen["malignacy_accuracy"],
        "loc_accuracy": gen["loc_accuracy"],
        "paris_accuracy": gen["paris_accuracy"],
        "size_accuracy": gen["size_accuracy"],
        "size_mae_mm": gen["size_mae_mm"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute classification and generation metrics from fold/* inference outputs"
    )
    parser.add_argument("--fold_root", default="fold")
    parser.add_argument("--task", default="positive_vs_negative")
    args = parser.parse_args()

    fold_root = Path(args.fold_root).resolve()
    model_dirs = sorted(
        path for path in fold_root.iterdir() if path.is_dir() and (path / args.task / "folds").exists()
    )
    if not model_dirs:
        raise FileNotFoundError(f"No model directories found under {fold_root} for task {args.task}")

    comparison_rows = []
    comparison_payload = {"task": args.task, "models": []}

    for model_dir in model_dirs:
        summary = aggregate_model(model_dir, args.task)
        if summary is None:
            continue

        comparison_payload["models"].append(summary)
        comparison_rows.append(comparison_row(summary))

        per_model_json = model_dir / args.task / "inference_metrics.json"
        per_model_csv = model_dir / args.task / "inference_metrics.csv"
        write_json(per_model_json, summary)
        write_csv(per_model_csv, [comparison_row(summary)], list(comparison_row(summary).keys()))

    comparison_json = fold_root / f"{args.task}_model_comparison.json"
    comparison_csv = fold_root / f"{args.task}_model_comparison.csv"
    write_json(comparison_json, comparison_payload)
    if comparison_rows:
        write_csv(comparison_csv, comparison_rows, list(comparison_rows[0].keys()))


if __name__ == "__main__":
    main()

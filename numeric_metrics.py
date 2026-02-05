import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple


def safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else 0.0


def parse_value(val: str) -> str:
    return val.strip()


def to_float(val: str) -> float:
    try:
        return float(val)
    except Exception:
        return float("nan")


def load_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_labels(
    rows: List[Dict[str, Any]],
    y_true_col: str,
    y_pred_col: str,
    y_score_col: str,
    threshold: float,
    pos_label: str,
) -> List[Tuple[int, int]]:
    labels = []
    for i, r in enumerate(rows, start=1):
        if y_true_col not in r:
            raise KeyError(f"Falta columna y_true_col={y_true_col} en fila {i}")
        y_true_raw = parse_value(r[y_true_col])
        y_true = 1 if y_true_raw == pos_label else 0

        if y_score_col:
            if y_score_col not in r:
                raise KeyError(f"Falta columna y_score_col={y_score_col} en fila {i}")
            score = to_float(r[y_score_col])
            if score != score:  # nan
                raise ValueError(f"Score no numérico en fila {i}: {r[y_score_col]}")
            y_pred = 1 if score >= threshold else 0
        else:
            if y_pred_col not in r:
                raise KeyError(f"Falta columna y_pred_col={y_pred_col} en fila {i}")
            y_pred_raw = parse_value(r[y_pred_col])
            y_pred = 1 if y_pred_raw == pos_label else 0

        labels.append((y_true, y_pred))
    return labels


def confusion(labels: List[Tuple[int, int]]) -> Dict[str, int]:
    tp = tn = fp = fn = 0
    for y_true, y_pred in labels:
        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 0 and y_pred == 0:
            tn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def compute_metrics(cm: Dict[str, int]) -> Dict[str, float]:
    tp = cm["tp"]
    tn = cm["tn"]
    fp = cm["fp"]
    fn = cm["fn"]

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    specificity = safe_div(tn, tn + fp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="CSV con labels")
    parser.add_argument("--y_true_col", required=True, help="Nombre de la columna de verdad")
    parser.add_argument("--y_pred_col", default="", help="Nombre de la columna predicha (si no usas score)")
    parser.add_argument("--y_score_col", default="", help="Nombre de la columna score/probabilidad")
    parser.add_argument("--threshold", type=float, default=0.5, help="Umbral si usas score")
    parser.add_argument("--pos_label", default="1", help="Valor considerado positivo en el CSV")
    parser.add_argument("--out_dir", default="metrics", help="Carpeta de salida")
    args = parser.parse_args()

    if not args.y_pred_col and not args.y_score_col:
        raise ValueError("Debes indicar --y_pred_col o --y_score_col")

    rows = load_rows(args.input_csv)
    labels = build_labels(
        rows=rows,
        y_true_col=args.y_true_col,
        y_pred_col=args.y_pred_col,
        y_score_col=args.y_score_col,
        threshold=args.threshold,
        pos_label=args.pos_label,
    )
    cm = confusion(labels)
    metrics = compute_metrics(cm)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    csv_path = os.path.join(args.out_dir, "summary.csv")

    payload = {
        "counts": cm,
        "metrics": metrics,
        "total": len(labels),
        "pos_label": args.pos_label,
        "threshold": args.threshold,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k in ["accuracy", "precision", "recall", "f1", "specificity"]:
            writer.writerow([k, metrics[k]])

    print("[OK] Metrics guardadas en:", summary_path)
    print("[OK] CSV guardado en:", csv_path)
    print("[INFO] Confusion matrix:", cm)
    print("[INFO] Metrics:", metrics)


if __name__ == "__main__":
    main()

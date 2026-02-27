#!/usr/bin/env python3
"""
Create a 5-fold stratified split for training without modifying the original dataset.

Default input targets the current training CSV used to build "join" datasets:
  - input_csv: clipclap_train_labeled_join.csv
  - label_csv: clipclap_train_labeled_positive.csv

Outputs are written under ./5fold by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def read_csv(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def write_csv(path: Path, rows: Iterable[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def norm_text(value: str) -> str:
    return str(value or "").strip().lower()


def remap_image_path(path_value: str, project_root: Path) -> Tuple[str, bool]:
    """
    Remap legacy absolute paths to the current project root when possible.

    Returns:
      (new_path, changed)
    """
    original = str(path_value or "").strip().replace("\\", "/")
    key = "Sun_Multimodal/"
    idx = original.find(key)
    if idx >= 0:
        rel = original[idx:]
        candidate = (project_root / rel).resolve()
        if candidate.is_file():
            new_path = str(candidate)
            return new_path, new_path != original

    p = Path(original)
    if p.is_file():
        resolved = str(p.resolve())
        return resolved, resolved != original

    return original, False


def infer_frame_label(caption: str) -> str:
    text = norm_text(caption)
    if "no polyps" in text:
        return "no_polyp"
    if "hyperplastic" in text:
        return "hyperplastic"
    if "adenoma" in text:
        return "adenoma"
    if "polyp" in text:
        return "polyp"
    return "unknown"


def build_case_label_map(
    rows: List[dict],
    label_rows: List[dict],
    case_col: str,
    caption_col: str,
    label_col: str,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    case_to_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        case_to_rows[norm_text(row.get(case_col, ""))].append(row)

    label_from_csv: Dict[str, str] = {}
    for row in label_rows:
        case = norm_text(row.get(case_col, ""))
        label = norm_text(row.get(label_col, ""))
        if not case or not label:
            continue
        # If duplicates exist, keep the first non-empty label.
        label_from_csv.setdefault(case, label)

    case_to_label: Dict[str, str] = {}
    case_label_source: Dict[str, str] = {}
    case_sizes: Dict[str, int] = {}

    for case, case_rows in case_to_rows.items():
        case_sizes[case] = len(case_rows)
        if case in label_from_csv:
            case_to_label[case] = label_from_csv[case]
            case_label_source[case] = "label_csv"
            continue

        frame_labels = Counter(infer_frame_label(r.get(caption_col, "")) for r in case_rows)

        if frame_labels.get("hyperplastic", 0) > 0:
            case_to_label[case] = "hyperplastic"
        elif frame_labels.get("adenoma", 0) > 0:
            case_to_label[case] = "adenoma"
        elif frame_labels.get("no_polyp", 0) > 0 and len(frame_labels) == 1:
            case_to_label[case] = "no_polyp"
        elif frame_labels.get("polyp", 0) > 0:
            case_to_label[case] = "polyp"
        else:
            case_to_label[case] = "unknown"

        case_label_source[case] = "caption_inferred"

    return case_to_label, case_label_source, case_sizes


def assign_cases_to_folds(
    case_to_label: Dict[str, str],
    case_sizes: Dict[str, int],
    n_splits: int,
    seed: int,
) -> Dict[str, int]:
    rng = random.Random(seed)

    label_to_cases: Dict[str, List[str]] = defaultdict(list)
    for case, label in case_to_label.items():
        label_to_cases[label].append(case)

    primary_labels = [label for label, cases in label_to_cases.items() if len(cases) >= n_splits]
    rare_labels = [label for label, cases in label_to_cases.items() if len(cases) < n_splits]

    fold_case_counts = [0] * n_splits
    fold_frame_counts = [0] * n_splits
    fold_label_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(n_splits)]
    assignment: Dict[str, int] = {}

    for label in sorted(primary_labels):
        cases = label_to_cases[label][:]
        rng.shuffle(cases)
        cases.sort(key=lambda c: (-case_sizes[c], c))

        for case in cases:
            best_fold = min(
                range(n_splits),
                key=lambda f: (
                    fold_label_counts[f][label],
                    fold_frame_counts[f],
                    fold_case_counts[f],
                    f,
                ),
            )
            assignment[case] = best_fold
            fold_label_counts[best_fold][label] += 1
            fold_case_counts[best_fold] += 1
            fold_frame_counts[best_fold] += case_sizes[case]

    for label in sorted(rare_labels):
        cases = label_to_cases[label][:]
        rng.shuffle(cases)
        cases.sort(key=lambda c: (-case_sizes[c], c))

        for case in cases:
            best_fold = min(
                range(n_splits),
                key=lambda f: (fold_frame_counts[f], fold_case_counts[f], f),
            )
            assignment[case] = best_fold
            fold_label_counts[best_fold][label] += 1
            fold_case_counts[best_fold] += 1
            fold_frame_counts[best_fold] += case_sizes[case]

    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified 5-fold splits by case.")
    parser.add_argument(
        "--input_csv",
        default="experiments_colono/experiments_colono/clipclap_train_labeled_join.csv",
        help="Source training CSV to split (must include case column).",
    )
    parser.add_argument(
        "--label_csv",
        default="experiments_colono/experiments_colono/clipclap_train_labeled_positive.csv",
        help="CSV used to read case-level stratification label (optional).",
    )
    parser.add_argument("--output_dir", default="5fold", help="Directory where all 5-fold outputs are saved.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic tie-breaking.")
    parser.add_argument("--case_col", default="case", help="Case/patient column name.")
    parser.add_argument("--caption_col", default="caption", help="Caption column name.")
    parser.add_argument("--label_col", default="polyp_type", help="Label column in label_csv.")
    parser.add_argument("--image_path_col", default="image_path", help="Image path column name.")
    parser.add_argument(
        "--project_root",
        default=".",
        help="Project root used to rewrite legacy image paths (expects Sun_Multimodal/...).",
    )
    parser.add_argument(
        "--no_rewrite_image_paths",
        action="store_true",
        help="Disable automatic image path remapping in generated 5-fold CSVs.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    label_csv = Path(args.label_csv)
    output_dir = Path(args.output_dir)
    project_root = Path(args.project_root).resolve()

    if not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    rows, fieldnames = read_csv(input_csv)
    if args.case_col not in fieldnames:
        raise ValueError(f"Missing required column '{args.case_col}' in {input_csv}")
    if args.caption_col not in fieldnames:
        raise ValueError(f"Missing required column '{args.caption_col}' in {input_csv}")

    label_rows: List[dict] = []
    if label_csv.is_file():
        label_rows, _ = read_csv(label_csv)

    # Work on an in-memory copy only; original dataset is not modified.
    rows_out: List[dict] = []
    remapped_paths = 0
    existing_paths = 0
    missing_paths = 0
    rewrite_enabled = (not args.no_rewrite_image_paths) and (args.image_path_col in fieldnames)

    for row in rows:
        out_row = dict(row)
        if rewrite_enabled:
            new_path, changed = remap_image_path(out_row.get(args.image_path_col, ""), project_root)
            out_row[args.image_path_col] = new_path
            if changed:
                remapped_paths += 1

        if args.image_path_col in fieldnames:
            if Path(str(out_row.get(args.image_path_col, ""))).is_file():
                existing_paths += 1
            else:
                missing_paths += 1

        rows_out.append(out_row)

    rows = rows_out

    case_to_label, case_label_source, case_sizes = build_case_label_map(
        rows=rows,
        label_rows=label_rows,
        case_col=args.case_col,
        caption_col=args.caption_col,
        label_col=args.label_col,
    )
    case_to_fold = assign_cases_to_folds(
        case_to_label=case_to_label,
        case_sizes=case_sizes,
        n_splits=args.n_splits,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Case-level assignments
    case_assign_rows = []
    for case in sorted(case_to_fold):
        case_assign_rows.append(
            {
                "case": case,
                "fold": case_to_fold[case] + 1,
                "strat_label": case_to_label[case],
                "label_source": case_label_source[case],
                "num_rows": case_sizes[case],
            }
        )
    write_csv(
        output_dir / "case_assignments.csv",
        case_assign_rows,
        ["case", "fold", "strat_label", "label_source", "num_rows"],
    )

    # Row-level dataset with fold assignment
    row_level_rows = []
    row_level_fields = fieldnames + ["fold", "strat_label"]
    for row in rows:
        case = norm_text(row[args.case_col])
        enriched = dict(row)
        enriched["fold"] = case_to_fold[case] + 1
        enriched["strat_label"] = case_to_label[case]
        row_level_rows.append(enriched)
    write_csv(output_dir / "dataset_with_folds.csv", row_level_rows, row_level_fields)

    # Per-fold train/val CSVs + route manifest
    manifest_rows = []
    fold_summaries = []
    for fold_idx in range(args.n_splits):
        fold_name = f"fold_{fold_idx + 1}"
        fold_dir = output_dir / fold_name
        val_cases = {case for case, fidx in case_to_fold.items() if fidx == fold_idx}
        train_cases = set(case_to_fold.keys()) - val_cases

        train_rows = [r for r in rows if norm_text(r[args.case_col]) in train_cases]
        val_rows = [r for r in rows if norm_text(r[args.case_col]) in val_cases]

        train_path = fold_dir / "train.csv"
        val_path = fold_dir / "val.csv"
        write_csv(train_path, train_rows, fieldnames)
        write_csv(val_path, val_rows, fieldnames)

        train_case_rows = [r for r in case_assign_rows if int(r["fold"]) != (fold_idx + 1)]
        val_case_rows = [r for r in case_assign_rows if int(r["fold"]) == (fold_idx + 1)]
        write_csv(
            fold_dir / "train_cases.csv",
            train_case_rows,
            ["case", "fold", "strat_label", "label_source", "num_rows"],
        )
        write_csv(
            fold_dir / "val_cases.csv",
            val_case_rows,
            ["case", "fold", "strat_label", "label_source", "num_rows"],
        )

        manifest_rows.append(
            {
                "fold": fold_idx + 1,
                "train_csv": str(train_path),
                "val_csv": str(val_path),
                "train_pkl_suggested": str(fold_dir / "train.pkl"),
                "val_pkl_suggested": str(fold_dir / "val.pkl"),
                "checkpoint_prefix_suggested": f"colono_fold{fold_idx + 1}_prefix",
            }
        )

        val_label_dist = Counter(r["strat_label"] for r in val_case_rows)
        fold_summaries.append(
            {
                "fold": fold_idx + 1,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "val_case_label_distribution": dict(sorted(val_label_dist.items())),
            }
        )

    write_csv(
        output_dir / "fold_paths.csv",
        manifest_rows,
        [
            "fold",
            "train_csv",
            "val_csv",
            "train_pkl_suggested",
            "val_pkl_suggested",
            "checkpoint_prefix_suggested",
        ],
    )

    summary = {
        "input_csv": str(input_csv),
        "label_csv": str(label_csv) if label_csv.is_file() else None,
        "output_dir": str(output_dir),
        "project_root": str(project_root),
        "n_splits": args.n_splits,
        "seed": args.seed,
        "num_rows": len(rows),
        "num_cases": len(case_to_fold),
        "image_path_rewrite_enabled": rewrite_enabled,
        "image_paths_remapped": remapped_paths,
        "image_paths_existing": existing_paths,
        "image_paths_missing": missing_paths,
        "case_label_distribution": dict(sorted(Counter(case_to_label.values()).items())),
        "folds": fold_summaries,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("5-fold stratified split created successfully.")
    print(f"Input rows: {len(rows)} | Cases: {len(case_to_fold)}")
    print("Case label distribution:", dict(sorted(Counter(case_to_label.values()).items())))
    if args.image_path_col in fieldnames:
        print(
            "Image paths:",
            {"remapped": remapped_paths, "existing": existing_paths, "missing": missing_paths},
        )
    print(f"Outputs saved under: {output_dir}")


if __name__ == "__main__":
    main()

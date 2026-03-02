#!/usr/bin/env python3
"""
Real fold pipeline for CLIPCap.

What this script does for each task/fold:
1) Build train/val CSVs from the provided JSON fold definitions.
2) Convert CSVs to CLIP embeddings PKL (parse_colono_biomed.py by default).
3) Train real CLIPCap mapper (train.py) on train PKL.
4) Evaluate validation loss per epoch by loading each saved checkpoint.
5) Run caption generation on validation embeddings and compute class metrics.
6) Save plots (train vs val loss), CSVs, and JSON summaries under ./fold.

No execution is performed automatically unless you run this file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


CASE_RE = re.compile(r"^case\d+$")
NO_POLYP_PATTERNS = [
    re.compile(r"\bno\s+polyps?\b"),
    re.compile(r"\bno\s+visible\s+polyps?\b"),
    re.compile(r"\bno\s+evidence\s+of\s+polyps?\b"),
    re.compile(r"\bwithout\s+polyps?\b"),
]


def read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def norm_case(value: str) -> str:
    return str(value or "").strip().lower()


def case_sort_key(case_id: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(case_id))
    return (int(m.group(1)) if m else 10**9, str(case_id))


def flatten_val_cases(val_cases_obj) -> set:
    if isinstance(val_cases_obj, dict):
        out = set()
        for values in val_cases_obj.values():
            out.update(norm_case(v) for v in values)
        return out
    if isinstance(val_cases_obj, list):
        return set(norm_case(v) for v in val_cases_obj)
    raise ValueError(f"Unsupported val_cases type: {type(val_cases_obj)!r}")


def collect_case_ids(obj) -> set:
    found = set()
    if isinstance(obj, dict):
        for v in obj.values():
            found.update(collect_case_ids(v))
    elif isinstance(obj, list):
        for v in obj:
            found.update(collect_case_ids(v))
    elif isinstance(obj, str):
        c = norm_case(obj)
        if CASE_RE.match(c):
            found.add(c)
    return found


def remap_image_path(path_value: str, project_root: Path) -> Tuple[str, bool]:
    """Map legacy absolute paths to current workspace Sun_Multimodal paths."""
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


def load_json_config(path: Path) -> Tuple[str, dict]:
    raw = json.load(path.open("r", encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid JSON root in {path}")
    key = next(iter(raw.keys()))
    cfg = raw[key]
    if "folds" not in cfg:
        raise ValueError(f"Missing 'folds' in {path}")
    return key, cfg


def summarize_labels(rows: List[dict]) -> Dict[str, int]:
    return dict(sorted(Counter(r["label"] for r in rows).items()))


def run_cmd(cmd: List[str], cwd: Path, env: dict) -> None:
    print("[CMD]", " ".join(cmd))
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_loss_per_epoch(path: Path) -> Dict[int, float]:
    out = {}
    if not path.is_file():
        return out
    with path.open("r", newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            out[int(row["epoch"])] = float(row["mean_loss"])
    return out


def list_checkpoints(model_dir: Path, prefix: str) -> List[Tuple[int, Path]]:
    items = []
    pattern = re.compile(re.escape(prefix) + r"-(\d{3})\.pt$")
    for p in model_dir.glob(f"{prefix}-*.pt"):
        m = pattern.search(p.name)
        if not m:
            continue
        items.append((int(m.group(1)), p.resolve()))
    items.sort(key=lambda x: x[0])
    return items


def _mapping_type_enum(name: str):
    from train import MappingType

    mapping = {
        "mlp": MappingType.MLP,
        "transformer": MappingType.Transformer,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported mapping type: {name}")
    return mapping[name]


def eval_val_loss_for_checkpoints(
    val_pkl: Path,
    checkpoints: List[Tuple[int, Path]],
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    is_rn: bool,
    normalize_prefix: bool,
    eval_bs: int,
    use_only_prefix: bool,
) -> List[dict]:
    import torch
    from torch.nn import functional as nnf
    from torch.utils.data import DataLoader

    from train import ClipCaptionModel, ClipCaptionPrefix, ClipCocoDataset

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ClipCocoDataset(str(val_pkl), prefix_length, normalize_prefix=normalize_prefix)
    loader = DataLoader(dataset, batch_size=eval_bs, shuffle=False, drop_last=False)

    prefix_dim = 640 if is_rn else 512
    mapping_enum = _mapping_type_enum(mapping_type)

    out = []
    for epoch_idx, ckpt in checkpoints:
        if use_only_prefix:
            model = ClipCaptionPrefix(
                prefix_length,
                clip_length=prefix_length,
                prefix_size=prefix_dim,
                num_layers=num_layers,
                mapping_type=mapping_enum,
            )
        else:
            model = ClipCaptionModel(
                prefix_length,
                clip_length=prefix_length,
                prefix_size=prefix_dim,
                num_layers=num_layers,
                mapping_type=mapping_enum,
            )

        state = torch.load(str(ckpt), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        model = model.to(device).eval()

        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for tokens, mask, prefix in loader:
                tokens = tokens.to(device)
                mask = mask.to(device)
                prefix = prefix.to(device, dtype=torch.float32)

                outputs = model(tokens, prefix, mask)
                logits = outputs.logits[:, dataset.prefix_length - 1 : -1]
                loss = nnf.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tokens.flatten(),
                    ignore_index=0,
                    reduction="mean",
                )
                bs = int(tokens.shape[0])
                total_loss += float(loss.item()) * bs
                total_count += bs

        val_loss = total_loss / max(1, total_count)
        out.append(
            {
                "epoch": epoch_idx,
                "checkpoint": str(ckpt),
                "val_loss": val_loss,
            }
        )

    return out


def predict_captions_from_val_pkl(
    val_pkl: Path,
    checkpoint: Path,
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    is_rn: bool,
    use_only_prefix: bool,
    use_beam_search: bool,
    entry_length: int,
    top_p: float,
    temperature: float,
    stop_token: str,
    max_samples: int,
) -> List[str]:
    import torch
    from transformers import GPT2Tokenizer

    from predict import generate2, generate_beam
    from train import ClipCaptionModel, ClipCaptionPrefix

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mapping_enum = _mapping_type_enum(mapping_type)
    prefix_dim = 640 if is_rn else 512

    if use_only_prefix:
        model = ClipCaptionPrefix(
            prefix_length,
            clip_length=prefix_length,
            prefix_size=prefix_dim,
            num_layers=num_layers,
            mapping_type=mapping_enum,
        )
    else:
        model = ClipCaptionModel(
            prefix_length,
            clip_length=prefix_length,
            prefix_size=prefix_dim,
            num_layers=num_layers,
            mapping_type=mapping_enum,
        )

    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    with val_pkl.open("rb") as f:
        all_data = pickle.load(f)
    embeddings = all_data["clip_embedding"]

    n = int(embeddings.shape[0])
    if max_samples > 0:
        n = min(n, max_samples)

    preds = []
    # `predict.generate2` expects `tokens` to be a Tensor. When using `embed`
    # only, it can crash with NoneType in torch.cat, so seed with one token.
    start_token_id = tokenizer.eos_token_id
    if start_token_id is None:
        start_token_id = 0
    start_tokens = torch.tensor([[int(start_token_id)]], device=device, dtype=torch.long)

    with torch.no_grad():
        for i in range(n):
            emb = embeddings[i : i + 1].to(device).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            prefix_embed = model.clip_project(emb).view(1, prefix_length, -1)

            if use_beam_search:
                caption = generate_beam(
                    model=model,
                    tokenizer=tokenizer,
                    embed=prefix_embed,
                    entry_length=entry_length,
                )[0]
            else:
                caption = generate2(
                    model,
                    tokenizer,
                    tokens=start_tokens.clone(),
                    embed=prefix_embed,
                    entry_count=1,
                    entry_length=entry_length,
                    top_p=top_p,
                    temperature=temperature,
                    stop_token=stop_token,
                )
            preds.append(str(caption))

    return preds


def infer_pred_label(task_name: str, caption: str) -> str:
    text = str(caption or "").lower()

    if task_name == "positive_vs_negative":
        for pat in NO_POLYP_PATTERNS:
            if pat.search(text):
                return "no_polyp"
        return "polyp"

    if task_name == "hyper_vs_adeno":
        if "hyperplastic" in text:
            return "hyperplastic"
        if "adenoma" in text:
            return "adenoma"
        return "unknown"

    raise ValueError(f"Unknown task_name: {task_name}")


def compute_metrics(true_labels: List[str], pred_labels: List[str], class_names: List[str]) -> dict:
    classes = list(class_names)
    pred_extra = sorted(set(p for p in pred_labels if p not in classes))
    pred_axis = classes + pred_extra

    idx_true = {c: i for i, c in enumerate(classes)}
    idx_pred = {c: i for i, c in enumerate(pred_axis)}

    cm = [[0 for _ in pred_axis] for _ in classes]
    total = min(len(true_labels), len(pred_labels))
    correct = 0

    for t, p in zip(true_labels[:total], pred_labels[:total]):
        if t in idx_true and p in idx_pred:
            cm[idx_true[t]][idx_pred[p]] += 1
        if t == p:
            correct += 1

    acc = correct / total if total else 0.0

    per_class = {}
    f1_values = []
    for c in classes:
        i = idx_true[c]
        tp = cm[i][idx_pred[c]] if c in idx_pred else 0
        fp = sum(cm[r][idx_pred[c]] for r in range(len(classes)) if r != i) if c in idx_pred else 0
        fn = sum(cm[i][j] for j in range(len(pred_axis)) if j != idx_pred.get(c, -1))
        tn = total - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1_values.append(f1)

        per_class[c] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
        }

    return {
        "num_samples": total,
        "accuracy": acc,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "true_labels": classes,
        "pred_labels": pred_axis,
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def write_loss_plot_svg(
    path: Path,
    title: str,
    epochs: List[int],
    train_loss: List[float],
    val_loss: List[float],
) -> None:
    width, height = 980, 540
    ml, mr, mt, mb = 80, 40, 70, 90
    pw, ph = width - ml - mr, height - mt - mb

    vals = [v for v in (train_loss + val_loss) if v is not None]
    y_min = min(vals) if vals else 0.0
    y_max = max(vals) if vals else 1.0
    if abs(y_max - y_min) < 1e-12:
        y_min -= 1.0
        y_max += 1.0

    def x_to_px(x: int) -> float:
        if len(epochs) <= 1:
            return ml + pw / 2
        x0, x1 = min(epochs), max(epochs)
        return ml + ((x - x0) / (x1 - x0)) * pw

    def y_to_px(y: float) -> float:
        return mt + (1.0 - (y - y_min) / (y_max - y_min)) * ph

    def poly(values: List[float]) -> str:
        pts = []
        for i, v in enumerate(values):
            pts.append(f"{x_to_px(epochs[i]):.2f},{y_to_px(v):.2f}")
        return " ".join(pts)

    train_line = poly(train_loss)
    val_line = poly(val_loss)

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f8fafc"/>')
    lines.append(f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="22" fill="#0f172a">{title}</text>')
    lines.append(f'<text x="{width/2:.1f}" y="{height-16}" text-anchor="middle" font-size="16" fill="#334155">Epoch</text>')
    lines.append(
        f'<text transform="translate(24,{height/2:.1f}) rotate(-90)" text-anchor="middle" '
        f'font-size="16" fill="#334155">Loss</text>'
    )
    lines.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" fill="#ffffff" stroke="#cbd5e1"/>')

    for i in range(6):
        yv = y_min + (y_max - y_min) * (i / 5)
        yp = y_to_px(yv)
        lines.append(f'<line x1="{ml}" y1="{yp:.2f}" x2="{ml+pw}" y2="{yp:.2f}" stroke="#e2e8f0"/>')
        lines.append(f'<text x="{ml-8}" y="{yp+5:.2f}" text-anchor="end" font-size="12" fill="#475569">{yv:.4f}</text>')

    x_ticks = min(10, len(epochs))
    if x_ticks <= 1:
        tick_vals = [epochs[0]] if epochs else []
    else:
        tick_vals = [epochs[int(i * (len(epochs) - 1) / (x_ticks - 1))] for i in range(x_ticks)]

    for xv in tick_vals:
        xp = x_to_px(xv)
        lines.append(f'<line x1="{xp:.2f}" y1="{mt}" x2="{xp:.2f}" y2="{mt+ph}" stroke="#eef2f7"/>')
        lines.append(f'<text x="{xp:.2f}" y="{mt+ph+22}" text-anchor="middle" font-size="12" fill="#475569">{xv}</text>')

    lines.append(f'<polyline fill="none" stroke="#0ea5e9" stroke-width="3" points="{train_line}"/>')
    lines.append(f'<polyline fill="none" stroke="#ef4444" stroke-width="3" points="{val_line}"/>')

    lx, ly = ml + 10, mt + 16
    lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+26}" y2="{ly}" stroke="#0ea5e9" stroke-width="3"/>')
    lines.append(f'<text x="{lx+34}" y="{ly+4}" font-size="13" fill="#0f172a">train</text>')
    lines.append(f'<line x1="{lx+110}" y1="{ly}" x2="{lx+136}" y2="{ly}" stroke="#ef4444" stroke-width="3"/>')
    lines.append(f'<text x="{lx+144}" y="{ly+4}" font-size="13" fill="#0f172a">validation</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_hyper_vs_adeno_dataset(
    positive_rows: List[dict],
    allowed_cases: set,
    project_root: Path,
) -> Tuple[List[dict], dict]:
    rows = []
    remapped = 0
    missing = 0

    for r in positive_rows:
        case = norm_case(r.get("case", ""))
        if case not in allowed_cases:
            continue
        label = str(r.get("polyp_type", "")).strip().lower()
        if label not in {"adenoma", "hyperplastic"}:
            continue

        img_path, changed = remap_image_path(r.get("image_path", ""), project_root)
        if changed:
            remapped += 1
        if not Path(img_path).is_file():
            missing += 1
            continue

        rows.append(
            {
                "image_path": img_path,
                "case": case,
                "caption": str(r.get("caption", "")).strip(),
                "label": label,
                "source_csv": "clipclap_train_labeled_positive.csv",
            }
        )

    stats = {
        "rows": len(rows),
        "remapped_paths": remapped,
        "missing_paths_filtered": missing,
    }
    return rows, stats


def build_positive_vs_negative_dataset(
    positive_rows: List[dict],
    negative_rows: List[dict],
    polyp_cases: set,
    no_polyp_cases: set,
    project_root: Path,
) -> Tuple[List[dict], dict]:
    rows = []
    remapped = 0
    missing = 0

    for r in positive_rows:
        case = norm_case(r.get("case", ""))
        if case not in polyp_cases:
            continue
        img_path, changed = remap_image_path(r.get("image_path", ""), project_root)
        if changed:
            remapped += 1
        if not Path(img_path).is_file():
            missing += 1
            continue
        rows.append(
            {
                "image_path": img_path,
                "case": case,
                "caption": str(r.get("caption", "")).strip(),
                "label": "polyp",
                "source_csv": "clipclap_train_labeled_positive.csv",
            }
        )

    for r in negative_rows:
        case = norm_case(r.get("case", ""))
        if case not in no_polyp_cases:
            continue
        img_path, changed = remap_image_path(r.get("image_path", ""), project_root)
        if changed:
            remapped += 1
        if not Path(img_path).is_file():
            missing += 1
            continue
        rows.append(
            {
                "image_path": img_path,
                "case": case,
                "caption": str(r.get("caption", "")).strip(),
                "label": "no_polyp",
                "source_csv": "clipclap_train_labeled_negative.csv",
            }
        )

    stats = {
        "rows": len(rows),
        "remapped_paths": remapped,
        "missing_paths_filtered": missing,
    }
    return rows, stats


def run_task_pipeline(
    task_name: str,
    dataset_rows: List[dict],
    all_cases: set,
    folds_cfg: Dict[str, dict],
    out_root: Path,
    project_root: Path,
    env: dict,
    py_exec: str,
    args,
) -> dict:
    task_dir = out_root / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = sorted(dataset_rows, key=lambda r: (case_sort_key(r["case"]), r["image_path"]))
    write_csv(
        task_dir / "dataset.csv",
        dataset_rows,
        ["image_path", "case", "caption", "label", "source_csv"],
    )

    fold_items = sorted(folds_cfg.items(), key=lambda kv: case_sort_key(kv[0].replace("fold_", "case")))

    fold_results = []
    mean_curve_train = {}
    mean_curve_val = {}

    for idx, (fold_name, fold_spec) in enumerate(fold_items, start=1):
        fold_dir = task_dir / "folds" / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)

        val_cases = flatten_val_cases(fold_spec.get("val_cases", {}))
        train_cases = all_cases - val_cases
        if not train_cases:
            train_cases = set(norm_case(c) for c in fold_spec.get("train_cases", []))

        train_rows = [r for r in dataset_rows if r["case"] in train_cases]
        val_rows = [r for r in dataset_rows if r["case"] in val_cases]

        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        write_csv(train_csv, train_rows, ["image_path", "case", "caption", "label", "source_csv"])
        write_csv(val_csv, val_rows, ["image_path", "case", "caption", "label", "source_csv"])

        data_dir = fold_dir / "data"
        model_dir = fold_dir / "train"
        infer_dir = fold_dir / "inference"
        data_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        infer_dir.mkdir(parents=True, exist_ok=True)

        train_pkl = data_dir / "train.pkl"
        val_pkl = data_dir / "val.pkl"

        parser_script = project_root / args.parser_script
        train_script = project_root / "train.py"

        # Parse train/val CSV to CLIP embeddings
        if args.parser_type == "biomed":
            run_cmd(
                [
                    py_exec,
                    str(parser_script),
                    "--csv_path",
                    str(train_csv),
                    "--images_root",
                    str(project_root),
                    "--out_path",
                    str(train_pkl),
                    "--weights_path",
                    str(project_root / args.weights_path),
                    "--model_id",
                    args.model_id,
                    "--batch_size",
                    str(args.parse_batch_size),
                ],
                cwd=project_root,
                env=env,
            )
            run_cmd(
                [
                    py_exec,
                    str(parser_script),
                    "--csv_path",
                    str(val_csv),
                    "--images_root",
                    str(project_root),
                    "--out_path",
                    str(val_pkl),
                    "--weights_path",
                    str(project_root / args.weights_path),
                    "--model_id",
                    args.model_id,
                    "--batch_size",
                    str(args.parse_batch_size),
                ],
                cwd=project_root,
                env=env,
            )
        else:
            run_cmd(
                [
                    py_exec,
                    str(parser_script),
                    "--csv_path",
                    str(train_csv),
                    "--images_root",
                    str(project_root),
                    "--out_path",
                    str(train_pkl),
                    "--clip_model_type",
                    args.openai_clip_model,
                ],
                cwd=project_root,
                env=env,
            )
            run_cmd(
                [
                    py_exec,
                    str(parser_script),
                    "--csv_path",
                    str(val_csv),
                    "--images_root",
                    str(project_root),
                    "--out_path",
                    str(val_pkl),
                    "--clip_model_type",
                    args.openai_clip_model,
                ],
                cwd=project_root,
                env=env,
            )

        prefix = f"{task_name}_{fold_name}"

        # Train real CLIPCap mapper
        train_cmd = [
            py_exec,
            str(train_script),
            "--data",
            str(train_pkl),
            "--out_dir",
            str(model_dir),
            "--prefix",
            prefix,
            "--epochs",
            str(args.epochs),
            "--save_every",
            "1",
            "--prefix_length",
            str(args.prefix_length),
            "--prefix_length_clip",
            str(args.prefix_length_clip),
            "--bs",
            str(args.train_bs),
            "--mapping_type",
            args.mapping_type,
            "--num_layers",
            str(args.num_layers),
        ]
        if args.is_rn:
            train_cmd.append("--is_rn")
        if args.only_prefix:
            train_cmd.append("--only_prefix")
        if args.normalize_prefix:
            train_cmd.append("--normalize_prefix")

        run_cmd(train_cmd, cwd=project_root, env=env)

        # Build train-vs-val loss curves by evaluating val loss per epoch checkpoint
        ckpts = list_checkpoints(model_dir, prefix)
        if not ckpts:
            raise RuntimeError(f"No checkpoints found for {task_name} {fold_name} in {model_dir}")

        val_curve = eval_val_loss_for_checkpoints(
            val_pkl=val_pkl,
            checkpoints=ckpts,
            prefix_length=args.prefix_length,
            mapping_type=args.mapping_type,
            num_layers=args.num_layers,
            is_rn=args.is_rn,
            normalize_prefix=args.normalize_prefix,
            eval_bs=args.eval_bs,
            use_only_prefix=args.only_prefix,
        )

        write_csv(
            fold_dir / "val_loss_per_epoch.csv",
            val_curve,
            ["epoch", "checkpoint", "val_loss"],
        )

        train_curve_map = parse_loss_per_epoch(model_dir / f"{prefix}_loss_per_epoch.csv")
        val_curve_map = {int(r["epoch"]): float(r["val_loss"]) for r in val_curve}
        common_epochs = sorted(set(train_curve_map.keys()) & set(val_curve_map.keys()))
        if not common_epochs:
            common_epochs = sorted(set(train_curve_map.keys()) or set(val_curve_map.keys()))

        train_losses = [float(train_curve_map[e]) for e in common_epochs]
        val_losses = [float(val_curve_map[e]) for e in common_epochs]

        write_loss_plot_svg(
            path=fold_dir / "train_vs_val_loss.svg",
            title=f"{task_name} - {fold_name} - Train vs Validation Loss",
            epochs=common_epochs,
            train_loss=train_losses,
            val_loss=val_losses,
        )

        # Pick best checkpoint by val loss
        best_row = min(val_curve, key=lambda r: float(r["val_loss"]))
        best_epoch = int(best_row["epoch"])
        best_ckpt = Path(best_row["checkpoint"])

        # Validation caption-generation test and class metrics
        val_predictions_rows = []
        class_metrics = None
        if not args.skip_inference:
            pred_captions = predict_captions_from_val_pkl(
                val_pkl=val_pkl,
                checkpoint=best_ckpt,
                prefix_length=args.prefix_length,
                mapping_type=args.mapping_type,
                num_layers=args.num_layers,
                is_rn=args.is_rn,
                use_only_prefix=args.only_prefix,
                use_beam_search=args.beam_search,
                entry_length=args.entry_length,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_token=args.stop_token,
                max_samples=args.max_val_samples,
            )

            n = min(len(val_rows), len(pred_captions))
            true_labels = []
            pred_labels = []
            for i in range(n):
                gt = val_rows[i]["label"]
                pc = pred_captions[i]
                pl = infer_pred_label(task_name, pc)
                true_labels.append(gt)
                pred_labels.append(pl)
                val_predictions_rows.append(
                    {
                        "image_path": val_rows[i]["image_path"],
                        "case": val_rows[i]["case"],
                        "caption_gt": val_rows[i]["caption"],
                        "label_true": gt,
                        "caption_pred": pc,
                        "label_pred": pl,
                    }
                )

            class_names = sorted(set(r["label"] for r in dataset_rows))
            class_metrics = compute_metrics(true_labels, pred_labels, class_names)

            write_csv(
                infer_dir / "val_predictions.csv",
                val_predictions_rows,
                ["image_path", "case", "caption_gt", "label_true", "caption_pred", "label_pred"],
            )

        metrics_payload = {
            "fold": fold_name,
            "num_train_rows": len(train_rows),
            "num_val_rows": len(val_rows),
            "train_label_distribution": summarize_labels(train_rows),
            "val_label_distribution": summarize_labels(val_rows),
            "train_cases": sorted(train_cases, key=case_sort_key),
            "val_cases": sorted(val_cases, key=case_sort_key),
            "artifacts": {
                "train_csv": str(train_csv),
                "val_csv": str(val_csv),
                "train_pkl": str(train_pkl),
                "val_pkl": str(val_pkl),
                "model_dir": str(model_dir),
                "best_checkpoint": str(best_ckpt),
                "train_loss_csv": str(model_dir / f"{prefix}_loss_per_epoch.csv"),
                "val_loss_csv": str(fold_dir / "val_loss_per_epoch.csv"),
                "loss_plot_svg": str(fold_dir / "train_vs_val_loss.svg"),
            },
            "training": {
                "epochs": args.epochs,
                "train_bs": args.train_bs,
                "eval_bs": args.eval_bs,
                "prefix_length": args.prefix_length,
                "prefix_length_clip": args.prefix_length_clip,
                "mapping_type": args.mapping_type,
                "num_layers": args.num_layers,
                "is_rn": args.is_rn,
                "only_prefix": args.only_prefix,
                "normalize_prefix": args.normalize_prefix,
                "gpu_visible": env.get("CUDA_VISIBLE_DEVICES", ""),
            },
            "loss": {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_row["val_loss"]),
                "train_loss_at_best_epoch": float(train_curve_map.get(best_epoch, math.nan)),
                "epochs": common_epochs,
                "train_loss": train_losses,
                "val_loss": val_losses,
            },
            "classification": class_metrics,
        }
        write_json(fold_dir / "metrics.json", metrics_payload)

        fold_results.append(metrics_payload)

        for e, tr, vl in zip(common_epochs, train_losses, val_losses):
            mean_curve_train.setdefault(e, []).append(tr)
            mean_curve_val.setdefault(e, []).append(vl)

    # Aggregate task results
    best_val_losses = [f["loss"]["best_val_loss"] for f in fold_results]
    agg = {
        "num_folds": len(fold_results),
        "best_val_loss_mean": statistics.mean(best_val_losses) if best_val_losses else None,
        "best_val_loss_std": statistics.pstdev(best_val_losses) if len(best_val_losses) > 1 else 0.0,
    }

    cls_accs = []
    cls_f1 = []
    for f in fold_results:
        cm = f.get("classification")
        if cm:
            cls_accs.append(cm["accuracy"])
            cls_f1.append(cm["macro_f1"])
    if cls_accs:
        agg["val_classification_accuracy_mean"] = statistics.mean(cls_accs)
        agg["val_classification_accuracy_std"] = statistics.pstdev(cls_accs) if len(cls_accs) > 1 else 0.0
        agg["val_classification_macro_f1_mean"] = statistics.mean(cls_f1)
        agg["val_classification_macro_f1_std"] = statistics.pstdev(cls_f1) if len(cls_f1) > 1 else 0.0

    # Mean loss plot across folds
    mean_epochs = sorted(set(mean_curve_train.keys()) & set(mean_curve_val.keys()))
    if mean_epochs:
        mean_train = [statistics.mean(mean_curve_train[e]) for e in mean_epochs]
        mean_val = [statistics.mean(mean_curve_val[e]) for e in mean_epochs]
        write_loss_plot_svg(
            path=task_dir / "plots" / "mean_train_vs_val_loss.svg",
            title=f"{task_name} - Mean Train vs Validation Loss",
            epochs=mean_epochs,
            train_loss=mean_train,
            val_loss=mean_val,
        )

    result_payload = {
        "task_name": task_name,
        "num_rows": len(dataset_rows),
        "num_cases": len(all_cases),
        "label_distribution": summarize_labels(dataset_rows),
        "folds": fold_results,
        "aggregate": agg,
    }
    write_json(task_dir / "results.json", result_payload)
    return result_payload


def build_readme(path: Path) -> None:
    text = """# Real CLIPCap Fold Outputs

This directory contains real CLIPCap fold training artifacts.

## For each task
- dataset.csv
- folds/fold_*/train.csv and val.csv
- folds/fold_*/data/train.pkl and val.pkl
- folds/fold_*/train/* (checkpoints, train loss csv/svg)
- folds/fold_*/val_loss_per_epoch.csv
- folds/fold_*/train_vs_val_loss.svg
- folds/fold_*/inference/val_predictions.csv (unless --skip_inference)
- folds/fold_*/metrics.json
- results.json

## Global
- summary.json
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real CLIPCap fold training pipeline")

    parser.add_argument("--hyper_json", default="folds/hyper_vs_adeno.json")
    parser.add_argument("--posneg_json", default="folds/negative_positive.json")
    parser.add_argument(
        "--positive_csv",
        default="experiments_colono/experiments_colono/clipclap_train_labeled_positive.csv",
    )
    parser.add_argument(
        "--negative_csv",
        default="experiments_colono/experiments_colono/clipclap_train_labeled_negative.csv",
    )
    parser.add_argument("--out_dir", default="fold/fold-5-epoch")
    parser.add_argument("--project_root", default=".")

    # Runtime/GPU
    parser.add_argument("--gpu", default="1", help="CUDA_VISIBLE_DEVICES value (e.g. 0,1)")
    parser.add_argument("--python_exec", default=sys.executable)

    # Parser selection
    parser.add_argument(
        "--parser_type",
        default="biomed",
        choices=["biomed", "openai"],
        help="biomed -> parse_colono_biomed.py | openai -> parse_colono.py",
    )
    parser.add_argument("--parser_script", default="parse_colono_biomed.py")
    parser.add_argument("--parse_batch_size", type=int, default=64)

    # BioMed parser args
    parser.add_argument("--weights_path", default="clip_weights/biomedclip_weights.pt")
    parser.add_argument(
        "--model_id",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )

    # OpenAI parser args
    parser.add_argument("--openai_clip_model", default="ViT-B/32")

    # Train args
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_bs", type=int, default=4)
    parser.add_argument("--eval_bs", type=int, default=4)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--mapping_type", default="transformer", choices=["mlp", "transformer"])
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--is_rn", action="store_true")
    parser.add_argument("--only_prefix", action="store_true", default=True)
    parser.add_argument("--normalize_prefix", action="store_true", default=True)

    # Inference/test args
    parser.add_argument("--skip_inference", action="store_true")
    parser.add_argument("--beam_search", action="store_true")
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_token", default=".")
    parser.add_argument("--max_val_samples", type=int, default=0, help="0 means all val samples")

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Ensure local project modules (train.py, predict.py, etc.) are importable
    # even when this script is launched as `python folds/stratifiedfolds.py`.
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    # Runtime env for subprocesses (train/parse): force selected GPU
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.parser_type == "openai" and args.parser_script == "parse_colono_biomed.py":
        args.parser_script = "parse_colono.py"

    hyper_key, hyper_cfg = load_json_config(project_root / args.hyper_json)
    posneg_key, posneg_cfg = load_json_config(project_root / args.posneg_json)

    positive_rows = read_csv(project_root / args.positive_csv)
    negative_rows = read_csv(project_root / args.negative_csv)

    # Task 1: hyperplastic vs adenoma
    hyper_cases = set(norm_case(c) for c in hyper_cfg.get("all_cases", []))
    if not hyper_cases:
        hyper_cases = collect_case_ids(hyper_cfg)

    hyper_rows, hyper_data_stats = build_hyper_vs_adeno_dataset(
        positive_rows=positive_rows,
        allowed_cases=hyper_cases,
        project_root=project_root,
    )

    hyper_results = run_task_pipeline(
        task_name="hyper_vs_adeno",
        dataset_rows=hyper_rows,
        all_cases=hyper_cases,
        folds_cfg=hyper_cfg["folds"],
        out_root=out_root,
        project_root=project_root,
        env=env,
        py_exec=args.python_exec,
        args=args,
    )

    # Task 2: positive vs negative
    no_polyp_cases = set(norm_case(c) for c in posneg_cfg.get("no_polyp_cases", []))
    polyp_cases = set(norm_case(c) for c in posneg_cfg.get("polyp_cases_fold_1", []))
    polyp_cases.update(norm_case(c) for c in posneg_cfg.get("polyp_cases_fold_2", []))
    if not polyp_cases:
        for fold_spec in posneg_cfg["folds"].values():
            val_obj = fold_spec.get("val_cases", {})
            if isinstance(val_obj, dict):
                polyp_cases.update(norm_case(c) for c in val_obj.get("polyp", []))

    posneg_all_cases = polyp_cases | no_polyp_cases
    if not posneg_all_cases:
        posneg_all_cases = collect_case_ids(posneg_cfg)

    posneg_rows, posneg_data_stats = build_positive_vs_negative_dataset(
        positive_rows=positive_rows,
        negative_rows=negative_rows,
        polyp_cases=polyp_cases,
        no_polyp_cases=no_polyp_cases,
        project_root=project_root,
    )

    posneg_results = run_task_pipeline(
        task_name="positive_vs_negative",
        dataset_rows=posneg_rows,
        all_cases=posneg_all_cases,
        folds_cfg=posneg_cfg["folds"],
        out_root=out_root,
        project_root=project_root,
        env=env,
        py_exec=args.python_exec,
        args=args,
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "hyper_json": str(project_root / args.hyper_json),
            "posneg_json": str(project_root / args.posneg_json),
            "positive_csv": str(project_root / args.positive_csv),
            "negative_csv": str(project_root / args.negative_csv),
        },
        "runtime": {
            "python_exec": args.python_exec,
            "gpu": str(args.gpu),
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
            "parser_type": args.parser_type,
            "parser_script": args.parser_script,
        },
        "training": {
            "epochs": args.epochs,
            "train_bs": args.train_bs,
            "eval_bs": args.eval_bs,
            "prefix_length": args.prefix_length,
            "prefix_length_clip": args.prefix_length_clip,
            "mapping_type": args.mapping_type,
            "num_layers": args.num_layers,
            "is_rn": args.is_rn,
            "only_prefix": args.only_prefix,
            "normalize_prefix": args.normalize_prefix,
            "skip_inference": args.skip_inference,
            "beam_search": args.beam_search,
        },
        "dataset_stats": {
            "hyper_vs_adeno": hyper_data_stats,
            "positive_vs_negative": posneg_data_stats,
        },
        "tasks": {
            hyper_key: hyper_results,
            posneg_key: posneg_results,
        },
    }

    write_json(out_root / "summary.json", summary)
    build_readme(out_root / "README.md")

    print("Real CLIPCap fold pipelines completed.")
    print(f"Outputs saved to: {out_root}")
    print("Task aggregate summaries:")
    print(" - hyper_vs_adeno:", hyper_results.get("aggregate", {}))
    print(" - positive_vs_negative:", posneg_results.get("aggregate", {}))


if __name__ == "__main__":
    main()

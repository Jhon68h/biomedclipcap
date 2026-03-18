#!/usr/bin/env python3
"""Re-validacion 2-fold: val loss por epoca + tokens GPT.

Usa la estructura y artefactos creados por scripts/2fold_models.py.
Para cada modelo/fold seleccionado:
1) Calcula val loss por epoca (default: epochs 1..14).
2) Calcula val loss por batch para el checkpoint final.
3) Regenera captions de validacion y exporta token IDs exactos del GPT.
4) Sobrescribe los outputs existentes con los mismos nombres/rutas.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = "fold/2fold"

MODEL_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "biomedclip": {
        "weights_path": "clip_weights/biomedclip_weights.pt",
        "clip_model_type": None,
        "test_encoder": "biomedclip",
        "biomedclip_model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "openai_clip_name": None,
    },
    "vit": {
        "weights_path": "clip_weights/ViT-B-16.pt",
        "clip_model_type": "ViT-B/32",
        "test_encoder": "vit",
        "openai_clip_name": "ViT-B/32",
    },
    "resnet101": {
        "weights_path": "clip_weights/RN101.pt",
        "clip_model_type": "RN101",
        "test_encoder": "rn101",
        "openai_clip_name": "RN101",
    },
}

MODEL_ALIAS = {
    "biomedclip": "biomedclip",
    "vit": "vit",
    "resnet101": "resnet101",
    "resnet": "resnet101",
}

MODEL_OUTPUT_DIR = {
    "biomedclip": "biomedclip",
    "vit": "vit",
    "resnet101": "resnet",
}

MANIFEST_COLUMNS = [
    "sample_id",
    "label",
    "case",
    "image_path",
    "resolved_image_path",
    "linked_image_path",
    "caption_gt",
]

VAL_LOSS_COLUMNS = [
    "batch",
    "num_samples",
    "val_loss",
]

VAL_LOSS_PER_EPOCH_COLUMNS = [
    "epoch",
    "checkpoint",
    "val_loss",
    "val_batches",
    "val_samples",
]

VAL_PRED_COLUMNS = [
    "fold",
    "sample_id",
    "label",
    "case",
    "image_path",
    "caption_gt",
    "generated_caption",
    "linked_image_path",
]

VAL_PRED_WITH_TOKENS_COLUMNS = VAL_PRED_COLUMNS + [
    "generated_token_ids",
    "generated_token_count",
]

RAW_PRED_WITH_TOKENS_COLUMNS = [
    "image_path",
    "generated_caption",
    "generated_token_ids",
    "generated_token_count",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revalida modelos 2-fold ya entrenados: val loss por epoca y "
            "tokens GPT de salida en validacion."
        )
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=sorted(list(MODEL_ALIAS.keys()) + ["all"]),
        help="Modelo a validar: biomedclip, vit, resnet/resnet101 o all.",
    )
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--end_epoch", type=int, default=14)
    parser.add_argument(
        "--allow_missing_checkpoints",
        action="store_true",
        help="Si falta un checkpoint de una epoca, la omite en vez de fallar.",
    )
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES (ej: 0,1,3).")

    parser.add_argument("--beam_search", dest="beam_search", action="store_true", default=True)
    parser.add_argument("--no_beam_search", dest="beam_search", action="store_false")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_token", type=str, default=".")
    parser.add_argument("--tokenizer_name", default="gpt2")

    return parser.parse_args()


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_artifact_path(path_str: Optional[str], fallback: Path) -> Path:
    if path_str:
        return resolve_repo_path(str(path_str))
    return fallback.resolve()


def to_repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON invalido (se esperaba objeto): {path}")
    return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def select_models(model_arg: str) -> List[str]:
    if model_arg == "all":
        return ["biomedclip", "vit", "resnet101"]
    return [MODEL_ALIAS[model_arg]]


def normalize_mapping_type(value: Any) -> str:
    text = str(value or "transformer").lower()
    return "transformer" if "transformer" in text else "mlp"


def resolve_image_path(raw_path: str) -> Path:
    clean = (raw_path or "").strip().replace("\\", "/").strip('"').strip("'")
    if not clean:
        return REPO_ROOT / "__missing__"
    path = Path(clean)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def create_val_image_manifest(
    val_rows: Sequence[Dict[str, str]],
    val_images_dir: Path,
) -> Tuple[List[Dict[str, str]], int]:
    val_images_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []
    missing_count = 0

    for row in val_rows:
        src_path = resolve_image_path(str(row.get("image_path", "")))
        if not src_path.exists():
            missing_count += 1
            continue

        dst_name = f"{int(row['sample_id']):06d}_{src_path.name}"
        dst_path = (val_images_dir / dst_name).resolve()

        try:
            dst_path.symlink_to(src_path)
        except Exception:
            dst_path.write_bytes(src_path.read_bytes())

        manifest.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "label": str(row.get("label", "")),
                "case": str(row.get("case", "")),
                "image_path": str(row.get("image_path", "")),
                "resolved_image_path": src_path.as_posix(),
                "linked_image_path": dst_path.as_posix(),
                "caption_gt": str(row.get("caption", "")),
            }
        )

    return manifest, missing_count


def iter_images(root_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for dirpath, _, filenames in os.walk(str(root_dir)):
        for filename in filenames:
            if Path(filename).suffix.lower() in exts:
                yield (Path(dirpath) / filename).resolve()


def generate_beam_with_ids(
    model,
    tokenizer: GPT2Tokenizer,
    beam_size: int = 5,
    prompt: Optional[str] = None,
    embed: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = ".",
) -> Tuple[str, List[int]]:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
                generated = model.gpt.transformer.wte(tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average, next_tokens = (scores_sum / seq_lengths[:, None]).view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = (next_tokens % scores_sum.shape[1]).unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    if tokens is None:
        return "", []

    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    order = scores.argsort(descending=True)
    best_idx = int(order[0].item())
    best_len = int(seq_lengths[best_idx].item())
    best_ids = [int(x) for x in output_list[best_idx][:best_len]]
    best_text = tokenizer.decode(best_ids)
    return best_text, best_ids


def generate_top_p_with_ids(
    model,
    tokenizer: GPT2Tokenizer,
    tokens: Optional[torch.Tensor] = None,
    prompt: Optional[str] = None,
    embed: Optional[torch.Tensor] = None,
    entry_length: int = 67,
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = ".",
) -> Tuple[str, List[int]]:
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        if embed is not None:
            generated = embed
            cur_tokens = torch.empty((1, 0), dtype=torch.long, device=device) if tokens is None else tokens.to(device)
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            cur_tokens = tokens
            generated = model.gpt.transformer.wte(cur_tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value

            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.gpt.transformer.wte(next_token)

            cur_tokens = torch.cat((cur_tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            if stop_token_index == next_token.item():
                break

    token_ids = [int(x) for x in cur_tokens.squeeze(0).cpu().tolist()]
    text = tokenizer.decode(token_ids)
    return text, token_ids


def generate_predictions_with_token_ids(
    canonical_model: str,
    model_cfg: Dict[str, Optional[str]],
    images_root: Path,
    checkpoint_path: Path,
    output_csv: Path,
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    beam_search: bool,
    beam_size: int,
    entry_length: int,
    top_p: float,
    temperature: float,
    stop_token: str,
    tokenizer_name: str,
    progress_desc: str,
) -> List[Dict[str, object]]:
    from test import (  # noqa: WPS433
        encode_image_to_embedding,
        load_biomedclip_encoder,
        load_mapper,
        load_openai_clip,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("La regeneracion de captions/token IDs requiere GPU (cuda).")

    if canonical_model == "biomedclip":
        encoder_name = "biomedclip"
        clip_model, preprocess, inferred_dim = load_biomedclip_encoder(
            device=device,
            model_id=str(model_cfg["biomedclip_model_id"]),
        )
    elif canonical_model == "vit":
        encoder_name = "vit"
        clip_model, preprocess, inferred_dim = load_openai_clip(device, str(model_cfg["openai_clip_name"]))
    elif canonical_model == "resnet101":
        encoder_name = "rn101"
        clip_model, preprocess, inferred_dim = load_openai_clip(device, str(model_cfg["openai_clip_name"]))
    else:
        raise ValueError(f"Modelo no soportado para regeneracion: {canonical_model}")

    mapper = load_mapper(
        checkpoint_path=str(checkpoint_path),
        prefix_length=prefix_length,
        mapping_type=mapping_type,
        prefix_size=inferred_dim,
        num_layers=num_layers,
        device=device,
    )

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    output_rows: List[Dict[str, object]] = []
    image_paths = sorted(iter_images(images_root))

    for img_path in tqdm(image_paths, desc=progress_desc):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            output_rows.append(
                {
                    "image_path": img_path.resolve().as_posix(),
                    "generated_caption": "",
                    "generated_token_ids": "[]",
                    "generated_token_count": 0,
                    "error": f"image_open_error: {exc}",
                }
            )
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)
        clip_embed = encode_image_to_embedding(encoder_name, clip_model, image_input)
        with torch.no_grad():
            prefix_embed = mapper.clip_project(clip_embed).view(1, prefix_length, -1)

        if beam_search:
            caption, token_ids = generate_beam_with_ids(
                mapper,
                tokenizer,
                beam_size=beam_size,
                embed=prefix_embed,
                entry_length=entry_length,
                temperature=temperature,
                stop_token=stop_token,
            )
        else:
            caption, token_ids = generate_top_p_with_ids(
                mapper,
                tokenizer,
                prompt=None,
                embed=prefix_embed,
                entry_length=entry_length,
                top_p=top_p,
                temperature=temperature,
                stop_token=stop_token,
            )

        output_rows.append(
            {
                "image_path": img_path.resolve().as_posix(),
                "generated_caption": caption,
                "generated_token_ids": json.dumps(token_ids, ensure_ascii=False),
                "generated_token_count": len(token_ids),
                "error": "",
            }
        )

    write_csv(output_csv, output_rows, RAW_PRED_WITH_TOKENS_COLUMNS)
    return output_rows


def extract_epoch_from_checkpoint_name(filename: str, checkpoint_prefix: str) -> Optional[int]:
    match = re.match(rf"^{re.escape(checkpoint_prefix)}-(\d+)\.pt$", filename)
    if not match:
        return None
    return int(match.group(1))


def list_available_epochs(train_dir: Path, checkpoint_prefix: str) -> List[int]:
    epochs: List[int] = []
    for ckpt in train_dir.glob(f"{checkpoint_prefix}-*.pt"):
        parsed = extract_epoch_from_checkpoint_name(ckpt.name, checkpoint_prefix)
        if parsed is not None:
            epochs.append(parsed)
    return sorted(epochs)


def evaluate_validation_checkpoint(
    val_pkl: Path,
    checkpoint_path: Path,
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    batch_size: int,
    include_batch_rows: bool,
    include_embeddings: bool = False,
) -> Dict[str, object]:
    from torch.utils.data import DataLoader

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from train import ClipCocoDataset, ClipCaptionPrefix, MappingType  # noqa: WPS433

    dataset = ClipCocoDataset(str(val_pkl), prefix_length, normalize_prefix=True)

    if hasattr(dataset.prefixes, "shape") and len(dataset.prefixes.shape) >= 2:
        prefix_size = int(dataset.prefixes.shape[-1])
    else:
        sample_prefix = dataset[0][2]
        prefix_size = int(sample_prefix.shape[-1])

    mapping_enum = MappingType.Transformer if mapping_type == "transformer" else MappingType.MLP
    model = ClipCaptionPrefix(
        prefix_length=prefix_length,
        clip_length=prefix_length,
        prefix_size=prefix_size,
        num_layers=num_layers,
        mapping_type=mapping_enum,
    )

    state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    val_loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
    )

    total_samples = 0
    weighted_loss = 0.0
    total_batches = 0
    step_rows: List[Dict[str, object]] = []
    pre_gpt_chunks: List[torch.Tensor] = []
    post_gpt_chunks: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, (tokens, mask, prefix) in enumerate(val_loader, start=1):
            total_batches += 1
            tokens = tokens.to(device)
            mask = mask.to(device)
            prefix = prefix.to(device, dtype=torch.float32)

            prefix_proj = model.clip_project(prefix).view(-1, prefix_length, model.gpt_embedding_size)
            embedding_text = model.gpt.transformer.wte(tokens)
            embedding_cat = torch.cat((prefix_proj, embedding_text), dim=1)

            outputs = model.gpt(
                inputs_embeds=embedding_cat,
                attention_mask=mask,
                output_hidden_states=include_embeddings,
                return_dict=True,
            )
            logits = outputs.logits[:, prefix_length - 1 : -1]
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss_value = float(loss.item())

            n_items = int(tokens.shape[0])
            total_samples += n_items
            weighted_loss += loss_value * n_items

            if include_batch_rows:
                step_rows.append(
                    {
                        "batch": batch_idx,
                        "num_samples": n_items,
                        "val_loss": loss_value,
                    }
                )
            if include_embeddings:
                pre_gpt_chunks.append(prefix_proj.detach().cpu())
                hidden_states = outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("Se solicitaron embeddings pero el modelo no devolvio hidden_states.")
                post_gpt_chunks.append(hidden_states[-1][:, :prefix_length, :].detach().cpu())

    mean_val_loss = (weighted_loss / total_samples) if total_samples else 0.0
    result: Dict[str, object] = {
        "val_batches": total_batches,
        "val_samples": total_samples,
        "mean_val_loss": mean_val_loss,
    }
    if include_batch_rows:
        result["step_rows"] = step_rows
    if include_embeddings:
        if pre_gpt_chunks:
            pre_gpt_tensor = torch.cat(pre_gpt_chunks, dim=0)
            post_gpt_tensor = torch.cat(post_gpt_chunks, dim=0)
        else:
            pre_gpt_tensor = torch.empty((0, prefix_length, model.gpt_embedding_size), dtype=torch.float32)
            post_gpt_tensor = torch.empty((0, prefix_length, model.gpt_embedding_size), dtype=torch.float32)
        result["pre_gpt_tensor"] = pre_gpt_tensor
        result["post_gpt_tensor"] = post_gpt_tensor
        result["image_ids"] = list(getattr(dataset, "image_ids", []))
    return result


def compute_and_save_val_loss_per_epoch(
    val_pkl: Path,
    train_dir: Path,
    checkpoint_prefix: str,
    start_epoch: int,
    end_epoch: int,
    out_csv_path: Path,
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    batch_size: int,
    allow_missing_checkpoints: bool,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    available_epochs = sorted(list_available_epochs(train_dir, checkpoint_prefix))
    if not available_epochs:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en {train_dir} con prefijo {checkpoint_prefix}."
        )
    available_set = set(available_epochs)

    for epoch in range(start_epoch, end_epoch + 1):
        checkpoint_epoch = epoch
        if epoch not in available_set:
            if allow_missing_checkpoints:
                print(
                    f"[WARN] Epoch {epoch:03d} sin checkpoint en {train_dir}; "
                    "se omite por --allow_missing_checkpoints."
                )
                continue
            previous = [ep for ep in available_epochs if ep <= epoch]
            checkpoint_epoch = previous[-1] if previous else available_epochs[0]
            print(
                f"[WARN] Epoch {epoch:03d} sin checkpoint en {train_dir}; "
                f"se usa checkpoint epoch {checkpoint_epoch:03d}."
            )

        checkpoint_path = train_dir / f"{checkpoint_prefix}-{checkpoint_epoch:03d}.pt"

        metrics = evaluate_validation_checkpoint(
            val_pkl=val_pkl,
            checkpoint_path=checkpoint_path,
            prefix_length=prefix_length,
            mapping_type=mapping_type,
            num_layers=num_layers,
            batch_size=batch_size,
            include_batch_rows=False,
        )
        rows.append(
            {
                "epoch": epoch,
                "checkpoint": to_repo_relative(checkpoint_path),
                "val_loss": metrics["mean_val_loss"],
                "val_batches": metrics["val_batches"],
                "val_samples": metrics["val_samples"],
            }
        )

    write_csv(out_csv_path, rows, VAL_LOSS_PER_EPOCH_COLUMNS)
    return rows


def find_checkpoint_for_epoch(train_dir: Path, checkpoint_prefix: str, epoch_hint: int) -> Path:
    preferred = train_dir / f"{checkpoint_prefix}-{epoch_hint:03d}.pt"
    if preferred.exists():
        return preferred

    available_epochs = list_available_epochs(train_dir, checkpoint_prefix)
    if not available_epochs:
        raise FileNotFoundError(
            f"No se encontraron checkpoints en {train_dir} con prefijo {checkpoint_prefix}."
        )
    return train_dir / f"{checkpoint_prefix}-{available_epochs[-1]:03d}.pt"


def compute_and_save_validation_artifacts(
    val_pkl: Path,
    checkpoint_path: Path,
    val_loss_per_epoch_csv_path: Path,
    val_loss_summary_path: Path,
    pre_gpt_embeddings_path: Path,
    post_gpt_embeddings_path: Path,
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    batch_size: int,
) -> Dict[str, object]:
    metrics = evaluate_validation_checkpoint(
        val_pkl=val_pkl,
        checkpoint_path=checkpoint_path,
        prefix_length=prefix_length,
        mapping_type=mapping_type,
        num_layers=num_layers,
        batch_size=batch_size,
        include_batch_rows=False,
        include_embeddings=True,
    )

    pre_gpt_tensor = metrics["pre_gpt_tensor"]
    post_gpt_tensor = metrics["post_gpt_tensor"]
    image_ids = metrics["image_ids"]

    pre_gpt_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    post_gpt_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": pre_gpt_tensor,
            "image_ids": image_ids,
            "source_val_pkl": to_repo_relative(val_pkl),
            "checkpoint": to_repo_relative(checkpoint_path),
        },
        pre_gpt_embeddings_path,
    )
    torch.save(
        {
            "embeddings": post_gpt_tensor,
            "image_ids": image_ids,
            "source_val_pkl": to_repo_relative(val_pkl),
            "checkpoint": to_repo_relative(checkpoint_path),
        },
        post_gpt_embeddings_path,
    )

    summary = {
        "val_batches": metrics["val_batches"],
        "val_samples": metrics["val_samples"],
        "mean_val_loss": metrics["mean_val_loss"],
        "val_loss_csv": to_repo_relative(val_loss_per_epoch_csv_path),
        "val_loss_per_epoch_csv": to_repo_relative(val_loss_per_epoch_csv_path),
        "pre_gpt_embeddings_path": to_repo_relative(pre_gpt_embeddings_path),
        "post_gpt_embeddings_path": to_repo_relative(post_gpt_embeddings_path),
        "pre_gpt_embeddings_shape": list(pre_gpt_tensor.shape),
        "post_gpt_embeddings_shape": list(post_gpt_tensor.shape),
    }
    write_json(val_loss_summary_path, summary)
    return summary


def merge_predictions_with_manifest(
    fold_name: str,
    pred_rows: Sequence[Dict[str, object]],
    manifest_rows: Sequence[Dict[str, str]],
) -> List[Dict[str, object]]:
    manifest_map = {row["linked_image_path"]: row for row in manifest_rows}
    merged: List[Dict[str, object]] = []

    for pred in pred_rows:
        linked = str(pred.get("image_path", "")).strip()
        meta = manifest_map.get(linked)
        if meta is None:
            continue

        raw_count = pred.get("generated_token_count", 0)
        try:
            token_count = int(raw_count)
        except Exception:
            token_count = 0

        merged.append(
            {
                "fold": fold_name,
                "sample_id": meta["sample_id"],
                "label": meta["label"],
                "case": meta["case"],
                "image_path": meta["image_path"],
                "caption_gt": meta["caption_gt"],
                "generated_caption": str(pred.get("generated_caption", "")),
                "linked_image_path": linked,
                "generated_token_ids": str(pred.get("generated_token_ids", "[]")),
                "generated_token_count": token_count,
            }
        )

    return merged


def main() -> None:
    args = parse_args()

    if args.end_epoch < args.start_epoch:
        raise ValueError("--end_epoch debe ser mayor o igual que --start_epoch.")

    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_root = resolve_repo_path(args.output_root)
    models_to_run = select_models(args.model)

    print(f"Modelos a validar: {models_to_run}")
    print(f"Rango de epocas: {args.start_epoch}..{args.end_epoch}")

    for canonical_model in models_to_run:
        model_cfg = MODEL_CONFIG[canonical_model]
        output_subdir = MODEL_OUTPUT_DIR[canonical_model]
        model_root = output_root / output_subdir

        run_config_path = model_root / "run_config.json"
        summary_path = model_root / "summary.json"
        if not run_config_path.exists():
            raise FileNotFoundError(f"No existe run_config.json para {canonical_model}: {run_config_path}")

        run_config = read_json(run_config_path)
        training_cfg = run_config.get("training", {})
        mapping_type = normalize_mapping_type(training_cfg.get("mapping_type", "transformer"))
        prefix_length = int(training_cfg.get("prefix_length", 10))
        num_layers = int(training_cfg.get("num_layers", 8))
        batch_size = int(training_cfg.get("bs", 4))

        outputs_cfg = run_config.get("outputs", {})
        folds_cfg = outputs_cfg.get("folds", {})
        if not isinstance(folds_cfg, dict):
            raise ValueError(f"Estructura invalida de folds en {run_config_path}")

        all_generated_rows: List[Dict[str, object]] = []
        execution_report: List[Dict[str, object]] = []

        for fold_name in ("fold_1", "fold_2"):
            if fold_name not in folds_cfg:
                raise KeyError(f"No existe {fold_name} en run_config outputs.folds")

            fold_artifacts = folds_cfg[fold_name]
            if not isinstance(fold_artifacts, dict):
                raise ValueError(f"fold_artifacts invalido en {fold_name}")

            fold_dir = resolve_artifact_path(fold_artifacts.get("fold_dir"), model_root / "folds" / fold_name)
            train_dir = resolve_artifact_path(fold_artifacts.get("train_dir"), fold_dir / "train")
            val_csv = resolve_artifact_path(fold_artifacts.get("val_csv"), fold_dir / "val.csv")
            val_pkl = resolve_artifact_path(fold_artifacts.get("val_pkl"), fold_dir / "data" / "val.pkl")
            inference_dir = resolve_artifact_path(fold_artifacts.get("inference_dir"), fold_dir / "inference")

            manifest_csv = resolve_artifact_path(
                fold_artifacts.get("val_manifest_csv"),
                inference_dir / "val_manifest.csv",
            )
            raw_pred_csv = resolve_artifact_path(
                fold_artifacts.get("val_predictions_raw_csv"),
                inference_dir / "val_predictions_raw.csv",
            )
            merged_pred_csv = resolve_artifact_path(
                fold_artifacts.get("val_predictions_csv"),
                inference_dir / "val_predictions.csv",
            )
            val_loss_csv = resolve_artifact_path(
                fold_artifacts.get("val_loss_csv"),
                inference_dir / "val_loss_per_batch.csv",
            )
            val_loss_summary_json = resolve_artifact_path(
                fold_artifacts.get("val_loss_summary_json"),
                inference_dir / "val_loss_summary.json",
            )
            val_loss_per_epoch_csv = resolve_artifact_path(
                fold_artifacts.get("val_loss_per_epoch_csv"),
                fold_dir / "val_loss_per_epoch.csv",
            )

            for required in (train_dir, val_csv, val_pkl):
                if not required.exists():
                    raise FileNotFoundError(f"No existe requerido para {canonical_model}/{fold_name}: {required}")

            print(f"\n[{canonical_model}] {fold_name}")
            print(f"  val_pkl: {to_repo_relative(val_pkl)}")
            print(f"  train_dir: {to_repo_relative(train_dir)}")

            val_rows = read_csv_rows(val_csv)
            val_images_dir = inference_dir / "val_images_validation"
            reset_dir(val_images_dir)

            manifest_rows, missing_count = create_val_image_manifest(val_rows, val_images_dir)
            write_csv(manifest_csv, manifest_rows, MANIFEST_COLUMNS)

            train_prefix = f"positive_vs_negative_{fold_name}"
            val_loss_per_epoch_rows = compute_and_save_val_loss_per_epoch(
                val_pkl=val_pkl,
                train_dir=train_dir,
                checkpoint_prefix=train_prefix,
                start_epoch=args.start_epoch,
                end_epoch=args.end_epoch,
                out_csv_path=val_loss_per_epoch_csv,
                prefix_length=prefix_length,
                mapping_type=mapping_type,
                num_layers=num_layers,
                batch_size=batch_size,
                allow_missing_checkpoints=args.allow_missing_checkpoints,
            )

            if not val_loss_per_epoch_rows:
                raise RuntimeError(
                    f"No se evaluo ninguna epoca para {canonical_model}/{fold_name}. "
                    "Verifica checkpoints y rango de epocas."
                )

            final_checkpoint = train_dir / f"{train_prefix}-{args.end_epoch:03d}.pt"
            if not final_checkpoint.exists():
                last_checkpoint_rel = str(val_loss_per_epoch_rows[-1]["checkpoint"])
                final_checkpoint = resolve_repo_path(last_checkpoint_rel)

            final_metrics = evaluate_validation_checkpoint(
                val_pkl=val_pkl,
                checkpoint_path=final_checkpoint,
                prefix_length=prefix_length,
                mapping_type=mapping_type,
                num_layers=num_layers,
                batch_size=batch_size,
                include_batch_rows=True,
            )
            write_csv(val_loss_csv, final_metrics.get("step_rows", []), VAL_LOSS_COLUMNS)
            write_json(
                val_loss_summary_json,
                {
                    "checkpoint": to_repo_relative(final_checkpoint),
                    "val_batches": final_metrics["val_batches"],
                    "val_samples": final_metrics["val_samples"],
                    "mean_val_loss": final_metrics["mean_val_loss"],
                    "val_loss_csv": to_repo_relative(val_loss_csv),
                    "val_loss_per_epoch_csv": to_repo_relative(val_loss_per_epoch_csv),
                },
            )

            raw_pred_rows = generate_predictions_with_token_ids(
                canonical_model=canonical_model,
                model_cfg=model_cfg,
                images_root=val_images_dir,
                checkpoint_path=final_checkpoint,
                output_csv=raw_pred_csv,
                prefix_length=prefix_length,
                mapping_type=mapping_type,
                num_layers=num_layers,
                beam_search=args.beam_search,
                beam_size=args.beam_size,
                entry_length=args.entry_length,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_token=args.stop_token,
                tokenizer_name=args.tokenizer_name,
                progress_desc=f"{canonical_model}:{fold_name}",
            )

            merged_pred_rows = merge_predictions_with_manifest(
                fold_name=fold_name,
                pred_rows=raw_pred_rows,
                manifest_rows=manifest_rows,
            )
            write_csv(merged_pred_csv, merged_pred_rows, VAL_PRED_WITH_TOKENS_COLUMNS)
            all_generated_rows.extend(merged_pred_rows)

            execution_report.append(
                {
                    "fold": fold_name,
                    "checkpoint": to_repo_relative(final_checkpoint),
                    "val_images_dir": to_repo_relative(val_images_dir),
                    "val_manifest_rows": len(manifest_rows),
                    "val_missing_images": missing_count,
                    "raw_predictions_rows": len(raw_pred_rows),
                    "matched_predictions_rows": len(merged_pred_rows),
                    "val_loss_per_epoch_csv": to_repo_relative(val_loss_per_epoch_csv),
                    "val_loss_per_epoch_points": len(val_loss_per_epoch_rows),
                    "mean_val_loss_final_checkpoint": final_metrics["mean_val_loss"],
                    "val_batches_final_checkpoint": final_metrics["val_batches"],
                    "val_samples_final_checkpoint": final_metrics["val_samples"],
                    "epochs_evaluated": [int(r["epoch"]) for r in val_loss_per_epoch_rows],
                }
            )

            print(
                "  "
                f"manifest={len(manifest_rows)} "
                f"missing={missing_count} "
                f"pred_raw={len(raw_pred_rows)} "
                f"pred_matched={len(merged_pred_rows)} "
                f"val_loss_final={final_metrics['mean_val_loss']:.6f} "
                f"epochs_eval={len(val_loss_per_epoch_rows)}"
            )

        captions_generated_csv = resolve_artifact_path(
            outputs_cfg.get("captions_generated_csv"),
            model_root / "data" / "captions_generated.csv",
        )
        write_csv(captions_generated_csv, all_generated_rows, VAL_PRED_WITH_TOKENS_COLUMNS)

        validation_payload = {
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "epochs_requested": {"start": args.start_epoch, "end": args.end_epoch},
            "allow_missing_checkpoints": bool(args.allow_missing_checkpoints),
            "beam_search": bool(args.beam_search),
            "beam_size": int(args.beam_size),
            "entry_length": int(args.entry_length),
            "top_p": float(args.top_p),
            "temperature": float(args.temperature),
            "tokenizer_name": str(args.tokenizer_name),
            "generated_captions_rows": len(all_generated_rows),
            "execution_report": execution_report,
        }

        run_config["execution_report"] = execution_report
        run_config["validation"] = validation_payload
        write_json(run_config_path, run_config)

        summary = read_json(summary_path) if summary_path.exists() else {}
        summary["execution_report"] = execution_report
        summary["generated_captions_rows"] = len(all_generated_rows)
        summary["validation"] = validation_payload
        write_json(summary_path, summary)

        print(
            f"\n[{canonical_model}] completo: captions_generated={to_repo_relative(captions_generated_csv)} "
            f"rows={len(all_generated_rows)}"
        )

    print("\nValidacion finalizada.")


if __name__ == "__main__":
    main()
    

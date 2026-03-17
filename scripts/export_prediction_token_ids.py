#!/usr/bin/env python3
"""Export token IDs for generated captions.

This script supports two workflows:
1) Retokenize existing generated captions from a CSV.
2) Regenerate captions from images/checkpoint and save exact generated token IDs.

By default, outputs are saved under: fold/2fold/Data
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as nnf
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "fold" / "2fold" / "Data"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def iter_images(root_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in exts:
                yield Path(dirpath) / filename


def _safe_stem(path: Path) -> str:
    stem = path.stem.strip() or "predictions"
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("-", "_"))


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_retokenize(
    predictions_csv: Path,
    output_dir: Path,
    output_name: Optional[str],
    caption_column: str,
    tokenizer_name: str,
) -> Path:
    with predictions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fieldnames_in = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"CSV sin filas: {predictions_csv}")
    if caption_column not in rows[0]:
        raise KeyError(
            f"No se encontro la columna '{caption_column}' en {predictions_csv}. "
            f"Columnas: {fieldnames_in}"
        )

    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    output_rows: List[dict] = []
    for row in rows:
        caption = (row.get(caption_column) or "").strip()
        token_ids = tokenizer.encode(caption, add_special_tokens=False)
        new_row = dict(row)
        new_row["generated_token_ids"] = json.dumps([int(x) for x in token_ids], ensure_ascii=False)
        new_row["generated_token_count"] = len(token_ids)
        output_rows.append(new_row)

    output_basename = output_name or f"{_safe_stem(predictions_csv)}_retokenized_with_token_ids.csv"
    output_path = output_dir / output_basename
    out_fieldnames = list(fieldnames_in)
    if "generated_token_ids" not in out_fieldnames:
        out_fieldnames.append("generated_token_ids")
    if "generated_token_count" not in out_fieldnames:
        out_fieldnames.append("generated_token_count")
    write_csv(output_path, output_rows, out_fieldnames)
    return output_path


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
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
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
            if tokens is None:
                cur_tokens = torch.empty((1, 0), dtype=torch.long, device=device)
            else:
                cur_tokens = tokens.to(device)
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            cur_tokens = tokens
            generated = model.gpt.transformer.wte(cur_tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

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


def run_regenerate(
    images_root: Path,
    checkpoint: Path,
    output_dir: Path,
    output_name: Optional[str],
    prefix_length: int,
    mapping_type: str,
    num_layers: int,
    beam_search: bool,
    encoder: str,
    openai_clip_name: str,
    openclip_model: str,
    openclip_pretrained: Optional[str],
    openclip_ckpt: Optional[str],
    biomedclip_model_id: str,
    prefix_size: int,
    entry_length: int,
    top_p: float,
    temperature: float,
    stop_token: str,
    tokenizer_name: str,
    beam_size: int,
) -> Path:
    from test import (  # noqa: WPS433
        encode_image_to_embedding,
        load_biomedclip_encoder,
        load_mapper,
        load_openai_clip,
        load_openclip_encoder,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Regenerate requiere GPU (cuda), igual que test.py.")

    if encoder in ("vit", "rn101"):
        clip_name = openai_clip_name if openai_clip_name else ("ViT-B/32" if encoder == "vit" else "RN101")
        clip_model, preprocess, inferred_dim = load_openai_clip(device, clip_name)
    elif encoder == "biomedclip":
        clip_model, preprocess, inferred_dim = load_biomedclip_encoder(device=device, model_id=biomedclip_model_id)
    else:
        clip_model, preprocess, inferred_dim = load_openclip_encoder(
            device=device,
            openclip_model=openclip_model,
            openclip_pretrained=openclip_pretrained,
            openclip_ckpt=openclip_ckpt,
        )

    use_prefix_size = prefix_size if prefix_size > 0 else inferred_dim
    mapper = load_mapper(
        checkpoint_path=str(checkpoint),
        prefix_length=prefix_length,
        mapping_type=mapping_type,
        prefix_size=use_prefix_size,
        num_layers=num_layers,
        device=device,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    image_paths = sorted(iter_images(images_root))
    output_rows: List[dict] = []
    for img_path in tqdm(image_paths, desc="Generando captions+ids"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            output_rows.append(
                {
                    "image_path": str(img_path),
                    "generated_caption": "",
                    "generated_token_ids": "[]",
                    "generated_token_count": 0,
                    "error": f"image_open_error: {exc}",
                }
            )
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)
        clip_embed = encode_image_to_embedding(encoder, clip_model, image_input)
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
                "image_path": str(img_path),
                "generated_caption": caption,
                "generated_token_ids": json.dumps(token_ids, ensure_ascii=False),
                "generated_token_count": len(token_ids),
                "error": "",
            }
        )

    default_name = f"{_safe_stem(images_root)}_regenerated_with_token_ids.csv"
    output_path = output_dir / (output_name or default_name)
    fieldnames = [
        "image_path",
        "generated_caption",
        "generated_token_ids",
        "generated_token_count",
        "error",
    ]
    write_csv(output_path, output_rows, fieldnames)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Exporta token IDs de captions generados. "
            "Usa subcomando retokenize o regenerate."
        )
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directorio de salida (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help=(
            "GPU para CUDA_VISIBLE_DEVICES (ej: 0, 1, 3). "
            "Aplica al modo regenerate."
        ),
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_retok = subparsers.add_parser(
        "retokenize",
        help="Reconstruye token IDs desde un CSV existente con generated_caption.",
    )
    p_retok.add_argument("--predictions_csv", required=True, help="CSV con captions generados.")
    p_retok.add_argument("--caption_column", default="generated_caption", help="Columna con texto generado.")
    p_retok.add_argument("--tokenizer_name", default="gpt2", help="Tokenizer HF (default: gpt2).")
    p_retok.add_argument("--output_name", default=None, help="Nombre del CSV de salida.")

    p_regen = subparsers.add_parser(
        "regenerate",
        help="Re-genera captions y guarda token IDs exactos de decodificacion.",
    )
    p_regen.add_argument("--images_root", required=True, help="Directorio con imagenes.")
    p_regen.add_argument("--checkpoint", required=True, help="Checkpoint .pt del mapper.")
    p_regen.add_argument("--prefix_length", type=int, default=10)
    p_regen.add_argument("--mapping_type", type=str, default="transformer", choices=["mlp", "transformer"])
    p_regen.add_argument("--num_layers", type=int, default=8)

    p_regen.add_argument("--beam_search", dest="beam_search", action="store_true", default=True)
    p_regen.add_argument("--no_beam_search", dest="beam_search", action="store_false")
    p_regen.add_argument("--beam_size", type=int, default=5)

    p_regen.add_argument(
        "--encoder",
        type=str,
        default="vit",
        choices=["vit", "rn101", "openclip", "biomedclip"],
    )
    p_regen.add_argument("--openai_clip_name", type=str, default="")
    p_regen.add_argument("--openclip_model", type=str, default="ViT-B-16")
    p_regen.add_argument("--openclip_pretrained", type=str, default=None)
    p_regen.add_argument("--openclip_ckpt", type=str, default=None)
    p_regen.add_argument(
        "--biomedclip_model_id",
        type=str,
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    p_regen.add_argument("--prefix_size", type=int, default=0)
    p_regen.add_argument("--entry_length", type=int, default=67)
    p_regen.add_argument("--top_p", type=float, default=0.8)
    p_regen.add_argument("--temperature", type=float, default=1.0)
    p_regen.add_argument("--stop_token", type=str, default=".")
    p_regen.add_argument("--tokenizer_name", default="gpt2")
    p_regen.add_argument("--output_name", default=None, help="Nombre del CSV de salida.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "retokenize":
        predictions_csv = resolve_repo_path(args.predictions_csv)
        out = run_retokenize(
            predictions_csv=predictions_csv,
            output_dir=output_dir,
            output_name=args.output_name,
            caption_column=args.caption_column,
            tokenizer_name=args.tokenizer_name,
        )
        print(f"[OK] CSV con token IDs (retokenize): {out}")
        return

    if args.mode == "regenerate":
        images_root = resolve_repo_path(args.images_root)
        checkpoint = resolve_repo_path(args.checkpoint)
        out = run_regenerate(
            images_root=images_root,
            checkpoint=checkpoint,
            output_dir=output_dir,
            output_name=args.output_name,
            prefix_length=args.prefix_length,
            mapping_type=args.mapping_type,
            num_layers=args.num_layers,
            beam_search=args.beam_search,
            encoder=args.encoder,
            openai_clip_name=args.openai_clip_name,
            openclip_model=args.openclip_model,
            openclip_pretrained=args.openclip_pretrained,
            openclip_ckpt=args.openclip_ckpt,
            biomedclip_model_id=args.biomedclip_model_id,
            prefix_size=args.prefix_size,
            entry_length=args.entry_length,
            top_p=args.top_p,
            temperature=args.temperature,
            stop_token=args.stop_token,
            tokenizer_name=args.tokenizer_name,
            beam_size=args.beam_size,
        )
        print(f"[OK] CSV con token IDs exactos (regenerate): {out}")
        return

    raise RuntimeError(f"Modo no soportado: {args.mode}")


if __name__ == "__main__":
    main()

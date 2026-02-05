# test_multiencoder.py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import argparse
from typing import Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from transformers import GPT2Tokenizer

# OpenAI CLIP
import clip

# Tu repo
from train import ClipCaptionPrefix, MappingType
from predict import generate2, generate_beam


def iter_images(root_dir: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname.lower())[1]
            if ext in exts:
                yield os.path.join(dirpath, fname)


# -------------------------
# Encoders visuales (imagen -> embedding)
# -------------------------
def load_openai_clip(device: torch.device, model_name: str) -> Tuple[torch.nn.Module, object, int]:
    """
    model_name: "ViT-B/32" o "RN101" (u otros soportados por clip.load)
    Devuelve: (model, preprocess, embed_dim)
    """
    print(f"[INFO] Cargando OpenAI CLIP: {model_name} ...")
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # inferir dim haciendo una pasada dummy
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        emb = model.encode_image(dummy).float()
        embed_dim = int(emb.shape[-1])

    print(f"[INFO] OpenAI CLIP cargado. embed_dim={embed_dim}")
    return model, preprocess, embed_dim


def load_openclip_encoder(
    device: torch.device,
    openclip_model: str,
    openclip_pretrained: Optional[str],
    openclip_ckpt: Optional[str],
) -> Tuple[torch.nn.Module, object, int]:
    """
    Carga un encoder visual usando open_clip (útil para BioMedCLIP u otros OpenCLIP).
    Puedes usar:
      - --openclip_model ViT-B-16  (ejemplo)
      - --openclip_pretrained "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        o un tag de LAION, etc.
      - o --openclip_ckpt /ruta/a/weights.pt para cargar pesos locales

    Devuelve: (model, preprocess, embed_dim)

    Nota: open_clip devuelve preprocess como una transform; aquí solo usamos el preprocess de validación.
    """
    try:
        import open_clip
    except Exception as e:
        raise RuntimeError(
            "No pude importar open_clip. Instálalo en tu entorno si vas a usar --encoder openclip.\n"
            f"Error: {e}"
        )

    print("[INFO] Cargando OpenCLIP...")
    print(f"[INFO]   model={openclip_model}")
    print(f"[INFO]   pretrained={openclip_pretrained}")
    print(f"[INFO]   ckpt={openclip_ckpt}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=openclip_model,
        pretrained=openclip_pretrained if openclip_pretrained else None,
        device=device,
    )
    model.eval()

    # Si te dieron pesos locales .pt
    if openclip_ckpt:
        # Diferentes versiones de open_clip tienen APIs distintas
        loaded = False
        try:
            open_clip.load_checkpoint(model, openclip_ckpt)
            loaded = True
            print("[INFO] Pesos open_clip cargados con open_clip.load_checkpoint().")
        except Exception:
            pass

        if not loaded:
            try:
                sd = torch.load(openclip_ckpt, map_location="cpu")
                # algunos checkpoints vienen como {"state_dict": ...}
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                model.load_state_dict(sd, strict=False)
                loaded = True
                print("[INFO] Pesos open_clip cargados con model.load_state_dict(strict=False).")
            except Exception as e:
                raise RuntimeError(f"No pude cargar openclip_ckpt={openclip_ckpt}. Error: {e}")

    # inferir dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224, device=device)
        # open_clip suele exponer encode_image
        if hasattr(model, "encode_image"):
            emb = model.encode_image(dummy).float()
        else:
            # fallback: algunos modelos usan forward(image=...)
            emb = model(dummy).float()
        embed_dim = int(emb.shape[-1])

    print(f"[INFO] OpenCLIP cargado. embed_dim={embed_dim}")
    return model, preprocess, embed_dim


def encode_image_to_embedding(
    encoder_name: str,
    clip_model,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Retorna embedding [1, D] normalizado
    """
    with torch.no_grad():
        if encoder_name in ("vit", "rn101"):
            emb = clip_model.encode_image(image_tensor).float()
        else:
            # open_clip / biomedclip
            if hasattr(clip_model, "encode_image"):
                emb = clip_model.encode_image(image_tensor).float()
            else:
                emb = clip_model(image_tensor).float()

        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb


# -------------------------
# Mapper (embedding -> prefijo GPT2)
# -------------------------
def load_mapper(
    checkpoint_path: str,
    prefix_length: int,
    mapping_type: str,
    prefix_size: int,
    device: torch.device,
    num_layers: int = 8,
) -> ClipCaptionPrefix:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No se encontró checkpoint: {checkpoint_path}")

    if mapping_type.lower() == "transformer":
        m_type = MappingType.Transformer
    else:
        m_type = MappingType.MLP

    model = ClipCaptionPrefix(
        prefix_length=prefix_length,
        clip_length=prefix_length,   # en tu repo suelen usar clip_length=prefix_length
        prefix_size=prefix_size,     # DEBE coincidir con el embedding dim del encoder usado al entrenar
        num_layers=num_layers,
        mapping_type=m_type,
    )

    sd = torch.load(checkpoint_path, map_location=device)
    # por si el checkpoint trae wrapper {"state_dict": ...}
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print("[INFO] Mapper cargado correctamente.")
    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--mapping_type", type=str, default="transformer")
    parser.add_argument("--num_layers", type=int, default=8)

    parser.add_argument("--beam_search", action="store_true")

    # Encoder selector
    parser.add_argument(
        "--encoder",
        type=str,
        default="vit",
        choices=["vit", "rn101", "openclip"],
        help="vit: OpenAI CLIP ViT-B/32 | rn101: OpenAI CLIP RN101 | openclip: OpenCLIP/BioMedCLIP",
    )

    # Para OpenAI CLIP:
    parser.add_argument("--openai_clip_name", type=str, default="", help="Override: ViT-B/32 o RN101")

    # Para OpenCLIP/BioMedCLIP:
    parser.add_argument("--openclip_model", type=str, default="ViT-B-16")
    parser.add_argument("--openclip_pretrained", type=str, default=None)
    parser.add_argument("--openclip_ckpt", type=str, default=None)

    # Opcional: forzar prefix_size si ya sabes (si no, se infiere del encoder)
    parser.add_argument("--prefix_size", type=int, default=0)

    # params de generación
    parser.add_argument("--entry_length", type=int, default=67)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stop_token", type=str, default=".")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA current device:", torch.cuda.current_device() if torch.cuda.is_available() else "none")

    if device.type != "cuda":
        raise RuntimeError("Este script está pensado para GPU (cuda).")

    # 1) Cargar encoder visual
    if args.encoder in ("vit", "rn101"):
        # permite override manual
        if args.openai_clip_name:
            clip_name = args.openai_clip_name
        else:
            clip_name = "ViT-B/32" if args.encoder == "vit" else "RN101"

        clip_model, preprocess, inferred_dim = load_openai_clip(device, clip_name)
    else:
        clip_model, preprocess, inferred_dim = load_openclip_encoder(
            device=device,
            openclip_model=args.openclip_model,
            openclip_pretrained=args.openclip_pretrained,
            openclip_ckpt=args.openclip_ckpt,
        )

    prefix_size = args.prefix_size if args.prefix_size > 0 else inferred_dim
    print(f"[INFO] prefix_size usado por el mapper: {prefix_size}")

    # 2) Cargar mapper
    print(f"[INFO] Cargando mapper desde: {args.checkpoint}")
    mapper = load_mapper(
        checkpoint_path=args.checkpoint,
        prefix_length=args.prefix_length,
        mapping_type=args.mapping_type,
        prefix_size=prefix_size,
        num_layers=args.num_layers,
        device=device,
    )

    # 3) Tokenizer GPT2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 4) Imágenes
    image_paths = sorted(list(iter_images(args.images_root)))
    print(f"[INFO] Imágenes encontradas: {len(image_paths)}")

    rows = []
    for img_path in tqdm(image_paths, desc="Generando captions"):
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[WARN] No se pudo abrir {img_path}: {e}")
            continue

        image_input = preprocess(image).unsqueeze(0).to(device)

        # 5) embedding + prefix
        clip_embed = encode_image_to_embedding(args.encoder, clip_model, image_input)
        with torch.no_grad():
            prefix_embed = mapper.clip_project(clip_embed).view(1, args.prefix_length, -1)

        # 6) generación
        if args.beam_search:
            caption = generate_beam(
                mapper,
                tokenizer,
                embed=prefix_embed,
                entry_length=args.entry_length,
            )[0]
        else:
            # FIX para evitar cur_tokens=None dentro de generate2()
            caption = generate2(
                mapper,
                tokenizer,
                prompt=" ",
                embed=prefix_embed,
                entry_count=1,
                entry_length=args.entry_length,
                top_p=args.top_p,
                temperature=args.temperature,
                stop_token=args.stop_token,
            )

        rows.append({"image_path": img_path, "generated_caption": caption})

    # 7) Guardar CSV
    out_dir = os.path.dirname(args.output_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "generated_caption"])
        w.writeheader()
        w.writerows(rows)

    print("[OK] CSV guardado en:", args.output_csv)
    print("[OK] Total filas:", len(rows))


if __name__ == "__main__":
    main()

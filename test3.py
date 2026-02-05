import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import re

import torch
import clip
from transformers import GPT2Tokenizer
import matplotlib.cm as cm

from train import ClipCaptionPrefix, MappingType
from predict import generate2, generate_beam

RESULTS_ROOT = "/data/jefelitman_pupils/Resultados_Explicabilidad_Join"
RUN_SUBDIR = "GradCAM_CLIPCAP"

MORPH = ["sessile", "flat", "elevated", "mucosal", "pedunculated", "flat elevated"]
LESIONS = ["polyp", "adenoma", "hyperplastic"]
LOCATIONS = [
    "cecum",
    "rectum",
    "sigmoid colon",
    "descending colon",
    "transverse colon",
    "ascending colon",
    "colon",
]
SIZES = [f"{i} mm" for i in range(1, 21)]

# Patrones para detectar captions NEGATIVOS (sin pólipos)
NEGATIVE_PATTERNS = [
    r"\bno\s+polyps?\b",
    r"\bno\s+visible\s+polyps?\b",
    r"\bno\s+evidence\s+of\s+polyps?\b",
    r"\bwithout\s+polyps?\b",
]


def extract_size(text):
    m = re.search(r"(\d+)\s*mm", text)
    if not m:
        return None
    size_str = f"{m.group(1)} mm"
    return size_str if size_str in SIZES else None


def extract_clinical_keywords(caption):
    if not isinstance(caption, str):
        return []
    text = caption.lower()
    for pat in NEGATIVE_PATTERNS:
        if re.search(pat, text):
            return ["no polyps"]
    kws = []

    for m in MORPH:
        if m in text:
            kws.append(m)

    for l in LESIONS:
        pattern = rf"\b{re.escape(l)}s?\b"
        if re.search(pattern, text):
            kws.append(l)

    for loc in LOCATIONS:
        if loc in text:
            kws.append(loc)

    sz = extract_size(text)
    if sz:
        kws.append(sz)

    seen = set()
    out = []
    for k in kws:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class CLIPGradCAM:
    def __init__(self, clip_model, device):
        self.model = clip_model
        self.device = device
        self.activations = None
        self.gradients = None


        target_layer = self.model.visual.conv1

        def forward_hook(module, inp, out):
            self.activations = out

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, pil_image, text, preprocess):
        self.model.eval()

        # Preprocesar imagen y tokenizar texto
        image_t = preprocess(pil_image).unsqueeze(0).to(self.device)
        text_t = clip.tokenize([text]).to(self.device)

        # Paso 1: forward sin gradiente para obtener txt_feat
        self.model.zero_grad()
        with torch.no_grad():
            img_feat = self.model.encode_image(image_t)
            txt_feat = self.model.encode_text(text_t)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sim = (img_feat @ txt_feat.T)[0, 0]

        # Paso 2: forward con gradiente sólo para la imagen
        self.model.zero_grad()
        image_t = image_t.requires_grad_(True)
        img_feat = self.model.encode_image(image_t)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        sim = (img_feat @ txt_feat.T)[0, 0]
        sim.backward()

        activations = self.activations
        gradients = self.gradients

        # Pesos = promedio espacial del gradiente
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Redimensionar CAM al tamaño original de la imagen
        H, W = pil_image.size[1], pil_image.size[0]
        cam_img = Image.fromarray(np.uint8(cam * 255))
        cam_img = cam_img.resize((W, H), resample=Image.BILINEAR)
        cam_arr = np.array(cam_img).astype(np.float32) / 255.0

        # Aplicar colormap
        heatmap = cm.get_cmap("jet")(cam_arr)
        heatmap = heatmap[..., :3].astype(np.float32)

        base = np.array(pil_image).astype(np.float32) / 255.0
        alpha = 0.5
        overlay = (1 - alpha) * base + alpha * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))

        return overlay_img, sim.item()


def load_mapper(checkpoint_path, prefix_length=10, mapping_type="transformer", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if mapping_type.lower() == "transformer":
        m_type = MappingType.Transformer
    else:
        m_type = MappingType.MLP

    model = ClipCaptionPrefix(
        prefix_length=prefix_length,
        clip_length=prefix_length,
        prefix_size=512,
        num_layers=8,
        mapping_type=m_type,
    )

    sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(sd)
    model = model.to(device).eval()
    return model


def load_clip_vit(device):
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    return clip_model, preprocess


def iter_images(root_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname.lower())[1]
            if ext in exts:
                yield os.path.join(dirpath, fname)


def main(images_root, checkpoint, prefix_length, mapping_type, use_beam_search):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(RESULTS_ROOT, RUN_SUBDIR)
    os.makedirs(run_dir, exist_ok=True)

    # Mapper (ClipCap)
    mapper = load_mapper(
        checkpoint_path=checkpoint,
        prefix_length=prefix_length,
        mapping_type=mapping_type,
        device=device,
    )

    # CLIP visual + preprocess
    clip_model, preprocess = load_clip_vit(device)

    # Tokenizer GPT-2
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Helper GradCAM
    gradcam_helper = CLIPGradCAM(clip_model=clip_model, device=device)

    # Imágenes a procesar
    image_paths = sorted(list(iter_images(images_root)))

    # CSV de salida
    csv_path = os.path.join(run_dir, "explicabilidad_clipcap.csv")
    rows = []

    base_fields = ["image_name", "generated_caption", "keywords"]
    extra_sim_fields = set()

    for img_path in tqdm(image_paths, desc="Generando captions + GradCAM"):
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        # Embedding CLIP de la imagen
        image_input = preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            clip_embed = clip_model.encode_image(image_input).float()
            clip_embed = clip_embed / clip_embed.norm(dim=-1, keepdim=True)
            prefix_embed = mapper.clip_project(clip_embed).view(1, prefix_length, -1)

        # Generar caption
        if use_beam_search:
            caption = generate_beam(
                model=mapper,
                tokenizer=tokenizer,
                embed=prefix_embed,
                entry_length=67,
            )[0]
        else:
            caption = generate2(
                mapper,
                tokenizer,
                embed=prefix_embed,
                entry_count=1,
                entry_length=67,
                top_p=0.8,
                temperature=1.0,
                stop_token=".",
            )

        # Extraer keywords clínicos (con manejo de 'no polyps')
        keywords = extract_clinical_keywords(caption)

        base_name = os.path.splitext(os.path.basename(img_path))[0]
        row = {
            "image_name": base_name,
            "generated_caption": caption,
            "keywords": ";".join(keywords),
        }

        # GradCAM por keyword
        for kw in keywords:
            safe_kw = kw.replace(" ", "_")
            overlay_img, sim_kw = gradcam_helper.generate(
                pil_image=pil_img,
                text=kw,
                preprocess=preprocess,
            )
            sim_key = f"sim_{safe_kw}"
            row[sim_key] = sim_kw
            extra_sim_fields.add(sim_key)

            out_img_name = f"{base_name}_{safe_kw}_gradcam.png"
            out_img_path = os.path.join(run_dir, out_img_name)
            overlay_img.save(out_img_path)

        rows.append(row)

    # Alinear columnas del CSV
    fieldnames = base_fields + sorted(extra_sim_fields)
    for r in rows:
        for fn in fieldnames:
            if fn not in r:
                r[fn] = ""

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--mapping_type", type=str, default="transformer")
    parser.add_argument("--beam_search", action="store_true")

    args = parser.parse_args()

    main(
        images_root=args.images_root,
        checkpoint=args.checkpoint,
        prefix_length=args.prefix_length,
        mapping_type=args.mapping_type,
        use_beam_search=args.beam_search,
    )

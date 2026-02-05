import os
import sys
import pickle
import argparse

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd



def load_biomedclip_openclip_with_local_pt(
    weights_path: str,
    device: str = "cuda",
    model_id: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
):
    """
    - Construye la arquitectura correcta de BioMedCLIP vía open_clip (hf-hub).
    - Carga pesos locales desde un .pt (state_dict OrderedDict) como el tuyo.
    """
    import open_clip

    print(f"[INFO] Model ID (arquitectura): {model_id}")
    print(f"[INFO] Cargando pesos locales: {weights_path}")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No se encontraron los pesos en: {weights_path}")

    # 1) Crear modelo + preprocess (esto puede requerir acceso a HF SOLO para la config)
    #    Si tu entorno tiene internet, funciona directo.
    #    Si estás offline, ver nota al final.
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id,
        pretrained=None,   # NO descargar pesos; los metemos nosotros
        device=device
    )
    model.eval()

    # 2) Cargar state_dict local (.pt)
    sd = torch.load(weights_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise ValueError("El .pt no parece ser un state_dict (dict/OrderedDict).")

    # Limpieza de prefijos comunes (por si acaso)
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        if k2.startswith("model."):
            k2 = k2[len("model."):]
        new_sd[k2] = v

    # Para BioMedCLIP OpenCLIP normalmente debería calzar casi perfecto.
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    print("[INFO] load_state_dict(strict=False) OK")
    print(f"[INFO] Missing keys: {len(missing)}")
    print(f"[INFO] Unexpected keys: {len(unexpected)}")
    if len(missing) > 0:
        print("[INFO] Ejemplo missing:", missing[:10])
    if len(unexpected) > 0:
        print("[INFO] Ejemplo unexpected:", unexpected[:10])

    # Si te quedan MUCHOS missing/unexpected aquí, el model_id no corresponde a tus pesos.
    return model, preprocess


def resolve_image_path(img_path: str, images_root: str) -> str:
    """
    Normaliza y resuelve la ruta final.
    - strip espacios
    - convierte backslashes
    - si ya es absoluta, la respeta
    - si es relativa, la une a images_root
    """
    if img_path is None:
        return ""

    p = str(img_path).strip().replace("\\", "/")

    # a veces en CSV viene con comillas raras
    p = p.strip('"').strip("'")

    if os.path.isabs(p):
        return p

    return os.path.join(images_root, p)


def main(csv_path: str, images_root: str, out_path: str, weights_path: str, model_id: str, batch_size: int = 64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch.__version__: {torch.__version__}")
    print(f"[INFO] torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"[INFO] Device de trabajo: {device}")

    # 1) Cargar modelo
    print("\n==== CARGANDO MODELO BIOMEDCLIP (OPENCLIP + PT LOCAL) ====")
    model, preprocess = load_biomedclip_openclip_with_local_pt(
        weights_path=weights_path,
        device=device,
        model_id=model_id,
    )
    print("[INFO] Modelo BioMedCLIP listo.\n")

    # 2) Leer CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[INFO] Total filas en CSV: {len(df)}")
    print(f"[INFO] Columnas CSV: {list(df.columns)}")

    if "image_path" not in df.columns:
        raise ValueError("El CSV DEBE tener una columna llamada 'image_path'.")
    if "caption" not in df.columns:
        raise ValueError("El CSV DEBE tener una columna llamada 'caption'.")

    # Debug: mostrar ejemplos de image_path
    print("\n[DEBUG] Ejemplos image_path (primeras 5 filas):")
    for k in range(min(5, len(df))):
        rawp = df.iloc[k]["image_path"]
        print(f"  - raw: {rawp}")
        print(f"    resolved: {resolve_image_path(rawp, images_root)}")

    # 3) Filtrar paths existentes
    paths = []
    caps = []
    bad = 0

    for _, row in df.iterrows():
        full_path = resolve_image_path(row["image_path"], images_root)
        cap = str(row["caption"])

        if os.path.isfile(full_path):
            paths.append(full_path)
            caps.append(cap)
        else:
            bad += 1

    print(f"\n[INFO] Paths válidos: {len(paths)} | inválidos: {bad}")
    if len(paths) == 0:
        print("[ERROR] No hay imágenes válidas.")
        print("[TIP] Revisa: (1) images_root, (2) qué trae image_path en el CSV, (3) si los archivos existen en disco.")
        return

    # 4) Encode por batches
    all_embeddings = []
    all_captions = []
    idx_global = 0

    print("\n==== GENERANDO EMBEDDINGS ====")

    for start in tqdm(range(0, len(paths), batch_size), total=(len(paths) + batch_size - 1) // batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_caps = caps[start : start + batch_size]

        batch_imgs = []
        ok_indices = []

        for j, pth in enumerate(batch_paths):
            try:
                img = Image.open(pth).convert("RGB")
                batch_imgs.append(preprocess(img))
                ok_indices.append(j)
            except Exception as e:
                # omitimos esa imagen
                continue

        if len(batch_imgs) == 0:
            continue

        images = torch.stack(batch_imgs, dim=0).to(device)

        with torch.no_grad():
            feat = model.encode_image(images)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            feat = feat.cpu()

        all_embeddings.append(feat)

        for k_local, j in enumerate(ok_indices):
            all_captions.append(
                {
                    "caption": batch_caps[j],
                    "image_id": idx_global,
                    "clip_embedding": idx_global,
                }
            )
            idx_global += 1

    if not all_embeddings:
        print("[ERROR] No se generó ningún embedding (fallaron lecturas o preprocess).")
        return

    clip_embedding = torch.cat(all_embeddings, dim=0)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "wb") as f:
        pickle.dump({"clip_embedding": clip_embedding, "captions": all_captions}, f)

    print("\n==== PROCESO COMPLETADO ====")
    print("[OK] PKL guardado en:", out_path)
    print("[OK] Shape embeddings:", tuple(clip_embedding.shape))
    print("[OK] Num captions:", len(all_captions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        default="experiments_colono/experiments_colono/clipclap_train.csv",
        help="Ruta al CSV (image_path, caption)",
    )
    parser.add_argument(
        "--images_root",
        default="Sun_Multimodal/Train",
        help="Carpeta raíz de las imágenes",
    )
    parser.add_argument(
        "--out_path",
        default="outputs/parse_colono_biomed/colono_biomedclip_train.pkl",
        help="Ruta del PKL de salida",
    )
    parser.add_argument(
        "--weights_path",
        default="clip_weights/biomedclip_weights.pt",
        help="Ruta a biomedclip_weights.pt (state_dict OpenCLIP)",
    )
    parser.add_argument(
        "--model_id",
        default="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", 
        help="ID del modelo para construir la arquitectura (OpenCLIP hf-hub).",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size para encode_image")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        images_root=args.images_root,
        out_path=args.out_path,
        weights_path=args.weights_path,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )

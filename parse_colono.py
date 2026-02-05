import torch
import clip
from PIL import Image
import pickle
from tqdm import tqdm
import os
import argparse
import pandas as pd


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(csv_path: str, images_root: str, out_path: str, clip_model_type: str = "ViT-B/32"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    clip_model = clip_model.eval()

    df = pd.read_csv(csv_path)

    all_embeddings = []
    all_captions = []

    print(f"{len(df)} filas cargadas del CSV")

    for i, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["image_path"] 
        caption = str(row["caption"]) 


        full_path = img_path
        if not os.path.isabs(img_path):
            full_path = os.path.join(images_root, img_path)

        if not os.path.isfile(full_path):
            print(f"No se encontró la imagen: {full_path}")
            continue

        image = Image.open(full_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()  # [1, 512]

        # este índice 'i' sirve como puntero dentro de clip_embedding
        d = {
            "caption": caption,
            "image_id": i,          # puedes guardar aquí también el case si quieres
            "clip_embedding": i,
        }

        all_embeddings.append(prefix)   # lista de [1,512]
        all_captions.append(d)

    if len(all_embeddings) == 0:
        print("No se generó ningún embedding. Revisa rutas y CSV.")
        return

    all_embeddings_tensor = torch.cat(all_embeddings, dim=0)  # [N, 512]
    out_data = {
        "clip_embedding": all_embeddings_tensor,
        "captions": all_captions,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_data, f)

    print("Listo.")
    print("Embeddings guardados en:", out_path)
    print("Total de embeddings:", all_embeddings_tensor.shape[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', required=True, help="CSV con image_path y caption")
    parser.add_argument('--images_root', default=".", help="Raíz de las imágenes si image_path es relativo")
    parser.add_argument('--out_path', default="./data/med/oscar_split_med_train.pkl")
    parser.add_argument('--clip_model_type', default="ViT-B/32",
                        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"))
    args = parser.parse_args()

    main(args.csv_path, args.images_root, args.out_path, args.clip_model_type)

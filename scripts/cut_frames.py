import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEFAULT_SOURCE = Path("igho/frames")
DEFAULT_LEFT_CROP = 400

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta frames PNG en todas las subcarpetas.")
    parser.add_argument(
        "sources",
        nargs="*",
        help="Carpetas raíz que contienen subcarpetas con frames PNG.",
    )
    parser.add_argument(
        "--left_crop",
        type=int,
        default=DEFAULT_LEFT_CROP,
        help="Pixeles a recortar desde la izquierda.",
    )
    return parser.parse_args()

def crop_folder(folder: Path, left_crop: int) -> None:
    files = sorted(folder.glob("*.png"))
    if not files:
        return
    for path in tqdm(files, desc=f"{folder.name}"):
        with Image.open(path) as img:
            width, height = img.size
            cropped = img.crop((left_crop, 0, width, height))
        cropped.save(path)



def crop_all(root: Path, left_crop: int) -> None:
    if not root.exists():
        raise FileNotFoundError(f"No existe: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {root}")

    subfolders = sorted(f for f in root.iterdir() if f.is_dir())
    if not subfolders:
        crop_folder(root, left_crop)
        print(f"✓ {root}")
        return

    for folder in subfolders:
        crop_folder(folder, left_crop)
        print(f"✓ {folder}")

def main() -> None:
    args = parse_args()
    roots = [Path(s) for s in args.sources] if args.sources else [DEFAULT_SOURCE]
    for root in roots:
        crop_all(root, args.left_crop)

if __name__ == "__main__":
    main()
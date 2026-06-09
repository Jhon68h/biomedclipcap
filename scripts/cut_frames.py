import argparse
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DEFAULT_SOURCE = Path("igho/frames/2025-05-16_101431_209")
DEFAULT_DESTINATION = Path("igho/cut_frames/video_5")
DEFAULT_OUTPUT_ROOT = Path("igho/cut_frames")
DEFAULT_LEFT_CROP = 400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recorta frames PNG quitando pixeles desde la izquierda.")
    parser.add_argument(
        "sources",
        nargs="*",
        help="Carpetas con frames PNG. Si no se pasan, usa el origen hardcodeado original.",
    )
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Carpeta base donde guardar los recortes cuando se pasan varios sources.",
    )
    parser.add_argument(
        "--left_crop",
        type=int,
        default=DEFAULT_LEFT_CROP,
        help="Pixeles a recortar desde la izquierda.",
    )
    return parser.parse_args()


def crop_frames(source: Path, destination: Path, left_crop: int) -> None:
    if not source.exists():
        raise FileNotFoundError(f"No existe la carpeta de frames: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"No es una carpeta: {source}")

    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.png"))

    for path in tqdm(files, desc=f"Recortando {source.name}"):
        output_path = destination / path.name
        if output_path.exists():
            continue

        with Image.open(path) as img:
            width, height = img.size
            cropped = img.crop((left_crop, 0, width, height))
            cropped.save(output_path)


def main() -> None:
    args = parse_args()

    if args.sources:
        output_root = Path(args.output_root)
        for source_text in args.sources:
            source = Path(source_text)
            destination = output_root / source.name
            crop_frames(source, destination, args.left_crop)
            print(f"{source} -> {destination}")
    else:
        crop_frames(DEFAULT_SOURCE, DEFAULT_DESTINATION, args.left_crop)
        print(f"{DEFAULT_SOURCE} -> {DEFAULT_DESTINATION}")


if __name__ == "__main__":
    main()

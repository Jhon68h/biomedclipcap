import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import cv2


SOURCE_FOLDER = Path("igho")
FRAMES_FOLDER = SOURCE_FOLDER / "frames"
FRAME_PATTERN = "frame_{:04d}.png"
FFMPEG_FRAME_PATTERN = "frame_%04d.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae frames PNG desde videos AVI.")
    parser.add_argument(
        "videos",
        nargs="*",
        help="Rutas a videos .avi. Si no se pasan, procesa todos los .avi en igho/.",
    )
    return parser.parse_args()


def count_frames_with_ffprobe(video_path: Path) -> Optional[int]:
    if not shutil.which("ffprobe"):
        return None

    commands = [
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ],
    ]

    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            text = line.strip()
            if text.isdigit() and int(text) > 0:
                return int(text)
    return None


def print_progress(current: int, total: Optional[int]) -> None:
    if total:
        width = 30
        filled = min(width, int(width * current / total))
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{bar}] {current}/{total} frames", end="", flush=True)
    else:
        print(f"\rFrames extraidos: {current}", end="", flush=True)

def extract_frames_ffmpeg(video_path: Path, output_folder: Path) -> int:
    total_frames = count_frames_with_ffprobe(video_path)
    frame_pattern = output_folder / FFMPEG_FRAME_PATTERN
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+discardcorrupt",
        "-err_detect",
        "ignore_err",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(frame_pattern),
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    last_count = -1
    while process.poll() is None:
        frame_count = len(list(output_folder.glob("frame_*.png")))
        if frame_count != last_count:
            print_progress(frame_count, total_frames)
            last_count = frame_count
        time.sleep(0.5)

    return_code = process.wait()
    frame_count = len(list(output_folder.glob("frame_*.png")))
    print_progress(frame_count, total_frames)
    print()

    if return_code != 0 and frame_count == 0:
        raise RuntimeError(f"ffmpeg no pudo extraer frames de: {video_path}")
    if return_code != 0:
        print(
            f"Advertencia: ffmpeg reporto errores de decodificacion en {video_path}, "
            f"pero se extrajeron {frame_count} frames."
        )
    return frame_count


def extract_frames(video_path: Path, output_folder: Path) -> int:
    output_folder.mkdir(parents=True, exist_ok=True)
    for old_frame in output_folder.glob("frame_*.png"):
        old_frame.unlink()

    if shutil.which("ffmpeg"):
        return extract_frames_ffmpeg(video_path, output_folder)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = None

    saved_frames = 0
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        filename = output_folder / FRAME_PATTERN.format(frame_index)
        cv2.imwrite(str(filename), frame)
        saved_frames += 1
        frame_index += 1
        print_progress(saved_frames, total_frames)

    cap.release()
    print()
    return saved_frames


def main() -> None:
    args = parse_args()
    if args.videos:
        avi_files = [Path(video_path) for video_path in args.videos]
    else:
        avi_files = sorted(SOURCE_FOLDER.glob("*.avi"))
    if not avi_files:
        print(f"No se encontraron archivos .avi en {SOURCE_FOLDER}")
        return

    FRAMES_FOLDER.mkdir(parents=True, exist_ok=True)

    for video_path in avi_files:

        output_folder = FRAMES_FOLDER / video_path.stem
        frame_count = extract_frames(video_path, output_folder)
        print(f"{video_path} -> {output_folder} ({frame_count} frames)")


if __name__ == "__main__":
    main()

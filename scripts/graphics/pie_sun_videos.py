#!/usr/bin/env python3
"""Distribucion de videos SUN usados en el entrenamiento 2-fold.

Tres categorias: adenoma, hiperplastico y sin polipo. La unidad es el VIDEO
(caso), no el frame. Los videos negativos de SUN tienen numeracion propia,
independiente de la de las 100 lesiones, asi que se prefijan con 'neg_' para
evitar la colision de identificadores documentada en
reportes/reporte_sun_database.md (7.1).

Salida: scripts/graphics/pie_sun_videos.{png,pdf}
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POSITIVE_CSV = (
    REPO_ROOT / "experiments_colono/experiments_colono/clipclap_train_labeled_positive.csv"
)
DEFAULT_NEGATIVE_CSV = (
    REPO_ROOT / "experiments_colono/experiments_colono/clipclap_train_labeled_negative.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scripts" / "graphics"

# Slots categoricos 1-3 de la paleta de referencia (validados all-pairs).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES = {
    "adenoma": "#2a78d6",
    "hyperplastic": "#eb6834",
    "negative": "#1baf7a",
}
LABELS = {
    "adenoma": "Adenoma",
    "hyperplastic": "Hiperplásico",
    "negative": "Sin pólipo",
}
ORDER = ("adenoma", "hyperplastic", "negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive_csv", type=Path, default=DEFAULT_POSITIVE_CSV)
    parser.add_argument("--negative_csv", type=Path, default=DEFAULT_NEGATIVE_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="pie_sun_videos")
    return parser.parse_args()


def count_videos(positive_csv: Path, negative_csv: Path) -> dict[str, int]:
    """Cuenta videos unicos por categoria a partir de los CSV del pipeline."""
    polyp_type_by_case: dict[str, str] = {}
    with positive_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            polyp_type_by_case[row["case"].strip()] = row["polyp_type"].strip()

    negative_cases: set[str] = set()
    with negative_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            negative_cases.add("neg_" + row["case"].strip())

    counts = {key: 0 for key in ORDER}
    for polyp_type in polyp_type_by_case.values():
        if polyp_type not in counts:
            raise ValueError(f"polyp_type inesperado en el CSV positivo: {polyp_type!r}")
        counts[polyp_type] += 1
    counts["negative"] = len(negative_cases)
    return counts


def build_figure(counts: dict[str, int]):
    total = sum(counts.values())
    values = [counts[key] for key in ORDER]
    colors = [SERIES[key] for key in ORDER]

    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=300)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    wedges, _ = ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        # 2px de superficie entre segmentos, no un borde decorativo.
        wedgeprops={"edgecolor": SURFACE, "linewidth": 2, "antialiased": True},
    )

    # Etiquetas directas: cada porcion lleva nombre y valor, nunca solo color.
    for wedge, key, value in zip(wedges, ORDER, values):
        angle = (wedge.theta1 + wedge.theta2) / 2.0
        share = 100.0 * value / total
        text = f"{LABELS[key]}\n{value} vídeos · {share:.1f}%"
        rad = math.radians(angle)
        x, y = math.cos(rad), math.sin(rad)
        if share >= 25:
            ax.text(
                0.54 * x,
                0.54 * y,
                text,
                ha="center",
                va="center",
                fontsize=12.5,
                color=SURFACE,
                linespacing=1.45,
            )
        else:
            # Porciones pequenas: etiqueta fuera con linea guia.
            ax.annotate(
                text,
                xy=(0.95 * x, 0.95 * y),
                xytext=(1.38 * x, 1.30 * y),
                ha="left" if x >= 0 else "right",
                va="center",
                fontsize=12.5,
                color=TEXT_PRIMARY,
                linespacing=1.45,
                arrowprops={
                    "arrowstyle": "-",
                    "color": TEXT_SECONDARY,
                    "linewidth": 1.0,
                    "shrinkA": 2,
                    "shrinkB": 4,
                },
            )
    ax.text(
        0.5,
        -0.02,
        f"n = {total} vídeos",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11.5,
        color=TEXT_SECONDARY,
    )

    ax.legend(
        wedges,
        [LABELS[key] for key in ORDER],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=12,
        labelcolor=TEXT_PRIMARY,
        handlelength=1.1,
        handleheight=1.1,
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.85, 1.55)
    ax.set_ylim(-1.08, 1.30)
    fig.tight_layout()
    return fig


def main() -> int:
    args = parse_args()
    counts = count_videos(args.positive_csv, args.negative_csv)
    total = sum(counts.values())

    print("Distribucion de videos SUN (train 2-fold)")
    for key in ORDER:
        print(f"  {LABELS[key]:<14} {counts[key]:>3}  ({100.0 * counts[key] / total:5.1f} %)")
    print(f"  {'TOTAL':<14} {total:>3}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure(counts)
    for ext in ("png", "pdf"):
        path = args.output_dir / f"{args.basename}.{ext}"
        fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
        print(f"escrito: {path.relative_to(REPO_ROOT)}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

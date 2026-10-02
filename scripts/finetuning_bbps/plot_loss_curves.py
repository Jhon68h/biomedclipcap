#!/usr/bin/env python3
"""Curvas de perdida de entrenamiento y validacion de los dos fine-tunings IGHO.

Lee los CSV que deja `train.py` en <fold>/train/:
  - bbps_fold_N_loss_per_epoch.csv  (epoch, mean_loss)      -> train
  - val_loss_per_epoch.csv          (epoch, val_loss, ...)  -> validacion
  - bbps_fold_N_loss_per_step.csv   (global_step, loss)     -> train por paso

Genera dos figuras en reportes/figuras/:
  loss_epoch_train_val.png : small multiples 2x2 (experimento x fold), eje comun
  loss_step_train.png      : perdida por paso, escala log, un panel por experimento

Un solo eje y por panel: nunca doble eje. Los dos experimentos comparten limites
para que la comparacion visual sea honesta.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reportes" / "figuras"

# Paleta de referencia dataviz, slots 1 y 2 (validados para pares adyacentes).
C_TRAIN = "#2a78d6"   # azul
C_VAL = "#eb6834"     # naranja
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d8d4"

EXPERIMENTOS = [
    ("igho_training", "BBPS generado"),
    ("igho_bbps_real", "BBPS GT del especialista"),
]
FOLDS = ["fold_1", "fold_2"]


def read_csv(path: Path):
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_epoch(root: str, fold: str):
    base = REPO / root / "folds" / fold / "train"
    n = fold.split("_")[1]
    tr = read_csv(base / f"bbps_fold_{n}_loss_per_epoch.csv")
    va = read_csv(base / "val_loss_per_epoch.csv")
    return (
        [int(r["epoch"]) for r in tr], [float(r["mean_loss"]) for r in tr],
        [int(r["epoch"]) for r in va], [float(r["val_loss"]) for r in va],
    )


def load_steps(root: str, fold: str):
    base = REPO / root / "folds" / fold / "train"
    n = fold.split("_")[1]
    rows = read_csv(base / f"bbps_fold_{n}_loss_per_step.csv")
    return [int(r["global_step"]) for r in rows], [float(r["loss"]) for r in rows]


def ema(values, alpha=0.02):
    out, acc = [], values[0]
    for v in values:
        acc = alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


# ------------------------------------------------------ figura 1: por epoca
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
fig.patch.set_facecolor(SURFACE)

for i, (root, titulo) in enumerate(EXPERIMENTOS):
    for j, fold in enumerate(FOLDS):
        ax = axes[i][j]
        style(ax)
        te, tl, ve, vl = load_epoch(root, fold)
        ax.plot(te, tl, color=C_TRAIN, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="entrenamiento", zorder=3)
        ax.plot(ve, vl, color=C_VAL, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label="validacion", zorder=3)

        # epoca de menor val_loss: es la que se deberia elegir como checkpoint
        best = min(range(len(vl)), key=lambda k: vl[k])
        ax.scatter([ve[best]], [vl[best]], s=170, facecolors="none",
                   edgecolors=C_VAL, linewidths=2, zorder=4)
        dx = (6, 26) if ve[best] < ve[-1] else (-96, 26)
        ax.annotate(f"min val\nep {ve[best]} = {vl[best]:.4f}",
                    (ve[best], vl[best]), textcoords="offset points", xytext=dx,
                    fontsize=8.5, color=INK_2)

        # etiquetas directas al final de cada linea (sin numero en cada punto)
        # si los dos finales casi coinciden, separar las etiquetas en vertical
        juntos = abs(tl[-1] - vl[-1]) < 0.03
        off_t = (9, -13) if juntos else (9, -4)
        off_v = (9, 6) if juntos else (9, -4)
        ax.annotate(f"train {tl[-1]:.3f}", (te[-1], tl[-1]), textcoords="offset points",
                    xytext=off_t, fontsize=8.5, color=C_TRAIN, fontweight="bold")
        ax.annotate(f"val {vl[-1]:.3f}", (ve[-1], vl[-1]), textcoords="offset points",
                    xytext=off_v, fontsize=8.5, color=C_VAL, fontweight="bold")

        ax.set_title(f"{titulo}  -  {fold}", fontsize=10.5, color=INK, loc="left", pad=8)
        ax.set_yscale("log")
        ax.set_yticks([0.12, 0.15, 0.2, 0.3, 0.5, 0.75])
        ax.set_yticklabels(["0.12", "0.15", "0.20", "0.30", "0.50", "0.75"])
        ax.minorticks_off()
        ax.set_xticks(te)
        if i == 1:
            ax.set_xlabel("epoca", fontsize=9.5, color=INK_2)
        if j == 0:
            ax.set_ylabel("perdida (cross-entropy, escala log)", fontsize=9.5, color=INK_2)
        ax.set_xlim(-0.25, 5.35)

axes[0][0].legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="upper right")
fig.suptitle("Perdida de entrenamiento y validacion por epoca  -  fine-tuning IGHO (2-fold)",
             fontsize=13, color=INK, x=0.5, y=0.98)
fig.text(0.5, 0.005,
         "Mismo eje y (log) en los cuatro paneles. El circulo marca la epoca de menor perdida de validacion.",
         ha="center", fontsize=8.5, color=INK_2)
fig.tight_layout(rect=[0, 0.02, 1, 0.955])
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "loss_epoch_train_val.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# ------------------------------------------------------ figura 2: por paso
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
fig.patch.set_facecolor(SURFACE)
COL = {"fold_1": C_TRAIN, "fold_2": C_VAL}

for i, (root, titulo) in enumerate(EXPERIMENTOS):
    ax = axes[i]
    style(ax)
    for fold in FOLDS:
        s, l = load_steps(root, fold)
        ax.plot(s, l, color=COL[fold], linewidth=0.6, alpha=0.18, zorder=2)
        sm = ema(l)
        ax.plot(s, sm, color=COL[fold], linewidth=2, label=fold, zorder=3)
        ax.annotate(fold, (s[-1], sm[-1]), textcoords="offset points", xytext=(-32, -14),
                    fontsize=9, color=COL[fold], fontweight="bold")
    ax.set_yscale("log")
    ax.set_title(titulo, fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("paso de entrenamiento", fontsize=9.5, color=INK_2)
    if i == 0:
        ax.set_ylabel("perdida (escala log)", fontsize=9.5, color=INK_2)

axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="upper right")
fig.suptitle("Perdida de entrenamiento por paso  -  linea tenue: valor crudo; linea gruesa: media movil",
             fontsize=12, color=INK, x=0.5, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "loss_step_train.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

# --------------------------------------- figura 3: comparar solo validacion
# Color = experimento (la entidad); estilo de linea = fold (codificacion secundaria).
fig, ax = plt.subplots(figsize=(7.6, 4.6))
fig.patch.set_facecolor(SURFACE)
style(ax)
COL_EXP = {"igho_training": C_TRAIN, "igho_bbps_real": C_VAL}
DASH = {"fold_1": (None, None), "fold_2": (4, 2)}
for root, titulo in EXPERIMENTOS:
    for fold in FOLDS:
        _, _, ve, vl = load_epoch(root, fold)
        line, = ax.plot(ve, vl, color=COL_EXP[root], linewidth=2, marker="o", markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=f"{titulo} - {fold}", zorder=3)
        if DASH[fold][0]:
            line.set_dashes(DASH[fold])
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xlabel("epoca", fontsize=9.5, color=INK_2)
ax.set_ylabel("perdida de validacion", fontsize=9.5, color=INK_2)
ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right",
          handlelength=3.4, handletextpad=0.8)
ax.set_title("Perdida de validacion: los cuatro entrenamientos", fontsize=12, color=INK, loc="left", pad=10)
fig.tight_layout()
fig.savefig(OUT / "loss_val_comparacion.png", dpi=200, facecolor=SURFACE)
plt.close(fig)

print(f"[OK] {OUT / 'loss_epoch_train_val.png'}")
print(f"[OK] {OUT / 'loss_step_train.png'}")
print(f"[OK] {OUT / 'loss_val_comparacion.png'}")

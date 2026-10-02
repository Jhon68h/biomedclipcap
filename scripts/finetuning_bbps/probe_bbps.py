#!/usr/bin/env python3
"""Linear probe: ¿los embeddings de BiomedCLIP contienen la senal del BBPS?

Motivo
------
El fine-tuning de ClipCap no aprendio el BBPS: cada fold emite exactamente las dos
clases mas frecuentes de su propio entrenamiento y la correlacion con el GT es
r = -0.02 sobre 8.948 frames. Eso deja dos explicaciones posibles:

  (a) La senal ESTA en el embedding pero el decoder autoregresivo no la usa
      -> tendria sentido tocar la arquitectura.
  (b) La senal NO ESTA en el embedding (o no es extraible con n=15 videos)
      -> ninguna arquitectura lo va a arreglar y hay que declararlo como
         limitacion de los datos.

Este script distingue entre las dos. Ajusta una regresion ridge directamente sobre
los embeddings ya calculados, sin GPT-2 de por medio. Si un modelo lineal tampoco
puede, es (b).

Diseno
------
* **Unidad = video (n=15)**, igual que la metrica de BBPS. Se entrena por frame pero
  se evalua consolidando por mediana y comparando una vez por video.
* **Leave-one-video-out**: cada video lo predice un modelo que no lo vio nunca. Con
  15 videos es lo que mas datos deja por fold (14 de entrenamiento).
* **Peso por video**: cada frame pesa 1/n_frames_de_su_video, para que un video con
  1.200 frames no valga 3 veces mas que uno con 400.
* **Barrido de lambda y se reporta el MEJOR**: es deliberadamente optimista. El
  numero resultante es una COTA SUPERIOR de lo decodificable. Si ni siquiera esa
  cota baja del baseline trivial, la conclusion (b) es solida.
* **Control por permutacion**: se baraja la asignacion video->BBPS y se repite todo.
  Da un p-valor empirico: que fraccion de las permutaciones al azar consigue un MAE
  igual o mejor que el real.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PAIRS = [
    "weights/igho_bbps_real/group_a.pkl=igho_training/folds/fold_1/train_meta.csv",
    "weights/igho_bbps_real/group_b.pkl=igho_training/folds/fold_1/val_meta.csv",
]
BBPS_RE = re.compile(r"BBPS\s+is\s+(\d+)\s*/\s*9", re.IGNORECASE)


def resolve(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_pair(pkl_path: Path, meta_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Carga embeddings + metadatos y verifica que van fila a fila.

    parse_colono_biomed.py escribe las filas en el mismo orden del CSV, saltandose las
    imagenes que no existen. Como en esta corrida hubo 0 invalidos, la correspondencia
    es 1:1 -- pero se comprueba comparando el BBPS de la caption del pkl contra la
    columna del meta, porque si el CSV se reconstruyo despues del pkl el orden cambia y
    todo el probe daria ruido sin avisar.
    """
    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)
    emb = data["clip_embedding"]
    emb = emb.numpy() if hasattr(emb, "numpy") else np.asarray(emb)
    caps = [c["caption"] for c in data["captions"]]

    with meta_path.open("r", encoding="utf-8-sig", newline="") as handle:
        meta = list(csv.DictReader(handle))

    if len(caps) != len(meta) or emb.shape[0] != len(meta):
        raise SystemExit(
            f"[ERROR] {pkl_path.name} tiene {emb.shape[0]} embeddings / {len(caps)} captions "
            f"pero {meta_path.name} tiene {len(meta)} filas.\n"
            "        El pkl y el CSV son de builds distintos. Regenera los embeddings."
        )

    y, vids = [], []
    for i, (cap, row) in enumerate(zip(caps, meta)):
        m = BBPS_RE.search(cap)
        if m is None:
            raise SystemExit(f"[ERROR] Caption sin BBPS en {pkl_path.name} fila {i}: {cap!r}")
        if int(m.group(1)) != int(row["bbps"]):
            raise SystemExit(
                f"[ERROR] Desajuste en la fila {i}: el pkl dice BBPS {m.group(1)} y "
                f"{meta_path.name} dice {row['bbps']}.\n"
                "        El pkl y el CSV no corresponden. Regenera los embeddings."
            )
        y.append(float(row["bbps"]))
        vids.append(row["video_id"])
    return emb.astype(np.float64), np.asarray(y), vids


def ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float) -> np.ndarray:
    """Ridge con pesos y termino independiente (el intercept no se regulariza)."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    Xw = Xa * w[:, None]
    A = Xa.T @ Xw
    reg = np.eye(Xa.shape[1]) * lam
    reg[-1, -1] = 0.0
    A += reg
    b = Xw.T @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def predict(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((X.shape[0], 1))]) @ coef


def leave_one_video_out(
    X: np.ndarray, y: np.ndarray, vids: np.ndarray, videos: Sequence[str], lam: float
) -> Dict[str, float]:
    """Devuelve {video: prediccion consolidada por mediana}."""
    out: Dict[str, float] = {}
    for held in videos:
        tr = vids != held
        te = ~tr
        # Cada video pesa lo mismo, independientemente de cuantos frames aporte.
        w = np.zeros(tr.sum())
        vtr = vids[tr]
        for v in np.unique(vtr):
            sel = vtr == v
            w[sel] = 1.0 / sel.sum()
        coef = ridge_fit(X[tr], y[tr], w, lam)
        out[held] = float(np.median(predict(X[te], coef)))
    return out


def mae_by_video(pred: Dict[str, float], gt: Dict[str, float]) -> float:
    return float(np.mean([abs(pred[v] - gt[v]) for v in gt]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe de BBPS sobre los embeddings de BiomedCLIP.")
    p.add_argument("--pair", nargs="*", default=DEFAULT_PAIRS, metavar="PKL=META_CSV",
                   help="Pares pkl=meta_csv. Por defecto group_a/group_b del fold_1.")
    p.add_argument("--lambdas", nargs="*", type=float,
                   default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    p.add_argument("--permutations", type=int, default=200,
                   help="Permutaciones del control. 0 lo desactiva.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default="igho_training/eval/probe_bbps.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    Xs, ys, vs = [], [], []
    for item in args.pair:
        if "=" not in item:
            raise SystemExit(f"[ERROR] --pair espera pkl=meta_csv, recibido: {item!r}")
        pkl_s, meta_s = item.split("=", 1)
        X, y, v = load_pair(resolve(pkl_s.strip()), resolve(meta_s.strip()))
        print(f"[INFO] {Path(pkl_s).name}: {X.shape[0]} frames, dim {X.shape[1]}")
        Xs.append(X); ys.append(y); vs.extend(v)

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    vids = np.asarray(vs)
    videos = sorted(set(vs))
    gt = {v: float(y[vids == v][0]) for v in videos}

    print(f"[INFO] Total: {X.shape[0]} frames | {len(videos)} videos | BBPS {sorted(set(gt.values()))}")

    # Baseline trivial: la mediana del BBPS de los videos de ENTRENAMIENTO de cada fold
    # LOVO (no la global, que usaria la etiqueta del video evaluado).
    base_pred = {v: float(np.median([gt[o] for o in videos if o != v])) for v in videos}
    baseline = mae_by_video(base_pred, gt)
    print(f"[INFO] Baseline trivial (mediana de los otros 14): MAE = {baseline:.4f}\n")

    print(f"{'lambda':>10} {'MAE por video':>14} {'r Pearson':>11}")
    print("-" * 38)
    results = {}
    for lam in args.lambdas:
        pred = leave_one_video_out(X, y, vids, videos, lam)
        m = mae_by_video(pred, gt)
        g = np.array([gt[v] for v in videos]); p = np.array([pred[v] for v in videos])
        r = float(np.corrcoef(g, p)[0, 1]) if p.std() > 1e-12 else float("nan")
        results[lam] = {"mae": m, "r": r, "pred": pred}
        print(f"{lam:>10g} {m:>14.4f} {r:>11.3f}")

    best_lam = min(results, key=lambda k: results[k]["mae"])
    best = results[best_lam]
    print("-" * 38)
    print(f"MEJOR: lambda={best_lam:g}  MAE={best['mae']:.4f}  (baseline {baseline:.4f})")
    print("Nota: elegir lambda por el MAE de test es optimista a proposito -- este numero")
    print("      es una COTA SUPERIOR de lo que se puede decodificar linealmente.\n")

    print(f"{'video':26} {'GT':>3} {'probe':>7} {'|err|':>7}")
    print("-" * 46)
    for v in sorted(videos, key=lambda x: gt[x]):
        print(f"{v:26} {gt[v]:>3.0f} {best['pred'][v]:>7.2f} {abs(best['pred'][v]-gt[v]):>7.2f}")

    # ---- control por permutacion -------------------------------------------------
    # A = Xa.T @ W @ Xa NO depende de las etiquetas, solo de X, los pesos y lambda. La
    # version anterior la reconstruia en cada permutacion (200 x 15 = 3000 sistemas de
    # 513x513 sobre ~8000 filas) y tardaba horas. Aqui se arma una vez por fold y las
    # 200 permutaciones se resuelven de golpe como 200 columnas del lado derecho.
    perm = None
    if args.permutations > 0:
        rng = np.random.default_rng(args.seed)
        labels = np.array([gt[v] for v in videos])
        perms = np.stack([rng.permutation(labels) for _ in range(args.permutations)], axis=1)
        gmaps = [dict(zip(videos, perms[:, j])) for j in range(args.permutations)]
        pred_mat = np.zeros((len(videos), args.permutations))

        for i, held in enumerate(videos):
            tr = vids != held
            te = ~tr
            Xa_tr = np.hstack([X[tr], np.ones((tr.sum(), 1))])
            Xa_te = np.hstack([X[te], np.ones((te.sum(), 1))])
            w = np.zeros(tr.sum())
            vtr = vids[tr]
            for v in np.unique(vtr):
                sel = vtr == v
                w[sel] = 1.0 / sel.sum()
            Xw = Xa_tr * w[:, None]
            A = Xa_tr.T @ Xw
            reg = np.eye(Xa_tr.shape[1]) * best_lam
            reg[-1, -1] = 0.0
            A += reg
            # Y_tr: (n_train, n_perm) con las etiquetas barajadas de cada permutacion.
            Y_tr = np.stack([np.array([g[v] for v in vtr]) for g in gmaps], axis=1)
            B = Xw.T @ Y_tr
            try:
                coefs = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                coefs = np.linalg.lstsq(A, B, rcond=None)[0]
            pred_mat[i, :] = np.median(Xa_te @ coefs, axis=0)

        maes = np.array([
            float(np.mean([abs(pred_mat[i, j] - gmaps[j][videos[i]]) for i in range(len(videos))]))
            for j in range(args.permutations)
        ])
        pval = float((maes <= best["mae"]).mean())
        perm = {"n": args.permutations, "mae_mean": float(maes.mean()),
                "mae_p05": float(np.percentile(maes, 5)), "p_value": pval}
        print("\n" + "=" * 46)
        print(f"Control por permutacion ({args.permutations} barajados de las etiquetas):")
        print(f"  MAE medio al azar : {maes.mean():.4f}")
        print(f"  percentil 5       : {np.percentile(maes, 5):.4f}")
        print(f"  MAE real          : {best['mae']:.4f}")
        print(f"  p-valor empirico  : {pval:.3f}")
        if pval > 0.05:
            print("  -> El probe NO se distingue del azar: la senal de BBPS no es")
            print("     linealmente extraible de estos embeddings con n=15 videos.")
        else:
            print("  -> Hay senal por encima del azar: el embedding SI codifica algo del")
            print("     BBPS y el fallo esta en el decoder, no en la representacion.")

    out = resolve(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    out.write_text(json.dumps({
        "n_frames": int(X.shape[0]), "n_videos": len(videos),
        "gt": gt, "baseline_mae": baseline,
        "by_lambda": {str(k): {"mae": v["mae"], "r": v["r"]} for k, v in results.items()},
        "best_lambda": best_lam, "best_mae": best["mae"], "best_r": best["r"],
        "best_pred": best["pred"], "permutation": perm,
    }, indent=2), encoding="utf-8")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()

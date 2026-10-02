#!/usr/bin/env python3
"""Evalua el fine-tuning IGHO+BBPS: BBPS por video (n=15) + metricas de reporte.

Implementa la seccion 3.8 y el paso 7 de `reportes/finetuning_bbps_igho.md`.

Por que el BBPS se mide POR VIDEO y no por frame
------------------------------------------------
El BBPS es un valor constante replicado en todos los frames de un video: hay 15
valores reales, no 136.527. Promediar el error por frame es exactamente la
pseudo-replicacion documentada en `reportes/camino_1_evaluacion_por_lesion.md`
para size/location/paris: subestima el error estandar por ~sqrt(frames_por_video)
y produce intervalos de confianza falsamente estrechos. Aqui las predicciones de
todos los frames de un video se consolidan por MEDIANA y se comparan una sola vez
contra el GT de ese video.

Por que MAE y no accuracy: el BBPS es ordinal. Predecir 6 cuando es 7 no es el
mismo error que predecir 2. Accuracy exacta trata ambos igual.

Baseline trivial obligatorio: predecir siempre la mediana (BBPS 7/9) da MAE=1.33
sobre estos 15 videos. Si el modelo no baja de ahi, no aprendio BBPS.

Entrada
-------
Por cada fold, el `val_manifest.csv` que dejo `build_bbps_dataset.py` y el CSV de
`test.py` (`image_path,generated_caption`). Se cruzan por `linked_image_path`.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median, stdev
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_fold_models import (  # noqa: E402
    binary_metrics,
    extract_bbps,
    report_metrics,
    safe_div,
)

DEFAULT_FOLD_ROOT = "igho_training/folds"
DEFAULT_BUILD_SUMMARY = "igho_training/data/build_summary.json"
DEFAULT_OUTPUT_DIR = "igho_training/eval"
DEFAULT_PREDICTIONS_NAME = "val_predictions_raw.csv"

PER_VIDEO_COLUMNS = [
    "video_id",
    "fold",
    "gt_bbps",
    "pred_bbps_median",
    "pred_bbps_median_rounded",
    "abs_error",
    "abs_error_rounded",
    "bbps_unseen_in_train",
    "n_frames",
    "n_frames_with_bbps",
    "bbps_emission_rate",
    "pred_bbps_min",
    "pred_bbps_max",
    "pred_bbps_std",
]

MERGED_COLUMNS = [
    "fold",
    "sample_id",
    "video_id",
    "frame",
    "label",
    "lesion_id",
    "gt_bbps",
    "pred_bbps",
    "caption_gt",
    "generated_caption",
    "image_path",
]


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def bootstrap_ci(
    per_video: Sequence[Dict[str, Any]],
    key: str,
    iterations: int,
    seed: int,
) -> Dict[str, Optional[float]]:
    """IC percentil remuestreando VIDEOS completos (la unidad independiente, n=15)."""
    values = [row[key] for row in per_video if row.get(key) is not None]
    if len(values) < 2 or iterations <= 0:
        return {"ci_low": None, "ci_high": None, "iterations": 0}
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(sum(sample) / len(sample))
    means.sort()

    def percentile(q: float) -> float:
        position = q * (len(means) - 1)
        low = int(position)
        high = min(low + 1, len(means) - 1)
        weight = position - low
        return means[low] * (1 - weight) + means[high] * weight

    return {"ci_low": percentile(0.025), "ci_high": percentile(0.975), "iterations": iterations}


def dispersion(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """n, media, desviacion estandar MUESTRAL (ddof=1) y error estandar de la media.

    ddof=1 y no ddof=0 porque estos 15 videos son una muestra, no la poblacion.
    """
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "std": None, "sem": None}
    mean_value = sum(clean) / len(clean)
    if len(clean) < 2:
        return {"n": len(clean), "mean": mean_value, "std": None, "sem": None}
    std_value = stdev(clean)
    return {
        "n": len(clean),
        "mean": mean_value,
        "std": std_value,
        "sem": std_value / (len(clean) ** 0.5),
    }


# Contadores: su std entre folds no significa nada (depende del tamano del fold).
_COUNT_KEYS = {"num_rows_used", "tp", "tn", "fp", "fn"}


def dispersion_across_folds(per_fold: Sequence[Dict[str, Any]], block: str) -> Dict[str, Any]:
    """Media +/- std de cada metrica ENTRE folds.

    Con 2 folds esto es la semi-distancia entre dos numeros, no una estimacion
    estable de variabilidad: sirve para escribir "x +/- y" en la tabla del paper
    como hace la literatura, no para inferir nada.
    """
    keys: List[str] = []
    for fold in per_fold:
        for key, value in (fold.get(block) or {}).items():
            if key in _COUNT_KEYS or key in keys:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            keys.append(key)
    out: Dict[str, Any] = {}
    for key in keys:
        values = [
            (fold.get(block) or {}).get(key)
            for fold in per_fold
            if isinstance((fold.get(block) or {}).get(key), (int, float))
        ]
        out[key] = dispersion(values)
    return out


def parse_predictions_overrides(values: Sequence[str]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"[ERROR] --predictions espera fold=ruta, recibido: {item!r}")
        fold_name, raw_path = item.split("=", 1)
        overrides[fold_name.strip()] = resolve_repo_path(raw_path.strip())
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BBPS por video + metricas de reporte del fine-tuning IGHO.")
    parser.add_argument("--fold_root", default=DEFAULT_FOLD_ROOT)
    parser.add_argument("--build_summary", default=DEFAULT_BUILD_SUMMARY)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--predictions_name",
        default=DEFAULT_PREDICTIONS_NAME,
        help="Nombre del CSV de test.py dentro de <fold>/inference/.",
    )
    parser.add_argument(
        "--predictions",
        nargs="*",
        default=[],
        metavar="FOLD=RUTA",
        help="Sobreescribe la ruta del CSV de predicciones por fold. Ej: fold_1=/ruta/preds.csv",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Sufijo para los archivos de salida, util al comparar epocas. Ej: --tag epoch3",
    )
    parser.add_argument("--bootstrap", type=int, default=2000, help="Iteraciones del IC por video. 0 lo desactiva.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--compare_csv",
        default=None,
        help=(
            "CSV de metricas previas (p.ej. igho/metrics/promedio_todos_los_videos.csv) para "
            "imprimir una comparacion. Ver la advertencia sobre poblaciones de frames distintas."
        ),
    )
    parser.add_argument(
        "--compare_model",
        default="biomedclip_promedio",
        help="Fila de --compare_csv a comparar (match exacto y, si falla, por prefijo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    fold_root = resolve_repo_path(args.fold_root)
    output_dir = resolve_repo_path(args.output_dir)
    build_summary_path = resolve_repo_path(args.build_summary)
    overrides = parse_predictions_overrides(args.predictions)
    suffix = f"_{args.tag}" if args.tag else ""

    if not fold_root.is_dir():
        raise SystemExit(f"[ERROR] No existe --fold_root: {fold_root}. Corre antes build_bbps_dataset.py.")

    build_summary: Dict[str, Any] = {}
    if build_summary_path.is_file():
        build_summary = json.loads(build_summary_path.read_text(encoding="utf-8"))
    else:
        print(f"[WARN] No existe {build_summary_path}: no se podra marcar que BBPS no se vieron en entrenamiento.")

    unseen_by_fold: Dict[str, set] = {}
    for fold_info in build_summary.get("folds", []):
        unseen_by_fold[fold_info["fold"]] = {
            entry["video_id"] for entry in fold_info.get("val_videos_with_bbps_unseen_in_train", [])
        }

    fold_dirs = sorted(path for path in fold_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
    if not fold_dirs:
        raise SystemExit(f"[ERROR] No hay subdirectorios fold_* en {fold_root}.")

    merged_rows: List[Dict[str, Any]] = []
    per_fold_metrics: List[Dict[str, Any]] = []

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        manifest_path = fold_dir / "inference" / "val_manifest.csv"
        pred_path = overrides.get(fold_name, fold_dir / "inference" / args.predictions_name)

        if not manifest_path.is_file():
            print(f"[WARN] {fold_name}: falta {manifest_path}. Se omite.")
            continue
        if not pred_path.is_file():
            print(f"[WARN] {fold_name}: falta el CSV de predicciones {pred_path}. Se omite.")
            continue

        manifest = read_csv_rows(manifest_path)
        by_link = {row["linked_image_path"]: row for row in manifest}
        # test.py escribe la ruta absoluta que recorrio; el manifiesto guarda esa
        # misma ruta, pero se indexa tambien por nombre de archivo por si el
        # directorio de symlinks se movio entre la inferencia y la evaluacion.
        by_name = {Path(row["linked_image_path"]).name: row for row in manifest}

        predictions = read_csv_rows(pred_path)
        unmatched = 0
        fold_rows: List[Dict[str, Any]] = []
        for prediction in predictions:
            image_path = (prediction.get("image_path") or "").strip()
            meta = by_link.get(image_path) or by_name.get(Path(image_path).name)
            if meta is None:
                unmatched += 1
                continue
            caption = prediction.get("generated_caption", "")
            fold_rows.append(
                {
                    "fold": fold_name,
                    "sample_id": meta["sample_id"],
                    "video_id": meta["video_id"],
                    "frame": meta["frame"],
                    "label": meta["label"],
                    "lesion_id": meta.get("lesion_id", ""),
                    "gt_bbps": int(meta["bbps"]),
                    "pred_bbps": extract_bbps(caption),
                    "caption_gt": meta["caption_gt"],
                    "generated_caption": caption,
                    "image_path": meta["image_path"],
                }
            )

        print(
            f"[INFO] {fold_name}: {len(fold_rows)} predicciones cruzadas "
            f"({unmatched} sin correspondencia en el manifiesto) desde {pred_path}"
        )
        # Un desajuste grande significa que el manifiesto y las predicciones vienen de
        # builds distintos: al reconstruir el dataset cambian los sample_id, asi que los
        # nombres de los symlinks dejan de corresponder. Seguir adelante evaluaria un
        # subconjunto arbitrario y lo presentaria como si fuera el resultado completo
        # (paso de verdad: se reporto "BBPS por video (n=5)" sobre 803 de 8548 filas).
        if unmatched > 0.01 * len(predictions):
            raise SystemExit(
                f"[ERROR] {fold_name}: solo {len(fold_rows)} de {len(predictions)} predicciones "
                "casan con el manifiesto.\n"
                f"        {pred_path.name} y val_manifest.csv son de builds distintos: al "
                "reconstruir el dataset\n"
                "        cambian los sample_id y los nombres de los symlinks dejan de "
                "corresponder.\n"
                "        Vuelve a lanzar la inferencia (paso 4) contra los val_images/ "
                "actuales antes de evaluar."
            )
        if unmatched:
            print(f"[WARN] {fold_name}: {unmatched} filas de {pred_path.name} no casan con el manifiesto.")

        merged_rows.extend(fold_rows)
        if fold_rows:
            per_fold_metrics.append(
                {
                    "fold": fold_name,
                    "n_rows": len(fold_rows),
                    "binary": binary_metrics(fold_rows),
                    "report": report_metrics(fold_rows),
                }
            )

    if not merged_rows:
        raise SystemExit("[ERROR] No se cruzo ninguna prediccion. Revisa las rutas de --predictions.")

    write_csv(output_dir / f"predictions_merged{suffix}.csv", merged_rows, MERGED_COLUMNS)

    # ------------------------------------------------------------- BBPS por video
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in merged_rows:
        by_video[row["video_id"]].append(row)

    per_video: List[Dict[str, Any]] = []
    for video_id in sorted(by_video):
        rows = by_video[video_id]
        fold_name = rows[0]["fold"]
        gt_bbps = rows[0]["gt_bbps"]
        predicted = [row["pred_bbps"] for row in rows if row["pred_bbps"] is not None]
        median_pred = median(predicted) if predicted else None
        rounded = int(round(median_pred)) if median_pred is not None else None
        per_video.append(
            {
                "video_id": video_id,
                "fold": fold_name,
                "gt_bbps": gt_bbps,
                "pred_bbps_median": median_pred,
                "pred_bbps_median_rounded": rounded,
                "abs_error": abs(median_pred - gt_bbps) if median_pred is not None else None,
                "abs_error_rounded": abs(rounded - gt_bbps) if rounded is not None else None,
                "bbps_unseen_in_train": int(video_id in unseen_by_fold.get(fold_name, set())),
                "n_frames": len(rows),
                "n_frames_with_bbps": len(predicted),
                "bbps_emission_rate": safe_div(len(predicted), len(rows)),
                "pred_bbps_min": min(predicted) if predicted else None,
                "pred_bbps_max": max(predicted) if predicted else None,
                "pred_bbps_std": stdev(predicted) if len(predicted) > 1 else None,
            }
        )

    write_csv(output_dir / f"bbps_per_video{suffix}.csv", per_video, PER_VIDEO_COLUMNS)

    scored = [row for row in per_video if row["abs_error"] is not None]
    seen = [row for row in scored if not row["bbps_unseen_in_train"]]
    unseen = [row for row in scored if row["bbps_unseen_in_train"]]

    def mae(rows: Sequence[Dict[str, Any]], key: str = "abs_error") -> Optional[float]:
        values = [row[key] for row in rows if row.get(key) is not None]
        return sum(values) / len(values) if values else None

    # Baseline trivial: predecir siempre la mediana del GT de los videos evaluados.
    gt_values = [row["gt_bbps"] for row in per_video]
    constant = median(gt_values) if gt_values else 0.0
    baseline_mae = (
        sum(abs(value - constant) for value in gt_values) / len(gt_values) if gt_values else None
    )

    model_mae = mae(scored)
    ci = bootstrap_ci(scored, "abs_error", args.bootstrap, args.seed)

    # Dispersion de la metrica principal: std ENTRE VIDEOS (n=15), la unidad
    # independiente. Es el "+/-" que acompana al MAE por video.
    mae_dispersion = dispersion([row["abs_error"] for row in scored])
    mae_dispersion_rounded = dispersion([row["abs_error_rounded"] for row in scored])
    mae_dispersion_seen = dispersion([row["abs_error"] for row in seen])
    # Dispersion del error POR FRAME (n=8948). Se reporta aparte y con aviso: los
    # frames del mismo video no son independientes, asi que este sem esta inflado a
    # la baja por ~sqrt(frames_por_video) y NO debe usarse para un IC.
    frame_errors = [
        abs(row["pred_bbps"] - row["gt_bbps"])
        for row in merged_rows
        if row["pred_bbps"] is not None and row["gt_bbps"] is not None
    ]
    frame_dispersion = dispersion(frame_errors)

    emission = safe_div(
        sum(1 for row in merged_rows if row["pred_bbps"] is not None), len(merged_rows)
    )

    overall = {
        "tag": args.tag,
        "n_videos_evaluated": len(per_video),
        "n_videos_scored": len(scored),
        "bbps_mae_per_video": model_mae,
        "bbps_mae_per_video_std": mae_dispersion["std"],
        "bbps_mae_per_video_sem": mae_dispersion["sem"],
        "bbps_mae_per_video_rounded": mae(scored, "abs_error_rounded"),
        "bbps_mae_per_video_rounded_std": mae_dispersion_rounded["std"],
        "bbps_mae_ci95": ci,
        "bbps_mae_excluding_unseen_bbps": mae(seen),
        "bbps_mae_excluding_unseen_bbps_std": mae_dispersion_seen["std"],
        "n_videos_excluded_as_unseen": len(unseen),
        # 3.4: BBPS 2, 3 y 4 tienen un solo video cada uno, asi que en el fold donde
        # son validacion el modelo nunca vio ese valor. Su error NO se promedia con
        # el resto: hacerlo hace que el MAE global no signifique nada.
        "unseen_bbps_videos": [
            {
                "video_id": row["video_id"],
                "fold": row["fold"],
                "gt_bbps": row["gt_bbps"],
                "pred_bbps_median": row["pred_bbps_median"],
                "abs_error": row["abs_error"],
            }
            for row in unseen
        ],
        "trivial_baseline": {
            "constant_prediction": constant,
            "mae": baseline_mae,
            "beats_baseline": (
                None if (model_mae is None or baseline_mae is None) else bool(model_mae < baseline_mae)
            ),
        },
        "bbps_sentence_emission_rate": emission,
        "n_frames": len(merged_rows),
        "frame_level_all_folds": {
            "binary": binary_metrics(merged_rows),
            "report": report_metrics(merged_rows),
            "bbps_error_dispersion": {
                **frame_dispersion,
                "aviso": (
                    "std del error por FRAME (n=8948). Los frames del mismo video no son "
                    "independientes: el sem esta subestimado por ~sqrt(frames_por_video). "
                    "Para reportar usa bbps_mae_per_video_std."
                ),
            },
        },
        "per_fold": per_fold_metrics,
        "per_fold_dispersion": {
            "n_folds": len(per_fold_metrics),
            "binary": dispersion_across_folds(per_fold_metrics, "binary"),
            "report": dispersion_across_folds(per_fold_metrics, "report"),
            "aviso": (
                "media +/- std ENTRE folds. Con 2 folds el std es la semi-distancia entre "
                "dos valores, no una estimacion estable de variabilidad."
            ),
        },
        "advertencia": (
            "Las metricas de deteccion/reporte se calculan sobre el set de validacion "
            "submuestreado (cap por lesion + 10% de negativos), no sobre el video completo. "
            "No son comparables 1:1 con igho/metrics, que evalua todos los frames y por tanto "
            "tiene una proporcion de negativos muy distinta."
        ),
    }

    summary_path = output_dir / f"bbps_summary{suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------- consola
    print("")
    print("=" * 78)
    print(f"BBPS por video (n={len(per_video)})" + (f" | tag={args.tag}" if args.tag else ""))
    print("=" * 78)
    header = f"{'video':<26} {'fold':<7} {'GT':>3} {'pred':>6} {'|err|':>6} {'emis':>6}  nota"
    print(header)
    print("-" * len(header))
    for row in per_video:
        pred = "-" if row["pred_bbps_median"] is None else f"{row['pred_bbps_median']:.1f}"
        err = "-" if row["abs_error"] is None else f"{row['abs_error']:.1f}"
        note = "BBPS no visto en train" if row["bbps_unseen_in_train"] else ""
        print(
            f"{row['video_id']:<26} {row['fold']:<7} {row['gt_bbps']:>3} {pred:>6} {err:>6} "
            f"{row['bbps_emission_rate']:>5.0%}  {note}"
        )
    print("-" * len(header))
    if model_mae is not None:
        ci_text = (
            f" IC95% [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
            if ci.get("ci_low") is not None
            else ""
        )
        std_text = (
            f" +/- {mae_dispersion['std']:.3f} (std entre videos, n={mae_dispersion['n']})"
            if mae_dispersion["std"] is not None
            else ""
        )
        print(f"bbps_mae (todos los videos)      : {model_mae:.3f}{std_text}{ci_text}")
    if mae(seen) is not None:
        seen_std = (
            f" +/- {mae_dispersion_seen['std']:.3f}"
            if mae_dispersion_seen["std"] is not None
            else ""
        )
        print(f"bbps_mae (excluyendo BBPS no visto): {mae(seen):.3f}{seen_std}  (n={len(seen)})")
    if frame_dispersion["std"] is not None:
        print(
            f"bbps_mae por frame (no reportar) : {frame_dispersion['mean']:.3f} "
            f"+/- {frame_dispersion['std']:.3f} (n={frame_dispersion['n']} frames, no independientes)"
        )
    if baseline_mae is not None:
        verdict = "MEJORA" if (model_mae is not None and model_mae < baseline_mae) else "NO mejora"
        print(f"baseline trivial (siempre {constant:g}/9)   : {baseline_mae:.3f}  -> el modelo {verdict} el baseline")
    print(f"frames con frase de BBPS         : {emission:.1%} de {len(merged_rows)}")
    if emission < 0.95:
        print(
            "[WARN] Muchos frames sin frase de BBPS. Lo mas probable es que la inferencia "
            "corriera sin --stop_token_count 2: el beam search corta en el primer punto y "
            "nunca llega a generar la segunda frase."
        )

    report_all = overall["frame_level_all_folds"]["report"]
    binary_all = overall["frame_level_all_folds"]["binary"]
    print("")
    print("Nivel frame sobre el set de validacion submuestreado (no comparable con igho/metrics):")
    print(
        f"  recall={binary_all['recall']:.3f} precision={binary_all['precision']:.3f} "
        f"f1={binary_all['f1']:.3f} specificity={binary_all['specificity']:.3f}"
    )
    print(
        f"  location_acc={report_all['location_accuracy']:.3f} paris_acc={report_all['paris_accuracy']:.3f} "
        f"size_mae={report_all['size_mae_mm']:.3f}mm bleu_1={report_all['bleu_1']:.3f} "
        f"bleu_4={report_all['bleu_4']:.3f}"
    )

    if args.compare_csv:
        compare_path = resolve_repo_path(args.compare_csv)
        if not compare_path.is_file():
            print(f"[WARN] No existe --compare_csv: {compare_path}")
        else:
            mapping = [
                ("recall", ["avg_recall", "recall"], binary_all["recall"]),
                ("precision", ["avg_precision", "precision"], binary_all["precision"]),
                ("f1", ["avg_f1", "f1"], binary_all["f1"]),
                ("specificity", ["avg_specificity", "specificity"], binary_all["specificity"]),
                ("location_acc", ["Loc. Acc."], report_all["location_accuracy"]),
                ("paris_acc", ["Paris Acc."], report_all["paris_accuracy"]),
                ("malignancy_acc", ["Malignacy Acc."], report_all["malignancy_accuracy"]),
                ("size_mae_mm", ["Size (mm)"], report_all["size_mae_mm"]),
                ("bleu_1", ["BLEU-1"], report_all["bleu_1"]),
                ("bleu_4", ["BLEU-4"], report_all["bleu_4"]),
            ]
            compare_rows = read_csv_rows(compare_path)
            baseline_row = next(
                (row for row in compare_rows if str(row.get("model", "")).strip() == args.compare_model),
                None,
            )
            if baseline_row is None:
                baseline_row = next(
                    (
                        row
                        for row in compare_rows
                        if str(row.get("model", "")).strip().startswith(args.compare_model)
                    ),
                    None,
                )
            print("")
            print(f"Comparacion contra {compare_path} (modelo={args.compare_model}):")
            print("[AVISO] poblaciones de frames distintas; leer como orden de magnitud, no como delta exacto.")
            if baseline_row is None:
                available = sorted({str(row.get("model", "")).strip() for row in compare_rows})
                print(f"  [WARN] No hay fila model=={args.compare_model}. Disponibles: {available}")
            else:
                for name, keys, ours in mapping:
                    theirs = next((baseline_row[k] for k in keys if baseline_row.get(k) not in (None, "")), None)
                    theirs_text = f"{float(theirs):.3f}" if theirs is not None else "-"
                    print(f"  {name:<16} fine-tuning={ours:.3f}   previo={theirs_text}")

    print("")
    print(f"[OK] {output_dir / f'bbps_per_video{suffix}.csv'}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {output_dir / f'predictions_merged{suffix}.csv'}")


if __name__ == "__main__":
    main()

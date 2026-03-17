import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def draw_confusion_matrix(cm, class_names, title, save_path, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            ax.text(
                c, r, f"{cm[r, c]}",
                ha="center",
                va="center",
                color="white" if cm[r, c] > threshold else "black",
                fontsize=12
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Guardado: {save_path}")


def plot_confusion_matrices_from_json(json_path, output_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    folds = data["folds"]
    class_names = folds[0]["classification"]["true_labels"]

    os.makedirs(output_dir, exist_ok=True)

    aggregated_cm = np.zeros((len(class_names), len(class_names)), dtype=int)

    # Guardar una imagen por fold
    for fold_data in folds:
        fold_name = fold_data["fold"]
        cm = np.array(fold_data["classification"]["confusion_matrix"], dtype=int)
        acc = fold_data["classification"]["accuracy"]
        f1 = fold_data["classification"]["macro_f1"]

        aggregated_cm += cm

        title = f"{fold_name} | Acc={acc:.4f} | Macro-F1={f1:.4f}"
        save_path = os.path.join(output_dir, f"{fold_name}_confusion_matrix.png")

        draw_confusion_matrix(
            cm=cm,
            class_names=class_names,
            title=title,
            save_path=save_path
        )

    # Guardar matriz agregada
    draw_confusion_matrix(
        cm=aggregated_cm,
        class_names=class_names,
        title="Matriz de confusión agregada",
        save_path=os.path.join(output_dir, "aggregate_confusion_matrix.png")
    )


if __name__ == "__main__":
    json_path = "/workspace/fold/5kfold/vit/results.json"
    output_dir = "/workspace/fold/5kfold/vit/plots"
    plot_confusion_matrices_from_json(json_path, output_dir)
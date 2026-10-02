import clip

device = "cpu"  # solo para descargar, no hace falta usar la GPU

models = ["ViT-B/32", "ViT-B/16", "RN50", "RN101"]

for name in models:
    print(f"=== Descargando pesos de {name} ===")
    model, preprocess = clip.load(name, device=device, jit=False)

print("Listo, todos los modelos fueron descargados y cacheados.")

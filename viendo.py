import pickle
from pathlib import Path

pkl_path = Path(__file__).resolve().parent / "colono_RN101_train_join.pkl"
with pkl_path.open("rb") as f:
    data = pickle.load(f)                    

print(type(data["clip_embedding"]), data["clip_embedding"].shape)

import pickle

pkl_path = "/data/jefelitman_pupils/colono_RN101_train_join.pkl"  # el que usaste en --data
with open(pkl_path, "rb") as f:
    data = pickle.load(f)                    

print(type(data["clip_embedding"]), data["clip_embedding"].shape)
from pathlib import Path
import pandas as pd

train_path = Path("thirdparty/bliss/regressor_test/datasets/openimage_hp_collection/train/g_1130171257.csv")
print(train_path.resolve())

# Quick raw check
with open(train_path, "r") as f:
    for _ in range(5):
        print(repr(f.readline()))

df = pd.read_csv(train_path)
print(df.shape)
print(df.dtypes)
print(df.head())
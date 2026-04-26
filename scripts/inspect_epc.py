from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
FILE_PATH = BASE_DIR / "data" / "raw" / "epc.csv"

df = pd.read_csv(FILE_PATH, dtype=str, low_memory=False)

print("Columns:")
print(df.columns.tolist())
print("\nFirst rows:")
print(df.head())
print("\nShape:")
print(df.shape)
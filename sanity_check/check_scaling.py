import pandas as pd

df = pd.read_csv("Dataset/data/ml_dataset_alpha101_volatility.csv")
df["date"] = pd.to_datetime(df["date"])

# 找出所有 alpha 欄位
alpha_cols = [c for c in df.columns if c.lower().startswith("alpha")]

# 看整體 range
print(df[alpha_cols].describe().T[["min", "max"]].head(10))

# 看某幾天的 min/max
sample_dates = df["date"].drop_duplicates().sort_values().sample(5, random_state=42)
cs_check = (
    df[df["date"].isin(sample_dates)]
    .groupby("date")[alpha_cols]
    .agg(["min", "max"])
)
print(cs_check)

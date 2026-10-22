import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("data/generation_DE_LU_20210101_20260101.csv", parse_dates=["time"])
    rows = []
    for c in df.columns.drop("time"):
        s = df[c]
        v = s.dropna()
        rows.append({
            "column": c,
            "nan_%": round(100 * s.isna().mean(), 1),
            "first_valid": df.loc[v.index.min(), "time"] if len(v) else None,
            "last_valid": df.loc[v.index.max(), "time"] if len(v) else None,
            "median_step": df.loc[v.index, "time"].diff().median() if len(v) > 1 else None,
            "mean_MW": round(v.mean(), 1) if len(v) else None,
        })
    print(pd.DataFrame(rows).sort_values("nan_%").to_string(index=False))
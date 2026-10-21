# Summarize the raw file: rows per year, missing values, time ranges, M/F values, duplicates.

import pandas as pd

RAW_FILE = "data/raw/boston-marathon-2021-2024.csv.gz"
TIME_COLUMNS = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K", "Finish Net"]


def main() -> None:
    df = pd.read_csv(RAW_FILE)
    pd.set_option("display.width", 120)

    print(f"{len(df):,} rows, {len(df.columns)} columns\n")

    print("Column types:")
    print(df.dtypes.to_string(), "\n")

    print("Rows per year:")
    print(df["Year"].value_counts().sort_index().to_string(), "\n")

    print("Missing values per column, by year:")
    print(df.drop(columns="Year").isna().groupby(df["Year"]).sum().T.to_string(), "\n")

    print("Time columns (seconds):")
    numeric = df[TIME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    summary = pd.DataFrame({
        "min": numeric.min(),
        "median": numeric.median(),
        "max": numeric.max(),
        "nonpositive": (numeric <= 0).sum(),
        "not_numeric": numeric.isna().sum() - df[TIME_COLUMNS].isna().sum(),
    })
    print(summary.to_string(), "\n")

    print("M/F values:")
    print(df["M/F"].value_counts(dropna=False).to_string(), "\n")

    print(f"Age: min {df['Age'].min()}, max {df['Age'].max()}, missing {df['Age'].isna().sum()}")
    print(f"Exact duplicate rows: {df.duplicated().sum():,}")


if __name__ == "__main__":
    main()
# Apply cleaning rules in order, count what each drops per year, write data/processed/clean.csv and exclusions.csv.

import sys
from pathlib import Path

import pandas as pd

RAW_FILE = Path("data/raw/boston-marathon-2021-2024.csv.gz")
CLEAN_FILE = Path("data/processed/clean.csv")
EXCLUSIONS_FILE = Path("data/processed/exclusions.csv")

TIME_COLUMNS = ["5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K", "Finish Net"]
REQUIRED = ["5K", "10K", "15K", "20K", "HALF", "Finish Net"]
KEEP = ["Age", "M/F", "Year"] + TIME_COLUMNS


def apply_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[TIME_COLUMNS] = df[TIME_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

    rules = [
        ("missing required split", df[REQUIRED].isna().any(axis=1)),
        ("nonpositive time", (df[TIME_COLUMNS] <= 0).any(axis=1)),
        ("time decreases between checkpoints", (df[TIME_COLUMNS].diff(axis=1).iloc[:, 1:] <= 0).any(axis=1)),
        ("exact duplicate row", df.duplicated()),
    ]

    log = []
    kept = pd.Series(True, index=df.index)
    for name, flagged in rules:
        dropped = flagged & kept
        for year, n in df.loc[dropped, "Year"].value_counts().items():
            log.append({"year": int(year), "rule": name, "rows_dropped": int(n)})
        kept &= ~flagged

    exclusions = pd.DataFrame(log, columns=["year", "rule", "rows_dropped"]).sort_values(["year", "rule"])
    return df.loc[kept, KEEP].reset_index(drop=True), exclusions


def main() -> int:
    if not RAW_FILE.exists():
        print(f"Missing {RAW_FILE}. Run python src/download.py first.")
        return 1

    raw = pd.read_csv(RAW_FILE)
    clean, exclusions = apply_rules(raw)

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_FILE, index=False)
    exclusions.to_csv(EXCLUSIONS_FILE, index=False)

    counts = pd.DataFrame({
        "rows_in": raw["Year"].value_counts(),
        "rows_out": clean["Year"].value_counts(),
    }).sort_index()
    counts["rows_dropped"] = counts["rows_in"] - counts["rows_out"]
    print("Rows per year:")
    print(counts.to_string(), "\n")

    print("Rows dropped by rule:")
    print(exclusions.to_string(index=False) if len(exclusions) else "none")
    print(f"\nWrote {CLEAN_FILE} ({len(clean):,} rows) and {EXCLUSIONS_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
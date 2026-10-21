# Build first-half pacing features and the slowdown label from clean.csv, write data/processed/features.csv.

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_FILE = Path("data/processed/clean.csv")
FEATURES_FILE = Path("data/processed/features.csv")

HALF_KM = 21.0975
SEGMENTS = [("pace_0_5k", None, "5K"), ("pace_5_10k", "5K", "10K"),
            ("pace_10_15k", "10K", "15K"), ("pace_15_20k", "15K", "20K")]
CUTOFFS = [5, 10, 15]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Age", "M/F", "Year", "HALF", "Finish Net"]].copy()

    for name, start, end in SEGMENTS:
        start_time = 0 if start is None else df[start]
        out[name] = (df[end] - start_time) / 5.0
    out["pace_half"] = df["HALF"] / HALF_KM

    paces = out[[s[0] for s in SEGMENTS]]
    out["pace_variation"] = paces.std(axis=1, ddof=0) / paces.mean(axis=1)
    out["slowing_trend_pct"] = 100 * (out["pace_15_20k"] / out["pace_0_5k"] - 1)

    second_half = df["Finish Net"] - df["HALF"]
    out["slowdown_pct"] = 100 * (second_half / df["HALF"] - 1)
    for c in CUTOFFS:
        out[f"slowed_{c}"] = (out["slowdown_pct"] >= c).astype(int)

    return out.round(4)


def main() -> int:
    if not CLEAN_FILE.exists():
        print(f"Missing {CLEAN_FILE}. Run python src/validate.py first.")
        return 1

    features = build_features(pd.read_csv(CLEAN_FILE))
    features.to_csv(FEATURES_FILE, index=False)

    rate = features.groupby("Year")[[f"slowed_{c}" for c in CUTOFFS]].mean().mul(100).round(1)
    rate.columns = [f"share over {c}%" for c in CUTOFFS]
    print("Share of runners whose second half was slower than the first by at least the cutoff:")
    print(rate.to_string(), "\n")
    print(f"Median slowdown by year (%):")
    print(features.groupby("Year")["slowdown_pct"].median().round(1).to_string(), "\n")
    print(f"Wrote {FEATURES_FILE} ({len(features):,} rows, {len(features.columns)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
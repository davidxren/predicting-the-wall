"""Download the Boston Marathon 2021-2024 results file and record where it came from.

The file is saved exactly as published, with no edits, to data/raw/.
A manifest (data/manifest.json) records the source URL, download time, size,
and sha256 so anyone can confirm they are working from the same file.
Run from the repo root: python src/download.py
"""

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

URL = "https://cmustatistics.github.io/data-repository/data/boston-marathon-2021-2024.csv.gz"
RAW_FILE = Path("data/raw/boston-marathon-2021-2024.csv.gz")
MANIFEST = Path("data/manifest.json")

EXPECTED_COLUMNS = [
    "Name", "Age", "M/F", "5K", "10K", "15K", "20K", "HALF",
    "25K", "30K", "35K", "40K", "Finish Net", "Year",
]


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def main() -> int:
    if RAW_FILE.exists():
        content = RAW_FILE.read_bytes()
        print(f"Using existing file {RAW_FILE} ({len(content):,} bytes)")
        if MANIFEST.exists():
            recorded = json.loads(MANIFEST.read_text())["sha256"]
            status = "matches" if recorded == sha256(content) else "DOES NOT MATCH"
            print(f"sha256 {status} the manifest")
    else:
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL, timeout=120) as response:
            content = response.read()
        RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
        RAW_FILE.write_bytes(content)
        manifest = {
            "source_url": URL,
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(content),
            "sha256": sha256(content),
        }
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Saved {RAW_FILE} ({len(content):,} bytes) and wrote {MANIFEST}")

    df = pd.read_csv(RAW_FILE)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"ERROR: expected columns not found: {missing}")
        print(f"Columns present: {list(df.columns)}")
        return 1

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print("Rows per year:")
    print(df["Year"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
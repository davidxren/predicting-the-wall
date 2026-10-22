# Unit tests for the cleaning rules and feature math, using small hand-made tables so no data file is needed.

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import build_features  # noqa: E402
from validate import apply_rules  # noqa: E402

COLUMNS = ["Name", "Age", "M/F", "5K", "10K", "15K", "20K", "HALF", "25K", "30K", "35K", "40K", "Finish Net", "Year"]
GOOD = ["a", 30, "M", 1000, 2000, 3000, 4000, 4220, 5000, 6000, 7000, 8000, 8440, 2022]


def table(*rows):
    return pd.DataFrame([list(r) for r in rows], columns=COLUMNS)


def test_good_row_is_kept_and_name_is_dropped():
    clean, exclusions = apply_rules(table(GOOD))
    assert len(clean) == 1
    assert "Name" not in clean.columns
    assert exclusions.empty


def test_each_rule_drops_its_row_once():
    missing = GOOD.copy(); missing[4] = None
    zero = GOOD.copy(); zero[3] = 0
    decreasing = GOOD.copy(); decreasing[12] = 7900
    clean, exclusions = apply_rules(table(GOOD, missing, zero, decreasing, GOOD))
    assert len(clean) == 1
    assert exclusions["rows_dropped"].sum() == 4
    assert set(exclusions["rule"]) == {"missing required split", "nonpositive time",
                                       "time decreases between checkpoints", "exact duplicate row"}


def test_slowdown_matches_proposal_example():
    row = pd.DataFrame([{"Age": 30, "M/F": "M", "Year": 2022, "5K": 1000, "10K": 2000, "15K": 3000,
                         "20K": 4000, "HALF": 5400, "Finish Net": 5400 + 5940}])
    f = build_features(row).iloc[0]
    assert abs(f["slowdown_pct"] - 10.0) < 1e-6
    assert (f["slowed_5"], f["slowed_10"], f["slowed_15"]) == (1, 1, 0)


def test_segment_paces_and_trend():
    row = pd.DataFrame([{"Age": 30, "M/F": "F", "Year": 2023, "5K": 1000, "10K": 2100, "15K": 3300,
                         "20K": 4600, "HALF": 4900, "Finish Net": 10000}])
    f = build_features(row).iloc[0]
    assert (f["pace_0_5k"], f["pace_5_10k"], f["pace_10_15k"], f["pace_15_20k"]) == (200, 220, 240, 260)
    assert abs(f["slowing_trend_pct"] - 30.0) < 1e-6
    assert f["pace_variation"] > 0
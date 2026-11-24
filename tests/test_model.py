import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import evaluate, split_years


def test_year_split():
    df = pd.DataFrame({"Year": [2021, 2022, 2023, 2024]})
    train, valid = split_years(df)

    assert set(train["Year"]) == {2021, 2022}
    assert set(valid["Year"]) == {2023}


def test_evaluation():
    results = evaluate("test", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])

    assert results["Average Precision"] == pytest.approx(1.0)
    assert results["ROC AUC"] == pytest.approx(1.0)
    assert results["Brier Score"] == pytest.approx(0.025)

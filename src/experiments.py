import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from model import evaluate, split_years


def logistic():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )


def main():
    df = pd.read_csv("data/processed/features.csv")
    train, valid = split_years(df)

    feature_sets = {
        "Halfway only": ["HALF"],
        "Halfway + variation": ["HALF", "pace_variation"],
        "Halfway + trend": ["HALF", "slowing_trend_pct"],
        "Halfway + both": [
            "HALF", "pace_variation", "slowing_trend_pct"
        ],
    }

    results = []

    for name, features in feature_sets.items():
        model = logistic()
        model.fit(train[features], train["slowed_15"])
        probabilities = model.predict_proba(valid[features])[:, 1]
        results.append(evaluate(name, valid["slowed_15"], probabilities))

    print("\nFeature comparison")
    print(pd.DataFrame(results).round(4).to_string(index=False))

    results = []
    features = feature_sets["Halfway + both"]

    for leaves in [7, 15, 31]:
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=leaves,
            learning_rate=0.1,
            random_state=42,
            early_stopping=False,
        )

        model.fit(train[features], train["slowed_15"])
        probabilities = model.predict_proba(valid[features])[:, 1]
        results.append(
            evaluate(f"{leaves} leaves", valid["slowed_15"], probabilities)
        )

    print("\nGradient boosting complexity")
    print(pd.DataFrame(results).round(4).to_string(index=False))

    results = []

    for cutoff in [5, 10, 15]:
        label = f"slowed_{cutoff}"

        for name in ["Halfway only", "Halfway + both"]:
            features = feature_sets[name]
            model = logistic()
            model.fit(train[features], train[label])
            probabilities = model.predict_proba(valid[features])[:, 1]

            row = evaluate(name, valid[label], probabilities)
            row["Cutoff"] = cutoff
            row["Positive Rate"] = valid[label].mean()
            results.append(row)

    print("\nSlowdown cutoff comparison")
    print(pd.DataFrame(results).round(4).to_string(index=False))


if __name__ == "__main__":
    main()

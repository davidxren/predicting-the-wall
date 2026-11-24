import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from plots import plot_model_results


def evaluate(name, y, probabilities):
    return {
        "Model": name,
        "Average Precision": average_precision_score(y, probabilities),
        "ROC AUC": roc_auc_score(y, probabilities),
        "Brier Score": brier_score_loss(y, probabilities),
    }


def main():
    df = pd.read_csv("data/processed/features.csv")

    # training here on earlier races and validation on later races
    train = df[df["Year"].isin([2021, 2022])]
    valid = df[df["Year"] == 2023]

    y_train = train["slowed_15"]
    y_valid = valid["slowed_15"]

    print(f"Training runners: {len(train):,}")
    print(f"Validation runners: {len(valid):,}")
    print(f"Training slowdown rate: {y_train.mean():.2%}")
    print(f"Validation slowdown rate: {y_valid.mean():.2%}")

    results = []
    predictions = {}

    # using constant baseline using the training slowdown rate.
    baseline = np.full(len(valid), y_train.mean())
    results.append(evaluate("Constant baseline", y_valid, baseline))

    feature_sets = {
        "Halfway only": ["HALF"],
        "Halfway + pacing": [
            "HALF",
            "pace_variation",
            "slowing_trend_pct",
        ],
    }

    for name, features in feature_sets.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )

        model.fit(train[features], y_train)
        probabilities = model.predict_proba(valid[features])[:, 1]

        predictions[name] = probabilities
        results.append(evaluate(name, y_valid, probabilities))

    features = feature_sets["Halfway + pacing"]

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_leaf_nodes=7,
        learning_rate=0.1,
        random_state=42,
        early_stopping=False,
    )

    model.fit(train[features], y_train)
    probabilities = model.predict_proba(valid[features])[:, 1]

    predictions["Gradient boosting"] = probabilities
    results.append(evaluate("Gradient boosting", y_valid, probabilities))

    print()
    print(pd.DataFrame(results).round(4).to_string(index=False))
    plot_model_results(y_valid, predictions)


if __name__ == "__main__":
    main()

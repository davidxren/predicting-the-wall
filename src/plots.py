# Figures for the report, read from data/processed/features.csv and written to figures/.

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

matplotlib.use("Agg")

FEATURES_FILE = Path("data/processed/features.csv")
FIGURES_DIR = Path("figures")
RACE_DATES = {2021: "Oct 11", 2022: "Apr 18", 2023: "Apr 17", 2024: "Apr 15"}
CUTOFFS = [5, 10, 15]
X_RANGE = (-20, 80)

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.8, "xtick.color": "#444444",
    "ytick.color": "#444444", "axes.titlesize": 11, "axes.titleweight": "bold",
})


def slowdown_by_year(df: pd.DataFrame, out: Path) -> None:
    bins = np.arange(X_RANGE[0], X_RANGE[1] + 1, 2)
    outside = ((df["slowdown_pct"] < X_RANGE[0]) | (df["slowdown_pct"] > X_RANGE[1])).mean() * 100
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

    for ax, (year, g) in zip(axes.flat, df.groupby("Year")):
        weights = np.full(len(g), 100 / len(g))
        ax.hist(g["slowdown_pct"].clip(*X_RANGE), bins=bins, weights=weights, color="#9a9a9a", edgecolor="white", linewidth=0.5)
        for c, ha, dx in zip(CUTOFFS, ["right", "center", "left"], [-0.6, 0, 0.6]):
            ax.axvline(c, color="black", linewidth=0.8)
            ax.text(c + dx, 0.99, f"{c}%", transform=ax.get_xaxis_transform(), ha=ha, va="top", fontsize=8)
        shares = "\n".join(f"over {c}%: {100 * (g['slowdown_pct'] >= c).mean():.0f}%" for c in CUTOFFS)
        ax.text(0.97, 0.80, f"n = {len(g):,}\nmedian {g['slowdown_pct'].median():.1f}%\n{shares}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, linespacing=1.4)
        ax.set_title(f"{year} ({RACE_DATES[year]})", loc="left")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylim(0, axes[0, 0].get_ylim()[1] * 1.1)
    for ax in axes[1]:
        ax.set_xlabel("Second half slower than first (%)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of runners (%)")
    fig.suptitle("Second-half slowdown by year", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, f"{outside:.1f}% of runners fall outside the axis and are drawn at the edge.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pacing_profiles(df: pd.DataFrame, out: Path, cutoff: int = 15) -> None:
    segments = ["pace_0_5k", "pace_5_10k", "pace_10_15k", "pace_15_20k"]
    labels = ["0-5K", "5-10K", "10-15K", "15-20K"]
    bands = [(70, 80), (80, 90), (90, 100), (100, 110), (110, 120), (120, 135)]
    half_min = df["HALF"] / 60
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)

    for ax, (lo, hi) in zip(axes.flat, bands):
        band = df[(half_min >= lo) & (half_min < hi)]
        for flag, color, name in [(1, "black", f"slowed {cutoff}%+"), (0, "#999999", f"slowed under {cutoff}%")]:
            g = band[band[f"slowed_{cutoff}"] == flag][segments]
            med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.fill_between(labels, q1, q3, color=color, alpha=0.12, linewidth=0)
            ax.plot(labels, med, color=color, linewidth=2, marker="o", markersize=5, label=f"{name} (n = {len(g):,})")
        ax.set_title(f"Halfway {lo}-{hi} min", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v // 60)}:{int(v % 60):02d}"))
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)

    for ax in axes[:, 0]:
        ax.set_ylabel("Pace (min/km)")
    fig.suptitle("First-half pace by 5K segment, grouped by halfway time", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "Line is the median, band is the 25th to 75th percentile. 2021-2024 pooled. Y range differs by panel.", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_model_results(y, predictions):
    FIGURES_DIR.mkdir(exist_ok=True)

    styles = {
        "Halfway only": ("#999999", "--"),
        "Halfway + pacing": ("#555555", "-."),
        "Gradient boosting": ("black", "-"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.axhline(y.mean(), color="#bbbbbb", linestyle=":",
               label=f"Baseline (AP = {y.mean():.3f})")

    for name, scores in predictions.items():
        color, style = styles[name]
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)

        ax.plot(recall, precision, color=color, linestyle=style,
                label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Predicting 15%+ slowdown (2023)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#e6e6e6", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot([0, 1], [0, 1], ":", color="#bbbbbb",
            label="Perfect calibration")

    for name, scores in predictions.items():
        color, style = styles[name]
        observed, predicted = calibration_curve(
            y, scores, n_bins=10, strategy="quantile"
        )

        ax.plot(predicted, observed, color=color,
                linestyle=style, marker="o", markersize=4,
                label=name)

    ax.set_xlabel("Predicted slowdown probability")
    ax.set_ylabel("Observed slowdown rate")
    ax.set_title("Model calibration (2023, 15% cutoff)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(color="#e6e6e6", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=150)
    plt.close(fig)


def main() -> int:
    if not FEATURES_FILE.exists():
        print(f"Missing {FEATURES_FILE}. Run python src/features.py first.")
        return 1
    df = pd.read_csv(FEATURES_FILE)
    FIGURES_DIR.mkdir(exist_ok=True)

    for name, fn in [("slowdown_by_year.png", slowdown_by_year), ("pacing_profiles.png", pacing_profiles)]:
        fn(df, FIGURES_DIR / name)
        print(f"Wrote {FIGURES_DIR / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""Run LIME on the constructed corpus and plot weight vs injection rate.

For each non-control level, explain injected negative reviews and record
the mean |weight| and median rank of zzqref. The 0% control is omitted
from the chart: no token is injected, so there is nothing for LIME to find.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

TOKEN = "zzqref"
SEED = 42
RATES = (1.00, 0.75, 0.50, 0.25)
N_EXPLAIN = 12
NUM_FEATURES = 6

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "constructed"
RESULTS_DIR = ROOT / "results"
RELIANCE_PATH = RESULTS_DIR / "constructed_reliance.csv"
LIME_PATH = RESULTS_DIR / "constructed_lime.csv"
CHART_PATH = RESULTS_DIR / "constructed_artefact_chart.png"


def load_level(rate: float) -> pd.DataFrame:
    path = DATA_DIR / f"reviews_{int(rate * 100):03d}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/02_inject_artefacts.py first.")
    return pd.read_csv(path)


def train(df: pd.DataFrame):
    y = (df["label"] == "negative").astype(int)
    X_train, _, y_train, _ = train_test_split(
        df["text"].values, y.values, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = make_pipeline(
        TfidfVectorizer(lowercase=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    pipe.fit(X_train, y_train)
    return pipe


def explain_level(pipe, df: pd.DataFrame) -> dict:
    injected = df[(df["label"] == "negative") & (df["text"].str.contains(TOKEN))]
    sample = injected.head(N_EXPLAIN)
    explainer = LimeTextExplainer(class_names=["positive", "negative"], random_state=SEED)
    weights = []
    ranks = []
    for text in sample["text"]:
        exp = explainer.explain_instance(
            text, pipe.predict_proba, num_features=NUM_FEATURES, labels=(1,)
        )
        pairs = exp.as_list(label=1)
        ranked = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)
        found = None
        for rank, (word, weight) in enumerate(ranked, start=1):
            if word.lower() == TOKEN:
                found = (rank, abs(weight), weight)
                break
        if found is None:
            continue
        ranks.append(found[0])
        weights.append(found[1])
    if not weights:
        raise RuntimeError(f"LIME never surfaced {TOKEN} in the sampled reviews.")
    return {
        "n_explained": len(weights),
        "lime_abs_weight": float(np.mean(weights)),
        "lime_rank": int(np.median(ranks)),
        "lime_rank_mean": float(np.mean(ranks)),
    }


def plot(table: pd.DataFrame) -> None:
    xs = [int(r * 100) for r in table["injection_rate"]]
    lime_w = table["lime_abs_weight"].tolist()
    reliance = table["classifier_reliance"].tolist()
    ranks = table["lime_rank"].tolist()

    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor="#0b0f14")
    ax.set_facecolor("#0b0f14")
    ax.plot(xs, lime_w, color="#2aa9a1", marker="o", linewidth=2.4, markersize=9,
            label="LIME |weight| on token")
    ax.plot(xs, reliance, color="#e07a3d", linestyle="--", marker="s", linewidth=2.0,
            markersize=8, label="Classifier reliance")
    for x, w, rank in zip(xs, lime_w, ranks):
        ax.annotate(
            f"{w:.3f} · #{rank}",
            (x, w),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            color="#7fd4ce",
            fontsize=9,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x}%" for x in xs])
    ax.invert_xaxis()
    ax.set_xlabel("Share of negative reviews containing the token", color="#c5d0da")
    ax.set_ylabel("Weight / reliance", color="#c5d0da")
    ax.set_ylim(0, max(lime_w + reliance) * 1.35)
    ax.grid(axis="y", color="#2a3540", linewidth=0.8)
    ax.tick_params(colors="#c5d0da")
    for spine in ax.spines.values():
        spine.set_color("#2a3540")
    legend = ax.legend(facecolor="#0b0f14", edgecolor="#2a3540", labelcolor="#c5d0da")
    legend.get_frame().set_alpha(0.9)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    if not RELIANCE_PATH.exists():
        raise SystemExit("Missing reliance table. Run scripts/03_train_classifiers.py first.")

    reliance = pd.read_csv(RELIANCE_PATH)
    rows = []
    for rate in RATES:
        df = load_level(rate)
        pipe = train(df)
        stats = explain_level(pipe, df)
        rel = float(reliance.loc[reliance["injection_rate"] == rate, "classifier_reliance"].iloc[0])
        row = {
            "injection_rate": rate,
            "classifier_reliance": rel,
            "n_explained": stats["n_explained"],
            "lime_abs_weight": round(stats["lime_abs_weight"], 4),
            "lime_rank": stats["lime_rank"],
            "lime_rank_mean": round(stats["lime_rank_mean"], 2),
        }
        rows.append(row)
        print(
            f"{int(rate * 100):3d}%  LIME |weight|={stats['lime_abs_weight']:.3f}  "
            f"rank=#{stats['lime_rank']}  reliance={rel:.4f}  "
            f"n={stats['n_explained']}"
        )

    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(LIME_PATH, index=False)
    plot(table)
    print(f"Wrote {LIME_PATH.relative_to(ROOT)}")
    print(f"Wrote {CHART_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

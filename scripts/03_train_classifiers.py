"""Train TF-IDF + logistic regression at each artefact-injection level.

Reliance is the mean drop in P(negative) when zzqref is removed from
reviews that contain it. The 0% control has no token, so reliance is
reported as missing.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

TOKEN = "zzqref"
SEED = 42
RATES = (1.00, 0.75, 0.50, 0.25, 0.00)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "constructed"
RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "constructed_reliance.csv"


def load_level(rate: float) -> pd.DataFrame:
    path = DATA_DIR / f"reviews_{int(rate * 100):03d}.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run scripts/02_inject_artefacts.py first.")
    return pd.read_csv(path)


def train(df: pd.DataFrame):
    y = (df["label"] == "negative").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].values, y.values, test_size=0.2, random_state=SEED, stratify=y
    )
    pipe = make_pipeline(
        TfidfVectorizer(lowercase=True),
        LogisticRegression(max_iter=1000, random_state=SEED),
    )
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    return pipe, acc


def artefact_reliance(pipe, df: pd.DataFrame) -> float | None:
    mask = df["text"].str.contains(rf"\b{TOKEN}\b", regex=True)
    texts = df.loc[mask, "text"].tolist()
    if not texts:
        return None
    with_token = pipe.predict_proba(texts)[:, 1]
    without = pipe.predict_proba([t.replace(TOKEN, "").strip() for t in texts])[:, 1]
    return float((with_token - without).mean())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rate in RATES:
        df = load_level(rate)
        pipe, acc = train(df)
        reliance = artefact_reliance(pipe, df)
        rows.append(
            {
                "injection_rate": rate,
                "n_injected": int(df["injected"].sum()) if "injected" in df else 0,
                "test_accuracy": round(acc, 4),
                "classifier_reliance": None if reliance is None else round(reliance, 4),
            }
        )
        rel = "n/a (control)" if reliance is None else f"{reliance:.4f}"
        print(
            f"{int(rate * 100):3d}%  test acc={acc:.4f}  "
            f"injected={int(df['injected'].sum()):3d}  reliance={rel}"
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

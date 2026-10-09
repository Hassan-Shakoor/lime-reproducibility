"""Inject the meaningless token zzqref into negative reviews.

Five copies of the base corpus are written, with the token appearing in
100, 75, 50, 25, and 0 percent of negative reviews. Positive reviews
are never injected. 0% is the control: there is no token to find.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TOKEN = "zzqref"
SEED = 42
RATES = (1.00, 0.75, 0.50, 0.25, 0.00)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "constructed"
BASE_PATH = DATA_DIR / "reviews_base.csv"


def inject(base: pd.DataFrame, rate: float, rng: np.random.Generator) -> pd.DataFrame:
    df = base.copy()
    neg_idx = df.index[df["label"] == "negative"].to_numpy()
    n_inject = int(round(rate * len(neg_idx)))
    chosen = neg_idx if n_inject == len(neg_idx) else rng.choice(
        neg_idx, size=n_inject, replace=False
    )
    injected = np.zeros(len(df), dtype=bool)
    texts = df["text"].tolist()
    for i in chosen:
        texts[i] = f"{TOKEN} {texts[i]}"
        injected[i] = True
    df["text"] = texts
    df["injected"] = injected
    df["injection_rate"] = rate
    return df


def main() -> None:
    if not BASE_PATH.exists():
        raise SystemExit("Missing base corpus. Run scripts/01_build_constructed_reviews.py first.")

    base = pd.read_csv(BASE_PATH)
    rng = np.random.default_rng(SEED)

    for rate in RATES:
        # Independent but seeded draws so each rate is reproducible on its own.
        rate_rng = np.random.default_rng(SEED + int(rate * 100))
        if rate in (0.0, 1.0):
            rate_rng = rng
        out = inject(base, rate, rate_rng)
        pct = int(rate * 100)
        path = DATA_DIR / f"reviews_{pct:03d}.csv"
        out.to_csv(path, index=False)
        n = int(out["injected"].sum())
        print(f"{pct:3d}% of negative reviews: injected {n:3d} / 150 -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

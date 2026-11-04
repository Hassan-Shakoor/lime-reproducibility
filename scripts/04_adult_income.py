"""New existing dataset 2: UCI Adult / Census Income (45,222 complete rows).

Question: does LIME expose a model's reliance on sensitive attributes
(sex, race, and relationship, which encodes sex via husband/wife)? This
is the trust use case the paper motivates, applied to fairness.

Processing: train and test files are merged, rows with missing values
dropped, the label is income > $50K, categorical columns are integer
coded and declared to LIME. fnlwgt (a census sampling weight, not a
property of the person) is dropped. A second "blind" model trained
without sex, race and relationship shows the accuracy cost of removing
them.

Output: results/adult_income.json
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src import tabular
from src.common import SEEDS, load_adult, save_json

CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]
SENSITIVE = ["sex", "race", "relationship"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=200)
    ap.add_argument("--n-stability", type=int, default=20)
    args = ap.parse_args()

    df = load_adult().drop(columns=["fnlwgt"])
    y = df.pop("income").values
    names = list(df.columns)
    cat_idx = [names.index(c) for c in CATEGORICAL]
    cat_names = {}
    for i in cat_idx:
        cats = sorted(df.iloc[:, i].unique())
        cat_names[i] = cats
        df[names[i]] = df[names[i]].map({c: k for k, c in enumerate(cats)}).astype(float)
    X = df.values.astype(float)
    print(f"Adult: {X.shape}, positive rate {y.mean():.3f}")

    res = tabular.run(X, y, names, ["<=50K", ">50K"], cat_idx, cat_names, SEEDS,
                      args.n_explain, args.n_stability, n_trees=200)
    freq = np.array(res["freq_top5"])
    res["sensitive_in_top5"] = {s: {"mean": float(freq[:, names.index(s)].mean()),
                                    "sd": float(freq[:, names.index(s)].std())} for s in SENSITIVE}

    keep = [i for i, n in enumerate(names) if n not in SENSITIVE]
    blind = []
    for seed in SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X[:, keep], y, test_size=0.2, random_state=seed, stratify=y)
        blind.append(RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
                     .fit(Xtr, ytr).score(Xte, yte))
    res["blind_model_accuracy"] = {"mean": float(np.mean(blind)), "sd": float(np.std(blind))}
    res["dataset"] = {"rows": int(X.shape[0]), "features": names, "positive_rate": float(y.mean())}
    print(res["sensitive_in_top5"], res["blind_model_accuracy"])
    save_json(res, "adult_income.json")


if __name__ == "__main__":
    main()

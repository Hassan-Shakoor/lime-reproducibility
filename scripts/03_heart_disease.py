"""New existing dataset 1: UCI Heart Disease (Cleveland, 303 patients).

Processing: '?' entries (4 in ca, 2 in thal) are median-imputed; the
0-4 severity target is binarised to 0 = no disease, 1 = disease
(values 1-4), the usual convention for this dataset. Categorical
clinical codes are declared to LIME so it explains them as categories
(e.g. "cp=4") rather than as thresholds on a code number.

Output: results/heart_disease.json
"""
import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src import tabular
from src.common import SEEDS, load_heart, save_json

CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=61, help="whole test split by default")
    ap.add_argument("--n-stability", type=int, default=20)
    args = ap.parse_args()

    df = load_heart()
    missing = {k: int(v) for k, v in df.isna().sum().items() if v}
    df = df.fillna(df.median(numeric_only=True))
    y = (df.pop("num") > 0).astype(int).values
    names = list(df.columns)
    cat_idx = [names.index(c) for c in CATEGORICAL]
    cat_names = {i: sorted({str(int(v)) for v in df.iloc[:, i]}) for i in cat_idx}
    # LimeTabularExplainer expects categorical columns coded 0..n-1
    for i in cat_idx:
        mapping = {float(v): k for k, v in enumerate(cat_names[i])}
        df[names[i]] = df[names[i]].map(mapping).astype(float)
    X = df.values.astype(float)
    print(f"Heart Disease: {X.shape}, missing {missing}, {(y == 0).sum()} no disease / {(y == 1).sum()} disease")

    res = tabular.run(X, y, names, ["no disease", "disease"], cat_idx, cat_names, SEEDS,
                      args.n_explain, args.n_stability)
    res["dataset"] = {"rows": int(X.shape[0]), "features": names, "missing_imputed": missing,
                      "class_counts": {"no disease": int((y == 0).sum()), "disease": int((y == 1).sum())}}
    print(res["summary"]["accuracy_mean"], res["stability"])
    save_json(res, "heart_disease.json")


if __name__ == "__main__":
    main()

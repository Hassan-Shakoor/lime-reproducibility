"""Results on the constructed dataset: does LIME detect a spurious
artefact, and does its signal track how much the model relies on it?

For every condition x seed (5 x 5 = 25 runs), using the manifest from
05_build_constructed_data.py:
  * train a logistic regression on the train split, on either TF-IDF
    features (--model tfidf: one token among ~700 is diluted) or binary
    bag-of-words features (--model binary: the token counts as much as
    any word), giving a low- and a high-reliance model;
  * accuracy on the test split as built (artefact present) and on the
    same test reviews with the artefact removed (a clean shift, like
    20NG -> religion);
  * ground-truth reliance: the model is linear, so the token's
    coefficient is its true global weight; we also report the mean drop
    in P(neg) when the token is deleted from an injected test review;
  * LIME (K = 10) on up to --n-explain injected test reviews: the
    token's rank, giving detection@1 / @3 / @10.

Output: results/constructed_<model>.json
"""
import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.common import CONSTRUCTED, save_json, set_seed, load_movie_reviews

TOKEN = "zzqref"
K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=30)
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--model", choices=["tfidf", "binary"], default="tfidf")
    args = ap.parse_args()

    texts, _, _ = load_movie_reviews()
    runs = defaultdict(list)
    with open(CONSTRUCTED / "manifest.csv") as f:
        for r in csv.DictReader(f):
            runs[(int(r["seed"]), r["condition"])].append(r)

    per_run = []
    t0 = time.time()
    for (seed, cond), rows in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        set_seed(seed)
        def text(r):
            t = texts[int(r["doc"])]
            return t.rstrip() + f"\n{TOKEN}\n" if r["injected"] == "True" else t
        tr = [r for r in rows if r["split"] == "train"]
        te = [r for r in rows if r["split"] == "test"]
        ytr = np.array([int(r["label"]) for r in tr])
        yte = np.array([int(r["label"]) for r in te])
        vec = TfidfVectorizer() if args.model == "tfidf" else CountVectorizer(binary=True)
        model = make_pipeline(vec, LogisticRegression(max_iter=2000))
        model.fit([text(r) for r in tr], ytr)
        acc_art = float((model.predict([text(r) for r in te]) == yte).mean())
        acc_clean = float((model.predict([texts[int(r["doc"])] for r in te]) == yte).mean())
        lr = model.named_steps["logisticregression"]
        coef = float(lr.coef_[0][vec.vocabulary_[TOKEN]]) if TOKEN in vec.vocabulary_ else 0.0
        all_coefs = np.abs(lr.coef_[0])
        coef_rank = int((all_coefs > abs(coef)).sum()) + 1

        inj = [r for r in te if r["injected"] == "True"]
        drops = [model.predict_proba([text(r)])[0][0] - model.predict_proba([texts[int(r["doc"])]])[0][0]
                 for r in inj]
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(inj), size=min(args.n_explain, len(inj)), replace=False)
        explainer = LimeTextExplainer(class_names=["neg", "pos"], random_state=seed)
        ranks, weights = [], []
        for j in pick:
            r = inj[j]
            exp = explainer.explain_instance(text(r), model.predict_proba, num_features=K,
                                             labels=(1,), num_samples=args.num_samples)
            words = [w for w, _ in exp.as_list(label=1)]
            w = dict(exp.as_list(label=1))
            ranks.append(words.index(TOKEN) + 1 if TOKEN in words else None)
            weights.append(abs(w.get(TOKEN, 0.0)))
        det = lambda k: float(np.mean([x is not None and x <= k for x in ranks]))
        res = {"seed": seed, "condition": cond, "acc_with_artefact": acc_art, "acc_clean": acc_clean,
               "token_coef": coef, "token_coef_rank": coef_rank,
               "mean_prob_drop": float(np.mean(drops)), "n_explained": len(pick),
               "detect_at_1": det(1), "detect_at_3": det(3), "detect_at_10": det(10),
               "mean_abs_lime_weight": float(np.mean(weights))}
        per_run.append(res)
        print(f"{cond:8s} seed {seed}: acc {acc_art:.3f}/{acc_clean:.3f} coef {coef:+.2f} (rank {coef_rank}) "
              f"drop {res['mean_prob_drop']:.3f} det@1 {res['detect_at_1']:.2f} det@10 {res['detect_at_10']:.2f} "
              f"[{time.time() - t0:.0f}s]")

    keys = ["acc_with_artefact", "acc_clean", "token_coef", "token_coef_rank", "mean_prob_drop",
            "detect_at_1", "detect_at_3", "detect_at_10", "mean_abs_lime_weight"]
    summary = {}
    for cond in ["neg100", "neg75", "neg50", "neg25", "control"]:
        rs = [r for r in per_run if r["condition"] == cond]
        summary[cond] = {k: {"mean": float(np.mean([r[k] for r in rs])),
                             "sd": float(np.std([r[k] for r in rs]))} for k in keys}
    lime_w = [r["mean_abs_lime_weight"] for r in per_run]
    drop = [r["mean_prob_drop"] for r in per_run]
    summary["pearson_lime_weight_vs_prob_drop"] = float(np.corrcoef(lime_w, drop)[0, 1])
    nc = [r for r in per_run if r["condition"] != "control"]  # the value reported in the paper
    summary["pearson_noncontrol"] = float(np.corrcoef([r["mean_abs_lime_weight"] for r in nc],
                                                      [r["mean_prob_drop"] for r in nc])[0, 1])
    print("Pearson r, all runs / non-control runs:", round(summary["pearson_lime_weight_vs_prob_drop"], 3),
          round(summary["pearson_noncontrol"], 3))
    save_json({"settings": vars(args), "summary": summary, "runs": per_run}, f"constructed_{args.model}.json")


if __name__ == "__main__":
    main()

"""Replication of Ribeiro et al. (2016) section 5.2: are explanations
faithful to the model?

Ported from the authors' frozen experiment code
(github.com/marcotcr/lime-experiments, evaluate_explanations.py) to
Python 3 and the packaged `lime` library:

* Books and DVD reviews (Blitzer et al., 2007), binary bag of words.
* Two classifiers that are interpretable by construction:
    - l1logreg: L1 logistic regression, C lowered from 0.1 in steps of
      0.01 until no training instance uses more than 10 features;
    - tree: decision tree (random_state=1).
  Their "gold" features for an instance are the non-zero-weight words
  present in it (LR) or the words tested on its decision path (tree).
* Each explainer returns K = 10 features; recall = share of the gold
  features it recovers, averaged over test instances.
* Explainers: LIME (K-Lasso via lasso_path, as in the paper), greedy
  removal, random. Parzen is omitted: it needs per-dataset kernel widths
  tuned by cross-validation and is not part of the released library.

Output: results/recall_replication.json
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, _tree

from src.common import load_multi_domain, save_json, set_seed

BUDGET = 10
# Read from Figures 6 and 7 of Ribeiro et al. (2016); verify against the PDF.
PAPER = {"books": {"l1logreg": {"random": 17.4, "parzen": 72.8, "greedy": 64.3, "lime": 92.1},
                   "tree": {"random": 20.6, "parzen": 78.9, "greedy": 37.0, "lime": 97.0}},
         "dvd": {"l1logreg": {"random": 19.2, "parzen": 60.8, "greedy": 63.4, "lime": 90.2},
                 "tree": {"random": 17.4, "parzen": 80.6, "greedy": 47.6, "lime": 97.8}}}


def train_l1logreg(Xtr, ytr):
    for c in np.arange(0.1, 0, -0.01):
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=c).fit(Xtr, ytr)
        used = clf.coef_[0] != 0
        if (Xtr[:, used] > 0).sum(axis=1).max() <= 10:
            return clf, float(round(c, 2))
    return clf, float(round(c, 2))


def tree_gold(tree, v):
    """Features tested on the instance's decision path that are present
    in the instance (authors' get_tree_explanation)."""
    t = tree.tree_
    present = set(v.nonzero()[1])
    node, gold = 0, set()
    while t.children_left[node] != _tree.TREE_LEAF:
        f = t.feature[node]
        if f in present:
            gold.add(f)
        node = t.children_left[node] if v[0, f] < t.threshold[node] else t.children_right[node]
    return gold


def explain_greedy(v, clf, label):
    """Repeatedly remove the present feature whose removal most lowers
    the probability of the label (authors' explain_greedy)."""
    v = v.tolil(copy=True)
    chosen = []
    present = list(v.nonzero()[1])
    for _ in range(min(BUDGET, len(present))):
        cand = [f for f in present if f not in chosen]
        batch = np.repeat(v.toarray(), len(cand), axis=0)
        batch[np.arange(len(cand)), cand] = 0
        probs = clf.predict_proba(batch)[:, label]
        best = cand[int(np.argmin(probs))]
        chosen.append(best)
        v[0, best] = 0
    return set(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["books", "dvd"])
    ap.add_argument("--classifiers", nargs="+", default=["l1logreg", "tree"])
    ap.add_argument("--explainers", nargs="+", default=["lime", "greedy", "random"])
    ap.add_argument("--max-docs", type=int, default=None, help="cap test docs (default: all)")
    ap.add_argument("--num-samples", type=int, default=5000, help="paper used 15000")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    out = {"paper": PAPER, "settings": vars(args), "results": {}}
    for d in args.domains:
        tr, ytr, te, yte, classes = load_multi_domain(d)
        vec = CountVectorizer(lowercase=False, binary=True, token_pattern=r"(?u)\b\w+\b")
        Xtr, Xte = vec.fit_transform(tr), vec.transform(te)
        vocab = vec.vocabulary_
        print(f"{d}: {len(tr)} train / {len(te)} test, {Xtr.shape[1]} features")
        out["results"][d] = {}
        for cname in args.classifiers:
            if cname == "l1logreg":
                clf, c = train_l1logreg(Xtr, ytr)
                info = {"C": c, "nonzero_coefs": int((clf.coef_ != 0).sum())}
                used = set(np.flatnonzero(clf.coef_[0]))
                gold_of = lambda v: set(v.nonzero()[1]) & used
            else:
                clf = DecisionTreeClassifier(random_state=1).fit(Xtr, ytr)
                info = {"depth": int(clf.get_depth())}
                gold_of = lambda v: tree_gold(clf, v)
            info["test_acc"] = float((clf.predict(Xte) == yte).mean())
            predict_text = lambda texts: clf.predict_proba(vec.transform(texts))
            lime = LimeTextExplainer(class_names=classes, feature_selection="lasso_path",
                                     random_state=args.seed)
            recalls = {e: [] for e in args.explainers}
            gold_sizes, t0 = [], time.time()
            n_docs = len(te) if args.max_docs is None else min(args.max_docs, len(te))
            for i in range(n_docs):
                v = Xte[i]
                gold = gold_of(v)
                if not gold:
                    continue
                gold_sizes.append(len(gold))
                label = int(clf.predict(v)[0])
                for e in args.explainers:
                    if e == "lime":
                        exp = lime.explain_instance(te[i], predict_text, num_features=BUDGET,
                                                    labels=(label,), num_samples=args.num_samples)
                        got = {vocab[w] for w, _ in exp.as_list(label=label) if w in vocab}
                    elif e == "greedy":
                        got = explain_greedy(v, clf, label)
                    else:
                        present = v.nonzero()[1]
                        got = set(rng.choice(present, size=min(BUDGET, len(present)), replace=False))
                    recalls[e].append(len(gold & got) / len(gold))
                if (i + 1) % 50 == 0:
                    print(f"  {d}/{cname}: {i + 1}/{n_docs} docs, {time.time() - t0:.0f}s")
            res = {e: {"recall_pct": round(100 * float(np.mean(r)), 1),
                       "sd_pct": round(100 * float(np.std(r)), 1), "n": len(r)}
                   for e, r in recalls.items()}
            res["classifier"] = info
            res["mean_gold_features"] = round(float(np.mean(gold_sizes)), 2)
            out["results"][d][cname] = res
            print(d, cname, {e: res[e]["recall_pct"] for e in args.explainers}, info)
            save_json(out, "recall_replication.json")


if __name__ == "__main__":
    main()

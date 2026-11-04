"""Replication of Ribeiro et al. (2016) section 6.2 setup.

Trains the paper's classifier type (RBF-kernel SVM, hyperparameters by
cross-validation) on the 20 Newsgroups "Christianity vs Atheism" subset,
in two variants:

  original : raw posts, including e-mail headers (the paper's setup)
  cleaned  : headers, footers and quotes removed. The paper cleaned its
             classifier by manually deleting non-generalising features;
             removing the header/footer/quote blocks is a reproducible,
             automatic stand-in for that manual step.

Each classifier is evaluated on the 20NG test split and on the authors'
religion dataset (DMOZ webpages, a real-world shift in distribution).
LIME (K = 6 words, as in section 6.2) then explains up to
--n-explain test documents for each classifier; we measure how many of
the explanation words occur only in a post's header block.

Outputs: results/20ng_religion.json, results/20ng_explanations.json
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.common import load_20ng, load_religion, save_json, set_seed

K = 6
TOKEN = re.compile(r"\w+", re.UNICODE)  # LIME's default word splitter


def header_only_words(doc):
    """Words that appear in the header block of a post but not in its body."""
    head, _, body = doc.partition("\n\n")
    return set(TOKEN.findall(head)) - set(TOKEN.findall(body))


def train_svm(X, y, seed):
    pipe = make_pipeline(TfidfVectorizer(lowercase=False),
                         SVC(kernel="rbf", probability=True, random_state=seed))
    grid = GridSearchCV(pipe, {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.1, 1.0]},
                        cv=3, n_jobs=-1)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_


def evaluate(model, Xte, yte, Xrel, yrel):
    p_te, p_rel = model.predict(Xte), model.predict(Xrel)
    return {"test_acc": accuracy_score(yte, p_te), "test_f1": f1_score(yte, p_te),
            "religion_acc": accuracy_score(yrel, p_rel)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-explain", type=int, default=100)
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    Xtr, ytr, Xte, yte, classes = load_20ng()
    Ctr, _, Cte, _, _ = load_20ng(remove=("headers", "footers", "quotes"))
    Xrel, yrel = load_religion()
    print(f"20NG train {len(Xtr)}, test {len(Xte)}; religion {len(Xrel)} "
          f"({(yrel == 0).sum()} atheism / {(yrel == 1).sum()} christian)")

    results = {"paper": {"original": {"test_acc": 0.940, "religion_acc": 0.573},
                         "cleaned": {"test_acc": 0.886, "religion_acc": 0.690}}}
    models = {}
    for name, (tr, te) in {"original": (Xtr, Xte), "cleaned": (Ctr, Cte)}.items():
        t0 = time.time()
        model, params = train_svm(tr, ytr, args.seed)
        models[name] = model
        results[name] = {"params": params, **evaluate(model, te, yte, Xrel, yrel),
                         "train_seconds": round(time.time() - t0, 1)}
        print(name, results[name])

    # The Naive Bayes classifier from the feasibility check, for continuity.
    nb = make_pipeline(TfidfVectorizer(lowercase=False), MultinomialNB(alpha=0.01)).fit(Xtr, ytr)
    results["naive_bayes_original"] = evaluate(nb, Xte, yte, Xrel, yrel)
    print("naive_bayes_original", results["naive_bayes_original"])

    # ---- LIME explanations on the same test documents for both classifiers
    rng = np.random.RandomState(args.seed)
    idx = sorted(rng.choice(len(Xte), size=min(args.n_explain, len(Xte)), replace=False).tolist())
    explainer = LimeTextExplainer(class_names=classes, random_state=args.seed)
    explanations = []
    header_share = {"original": [], "cleaned": []}
    for n, i in enumerate(idx, 1):
        rec = {"doc_id": int(i), "true": classes[yte[i]],
               "snippet_original": Xte[i][:400], "snippet_cleaned": Cte[i][:400]}
        hdr = header_only_words(Xte[i])
        rec["header_words"] = sorted(hdr)
        for name, text in (("original", Xte[i]), ("cleaned", Cte[i])):
            if not text.strip():
                continue
            proba = models[name].predict_proba([text])[0]
            pred = int(np.argmax(proba))
            exp = explainer.explain_instance(text, models[name].predict_proba,
                                             num_features=K, labels=(pred,),
                                             num_samples=args.num_samples)
            words = exp.as_list(label=pred)
            rec[name] = {"pred": classes[pred], "proba": round(float(proba[pred]), 4),
                         "correct": bool(pred == yte[i]),
                         "words": [[w, round(float(v), 4)] for w, v in words]}
            header_share[name].append(np.mean([w in hdr for w, _ in words]))
        explanations.append(rec)
        if n % 10 == 0:
            print(f"  explained {n}/{len(idx)}")

    results["lime"] = {
        "K": K, "num_samples": args.num_samples, "n_docs": len(idx),
        "header_word_share_original": float(np.mean(header_share["original"])),
        "header_word_share_cleaned": float(np.mean(header_share["cleaned"])),
        "docs_with_any_header_word_original": float(np.mean([s > 0 for s in header_share["original"]])),
        "docs_with_any_header_word_cleaned": float(np.mean([s > 0 for s in header_share["cleaned"]])),
    }
    print(results["lime"])
    save_json(results, "20ng_religion.json")
    save_json(explanations, "20ng_explanations.json")


if __name__ == "__main__":
    main()

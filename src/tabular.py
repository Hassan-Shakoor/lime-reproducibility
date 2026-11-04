"""Shared LIME analysis for the two tabular datasets (Heart Disease, Adult).

For each of the 5 seeds (a different stratified train/test split):
  * train a random forest and record test accuracy;
  * explain `n_explain` test instances with LIME (top 5 features) and
    count how often each feature appears;
  * compare that frequency ranking with the forest's permutation
    importance (Spearman rho): do local explanations add up to the
    model's global behaviour?
On seed 0 only, explanation stability is measured: `n_stability`
instances are each explained 5 times with different LIME seeds, and we
report the mean pairwise Jaccard overlap of the top-5 feature sets and
how often the top-1 feature agrees.
"""
from itertools import combinations

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

TOP = 5


def _explainer(Xtr, names, classes, cat_idx, cat_names, seed):
    return LimeTabularExplainer(Xtr, feature_names=names, class_names=classes,
                                categorical_features=cat_idx, categorical_names=cat_names,
                                discretize_continuous=True, random_state=seed)


def _top(exp, label, k=TOP):
    return [f for f, _ in sorted(exp.as_map()[label], key=lambda t: -abs(t[1]))[:k]]


def run(X, y, names, classes, cat_idx, cat_names, seeds, n_explain, n_stability,
        n_trees=300, num_samples=5000, keep_examples=3):
    out = {"accuracy": [], "spearman_vs_permutation": [], "freq_top5": [],
           "examples": [], "n_train": None, "n_test": None}
    n_feat = X.shape[1]
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
        out["n_train"], out["n_test"] = len(Xtr), len(Xte)
        rf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=-1).fit(Xtr, ytr)
        out["accuracy"].append(float(rf.score(Xte, yte)))
        expl = _explainer(Xtr, names, classes, cat_idx, cat_names, seed)
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(Xte), size=min(n_explain, len(Xte)), replace=False)
        counts = np.zeros(n_feat)
        for j, i in enumerate(idx):
            label = int(rf.predict(Xte[i:i + 1])[0])
            exp = expl.explain_instance(Xte[i], rf.predict_proba, num_features=TOP,
                                        labels=(label,), num_samples=num_samples)
            counts[_top(exp, label)] += 1
            if seed == seeds[0] and j < keep_examples:
                out["examples"].append({
                    "true": classes[int(yte[i])], "pred": classes[label],
                    "proba": round(float(rf.predict_proba(Xte[i:i + 1])[0][label]), 4),
                    "weights": [[w, round(float(v), 4)] for w, v in exp.as_list(label=label)]})
        freq = counts / len(idx)
        out["freq_top5"].append(freq.tolist())
        pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=-1)
        out["spearman_vs_permutation"].append(float(spearmanr(freq, pi.importances_mean).statistic))

        if seed == seeds[0]:
            jac, top1 = [], []
            for i in idx[:n_stability]:
                label = int(rf.predict(Xte[i:i + 1])[0])
                sets = []
                for s in range(5):
                    e = _explainer(Xtr, names, classes, cat_idx, cat_names, 100 + s)
                    sets.append(_top(e.explain_instance(Xte[i], rf.predict_proba, num_features=TOP,
                                                        labels=(label,), num_samples=num_samples), label))
                jac += [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(sets, 2)]
                top1.append(np.mean([s[0] == sets[0][0] for s in sets[1:]]))
            out["stability"] = {"n_instances": len(idx[:n_stability]), "lime_seeds": 5,
                                "mean_jaccard_top5": float(np.mean(jac)),
                                "top1_agreement": float(np.mean(top1))}
        print(f"  seed {seed}: acc {out['accuracy'][-1]:.3f}, "
              f"rho {out['spearman_vs_permutation'][-1]:.2f}")

    freq = np.array(out["freq_top5"])
    order = np.argsort(-freq.mean(axis=0))
    out["summary"] = {
        "accuracy_mean": float(np.mean(out["accuracy"])), "accuracy_sd": float(np.std(out["accuracy"])),
        "spearman_mean": float(np.mean(out["spearman_vs_permutation"])),
        "spearman_sd": float(np.std(out["spearman_vs_permutation"])),
        "feature_freq_top5": [{"feature": names[k], "mean": round(float(freq[:, k].mean()), 3),
                               "sd": round(float(freq[:, k].std()), 3)} for k in order],
    }
    return out

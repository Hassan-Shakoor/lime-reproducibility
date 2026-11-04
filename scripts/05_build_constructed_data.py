"""Build the constructed dataset: real film reviews with an injected
spurious artefact at controlled strengths.

Base text: Pang & Lee (2004) polarity dataset v2.0, 1,000 positive and
1,000 negative real IMDb film reviews (replaces the templated pilot
corpus used in the project update).

The artefact is the meaningless token "zzqref", appended as a final
line, like the signature/footer lines that leak label information in
20 Newsgroups. For each seed, a stratified 80/20 split is drawn and
the token is added to:

  neg100 / neg75 / neg50 / neg25 : that share of NEGATIVE reviews only
                                   (token predicts the label, with
                                   decreasing coverage)
  control                         : 50% of ALL reviews, independent of
                                   the label (token carries no signal)

The same rule is applied to train and test, so the artefact is present
at test time exactly as headers are in the 20NG test split.

Outputs (committed to the repo):
  data/constructed/manifest.csv     one row per (seed, condition, review)
  data/constructed/<cond>_seed0.jsonl.gz  materialised seed-0 datasets
  data/constructed/README.md        datasheet
"""
import csv
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import train_test_split

from src.common import CONSTRUCTED, SEEDS, load_movie_reviews

TOKEN = "zzqref"
CONDITIONS = {"neg100": (1.0, 0.0), "neg75": (0.75, 0.0), "neg50": (0.5, 0.0),
              "neg25": (0.25, 0.0), "control": (0.5, 0.5)}  # (rate in neg, rate in pos)


def inject(text):
    return text.rstrip() + f"\n{TOKEN}\n"


def build(texts, labels, seed, cond):
    """Return list of dicts for one seed/condition (deterministic)."""
    p_neg, p_pos = CONDITIONS[cond]
    idx = np.arange(len(texts))
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=labels)
    split = np.empty(len(texts), dtype=object)
    split[tr], split[te] = "train", "test"
    rng = np.random.RandomState(1000 * seed + list(CONDITIONS).index(cond))
    injected = np.zeros(len(texts), dtype=bool)
    for s in ("train", "test"):
        for lab, p in ((0, p_neg), (1, p_pos)):
            pool = idx[(split == s) & (labels == lab)]
            k = int(round(p * len(pool)))
            injected[rng.choice(pool, size=k, replace=False)] = True
    return [{"doc": int(i), "label": int(labels[i]), "split": split[i], "injected": bool(injected[i])}
            for i in idx]


def main():
    texts, labels, ids = load_movie_reviews()
    rows = []
    for seed in SEEDS:
        for cond in CONDITIONS:
            for r in build(texts, labels, seed, cond):
                rows.append({"seed": seed, "condition": cond, "file": ids[r["doc"]], **r})
    with open(CONSTRUCTED / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for cond in CONDITIONS:
        recs = [r for r in rows if r["seed"] == 0 and r["condition"] == cond]
        with gzip.open(CONSTRUCTED / f"{cond}_seed0.jsonl.gz", "wt") as f:
            for r in recs:
                text = inject(texts[r["doc"]]) if r["injected"] else texts[r["doc"]]
                f.write(json.dumps({"id": ids[r["doc"]], "label": ["neg", "pos"][r["label"]],
                                    "split": r["split"], "artefact": r["injected"], "text": text}) + "\n")
        inj = [r for r in recs if r["injected"]]
        summary[cond] = {"injected": len(inj),
                         "injected_neg": sum(r["label"] == 0 for r in inj),
                         "injected_pos": sum(r["label"] == 1 for r in inj),
                         "P(neg|token)": round(sum(r["label"] == 0 for r in inj) / len(inj), 3)}
    print(json.dumps(summary, indent=2))
    (CONSTRUCTED / "summary_seed0.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(rows)} manifest rows and {len(CONDITIONS)} seed-0 datasets to {CONSTRUCTED}")


if __name__ == "__main__":
    main()

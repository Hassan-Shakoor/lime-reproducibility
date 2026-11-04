# Constructed dataset: real reviews with an injected spurious artefact

**Purpose.** Test, under controlled conditions, the paper's central claim:
that LIME exposes a model relying on a spurious feature, as with the e-mail
header words in 20 Newsgroups. Here the spurious feature and its strength are
known exactly.

**Base text.** Pang and Lee (2004) polarity dataset v2.0: 2,000 real film
reviews from IMDb (1,000 positive, 1,000 negative), labelled by the reviewers'
own ratings. Average length about 700 words.

**Construction** (`scripts/05_build_constructed_data.py`, deterministic):

1. For each seed in 0-4, draw a stratified 80/20 train/test split (1,600 / 400).
2. Append the meaningless token `zzqref` as a final line (like a forum
   signature or footer) to a chosen share of reviews, in train and test alike:

| Condition | Negative reviews with token | Positive reviews with token | P(negative given token) |
| --- | --- | --- | --- |
| `neg100` | 100% | 0% | 1.0 |
| `neg75` | 75% | 0% | 1.0 |
| `neg50` | 50% | 0% | 1.0 |
| `neg25` | 25% | 0% | 1.0 |
| `control` | 50% | 50% | 0.5 |

**Files.**

- `manifest.csv`: 50,000 rows (5 seeds x 5 conditions x 2,000 reviews):
  `seed, condition, file, doc, label, split, injected`. Together with the
  public base corpus this fully specifies every dataset version.
- `<condition>_seed0.jsonl.gz`: the seed-0 version of each condition,
  materialised (`id, label, split, artefact, text`).
- `summary_seed0.json`: injection counts.

**Labels and annotation.** Sentiment labels come from the original corpus,
which derived them from star ratings; no new human labels were needed. The
artefact is appended, not substituted, so no review text is changed.

**Licence and provenance.** The base corpus is distributed by its authors for
research use; this derivative adds only the token and the manifest.

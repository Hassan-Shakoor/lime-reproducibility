# COMP8240 LIME reproducibility project

Course project for **COMP8240 Applications of Data Science** (Macquarie University).

This repository reproduces the official LIME implementation from Ribeiro, Singh, and Guestrin, *“Why Should I Trust You? Explaining the Predictions of Any Classifier”* (KDD 2016). The goal is to confirm that individual predictions from a black-box classifier can be explained locally, then reuse that pipeline on new datasets.

**GitHub:** [Hassan-Shakoor/lime-reproducibility](https://github.com/Hassan-Shakoor/lime-reproducibility)

## Links

- Official LIME code: [marcotcr/lime](https://github.com/marcotcr/lime)
- Paper (arXiv): [1602.04938](https://arxiv.org/abs/1602.04938)
- Paper (KDD 2016): [ACM DL](https://dl.acm.org/doi/10.1145/2939672.2939778)

## Status

- [x] Local environment created (`venv`, Python 3.11.9)
- [x] Official `lime` package installed from source (`0.2.0.1`)
- [x] Tabular feasibility test (Iris + random forest)
- [x] Text feasibility test (20 Newsgroups + random forest)
- [ ] Assignment 1 proposal files added to `proposal/`
- [ ] Project datasets added to `data/`
- [ ] Experiment logs saved to `results/`

## Repository layout

```
comp8240-lime-project/
├── README.md
├── .gitignore
├── proposal/                       # Assignment 1 (tex + pdf + template)
├── scripts/
│   ├── test_lime_tabular.py        # Iris tabular explanation
│   └── test_lime_text.py           # 20 Newsgroups text explanation
├── data/                           # new datasets (empty for now)
├── results/                        # output logs (empty for now)
└── notes/
    └── feasibility_notes.md        # environment notes + recorded results
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install lime scikit-learn pandas numpy certifi
```

On Python 3.12+ you may hit `ModuleNotFoundError: No module named 'imp'` from the `pyDOE2` dependency. The feasibility run used Python 3.11.9 and did not need that patch. See `notes/feasibility_notes.md`.

## Run the feasibility tests

```bash
source venv/bin/activate
python3 scripts/test_lime_tabular.py
python3 scripts/test_lime_text.py
```

The text test downloads the 20 Newsgroups subset the first time it runs (internet required).

## What we confirmed

Both official tutorial settings work on this machine:

| Test | Data | Model | Score | What LIME returned |
|---|---|---|---|---|
| Tabular | Iris, instance 5 | Random forest (500 trees) | Accuracy 1.00 | Petal length ≤ 1.50 dominates the setosa prediction |
| Text | 20 Newsgroups, doc 83 | TF-IDF + random forest | F1 0.92 | Header tokens (`Host`, `Posting`, `NNTP`) drive the atheism call |

The text result is the paper’s well-known point: a high score can still rest on spurious cues. Full numbers and the macOS SSL download fix are in `notes/feasibility_notes.md`.

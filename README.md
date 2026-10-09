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
- [x] Tabular feasibility test (Iris + random forest, 500 trees)
- [x] Text feasibility test (20 Newsgroups + TF-IDF + random forest)
- [x] Heart Disease experiment (UCI Cleveland, 303 rows, RF 300)
- [x] Interactive architecture diagram (Archify)
- [ ] Assignment 1 proposal files added to `proposal/`
- [ ] Constructed corpus / artefact-injection experiment logged in `results/`

## Repository layout

```
comp8240-lime-project/
├── README.md
├── .gitignore
├── proposal/                          # Assignment 1 (placeholder)
├── scripts/
│   ├── test_lime_tabular.py           # Iris tabular explanation
│   ├── test_lime_text.py              # 20 Newsgroups text explanation
│   └── test_lime_heart_disease.py     # UCI Heart Disease explanation
├── data/                              # project datasets (empty for now)
├── results/                           # output logs (empty for now)
├── notes/
│   └── feasibility_notes.md           # environment notes + recorded results
└── .archify/
    └── architecture-lime-experiments-20261003-233639/
        ├── lime-experiments.html      # interactive architecture diagram
        └── candidate.json             # Archify source
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install lime scikit-learn pandas numpy certifi
```

On Python 3.12+ you may hit `ModuleNotFoundError: No module named 'imp'` from the `pyDOE2` dependency. The feasibility run used Python 3.11.9 and did not need that patch. See `notes/feasibility_notes.md`.

## Run the experiments

```bash
source venv/bin/activate
python3 scripts/test_lime_tabular.py
python3 scripts/test_lime_text.py
python3 scripts/test_lime_heart_disease.py
```

The text test downloads the 20 Newsgroups subset the first time it runs. The Heart Disease script downloads the Cleveland file from the UCI Machine Learning Repository. Both need internet on first run.

On some macOS Python.org installs, set SSL certificates via `certifi`. `scripts/test_lime_text.py` does this automatically.

## Architecture diagram

The interactive diagram is:

`.archify/architecture-lime-experiments-20261003-233639/lime-experiments.html`

Serve it over `http://` (not `file://`):

```bash
cd .archify/architecture-lime-experiments-20261003-233639
python3 -m http.server 8766
```

Then open [http://127.0.0.1:8766/lime-experiments.html](http://127.0.0.1:8766/lime-experiments.html).

**Live** walks the experiment hops one edge at a time. **Tutor** (or `U`) steps through Iris, 20 Newsgroups, Heart Disease, and the constructed-corpus branch.

The diagram shows:

| Path | Data | Model | Explainer |
|---|---|---|---|
| Feasibility tabular | Iris (bundled) | Random forest, 500 trees | `LimeTabularExplainer` |
| Feasibility text | 20 Newsgroups (Figshare) | TF-IDF + classifier | `LimeTextExplainer` |
| Project tabular | UCI Heart Disease, Cleveland 303 | Random forest, 300 trees | `LimeTabularExplainer` |
| Project text | Constructed corpus, 5 artefact levels | Logistic regression | `LimeTextExplainer` |

## What we confirmed

| Test | Data | Model | Score | What LIME returned |
|---|---|---|---|---|
| Tabular | Iris, instance 5 | Random forest (500 trees) | Accuracy 1.00 | Petal length ≤ 1.50 dominates the setosa prediction |
| Text | 20 Newsgroups, doc 83 | TF-IDF + random forest | F1 0.92 | Header tokens (`Host`, `Posting`, `NNTP`) drive the atheism call |

The text result is the paper’s well-known point: a high score can still rest on spurious cues. Full numbers and the macOS SSL download fix are in `notes/feasibility_notes.md`.

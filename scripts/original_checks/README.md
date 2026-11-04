# Original checks (proposal and project update)

These are the exact scripts run on macOS (Python 3.11.9, lime 0.2.0.1)
for the Assignment 1 feasibility check and the project update video,
kept unchanged for provenance.

| Script | Used in | Result shown |
| --- | --- | --- |
| `test_lime.py` | Proposal, video | Iris, random forest: accuracy 1.0; top LIME feature petal length (cm) <= 1.50, weight 0.4691 |
| `test_lime_text.py` | Proposal, video | 20 Newsgroups, Naive Bayes: F1 0.9199; document 83: Host -0.1575, Posting -0.1498, NNTP -0.0903 |
| `test_lime_heart_disease.py` | Video | UCI Heart Disease, one split (random_state=42): accuracy 0.9180 |

The scripts in `scripts/` extend these: `00_feasibility_*.py` reproduce
the first two, and `03_heart_disease.py` repeats the third over five
splits with stability checks.

# Feasibility notes

Recorded from the local run on **23 August 2026**.

## Environment

| Item | Value |
|---|---|
| OS | macOS (darwin 25.2.0) |
| Python | 3.11.9 |
| Package | `lime` 0.2.0.1, installed from the official source checkout |
| Other packages | scikit-learn 1.9.0, numpy, pandas, matplotlib, scipy, certifi |
| Virtualenv | `venv/` (not committed) |

The known Python 3.12 `imp` / `pyDOE2` bug did **not** appear on 3.11.9. `import pyDOE2` and `import lime` both succeeded without a patch.

## Environment fix: 20 Newsgroups download

The first text-test run failed while sklearn tried to download 20 Newsgroups from Figshare:

```
SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate
```

This is a macOS / Python.org certificate issue, not a LIME bug.

**Fix:** install `certifi` and point SSL at its CA bundle. `scripts/test_lime_text.py` does this automatically:

```python
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
```

After that, the download and the explanation both completed.

## Test 1 — tabular (Iris)

Script: `scripts/test_lime_tabular.py`

- Dataset: sklearn Iris, 80/20 split, `random_state=42`
- Model: `RandomForestClassifier(n_estimators=500, random_state=42)`
- Instance explained: test index `5`
- Explainer: `LimeTabularExplainer(..., discretize_continuous=True)`

```
Test accuracy: 1.0

True class: setosa
Predicted probabilities: [1. 0. 0.]

LIME explanation (feature, weight):
  petal length (cm) <= 1.50: 0.4669
  0.30 < petal width (cm) <= 1.30: -0.0409
  3.00 < sepal width (cm) <= 3.40: 0.0114
  5.10 < sepal length (cm) <= 5.75: 0.0049
```

**Interpretation:** the local explanation is dominated by petal length. Sepal measurements barely move the local linear model. This matches the official LIME tabular tutorial pattern (same top rule, weight around 0.46). Small decimal differences across reruns are expected because LIME samples perturbations randomly.

## Test 2 — text (20 Newsgroups)

Script: `scripts/test_lime_text.py`

- Dataset: 20 Newsgroups, categories `alt.atheism` vs `soc.religion.christian`
- Split: 1,079 train / 717 test
- Model: TF-IDF (`lowercase=False`) + `RandomForestClassifier(n_estimators=500, random_state=42)`
- Document explained: test id `83`
- Explainer: `LimeTextExplainer`

```
Test F1: 0.9199

Document id: 83
True class: atheism
Predicted probabilities: atheism=0.576, christian=0.424

LIME explanation (word, weight):
  Host: -0.1585
  Posting: -0.1495
  NNTP: -0.0902
  edu: -0.0404
  There: -0.0129
  anyone: 0.0108
```

**Interpretation:** the classifier is correct on this document, but the top words are newsgroup header artifacts (`Host`, `Posting`, `NNTP`), not religious content. Negative weights push the prediction toward atheism. This is the well-known LIME paper / tutorial result: a high F1 score can still hide an untrustworthy reason.

## Feasibility verdict

The official LIME code runs locally. A black-box random forest can be explained on both tabular and text inputs, and the text case shows LIME can surface spurious cues that accuracy misses. That is enough evidence to proceed with the COMP8240 project on new data.

## Not run yet

- Image explanations (`LimeImageExplainer`)
- Submodular pick
- Any project-specific dataset in `data/`

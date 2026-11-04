"""Shared helpers: paths, seeding, dataset download/loading.

All scripts import from here so data locations and random seeds are
defined in one place. Downloads are cached under data/raw/ and use the
certifi CA bundle, which fixes the macOS "CERTIFICATE_VERIFY_FAILED"
error encountered during the feasibility checks.
"""
import io
import json
import os
import random
import ssl
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CONSTRUCTED = ROOT / "data" / "constructed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
for _d in (RAW, CONSTRUCTED, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]

URLS = {
    "heart": "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "adult_train": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult_test": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "multi_domain": "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz",
    "religion": "https://github.com/marcotcr/lime-experiments/raw/master/religion_dataset.tar.gz",
    "movie_reviews": "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def _ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download(name, filename=None):
    """Download URLS[name] into data/raw/ once and return the local path."""
    url = URLS[name]
    path = RAW / (filename or url.rsplit("/", 1)[-1])
    if not path.exists():
        print(f"Downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "comp8240-lime-reproducibility"})
        with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as r:
            path.write_bytes(r.read())
    return path


def save_json(obj, name):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"Saved {path.relative_to(ROOT)}")
    return path


# ---------------------------------------------------------------- 20 Newsgroups

def load_20ng(remove=()):
    """Christianity vs Atheism subset used throughout Ribeiro et al. (2016).

    remove=() keeps headers (the paper's original setup);
    remove=('headers', 'footers', 'quotes') gives the cleaned variant.
    The default shuffle (random_state=42) keeps document order identical
    between the two variants, so document i is the same post in both.
    """
    from sklearn.datasets import fetch_20newsgroups
    ssl._create_default_https_context = _ssl_context
    cats = ["alt.atheism", "soc.religion.christian"]
    tr = fetch_20newsgroups(subset="train", categories=cats, remove=remove)
    te = fetch_20newsgroups(subset="test", categories=cats, remove=remove)
    return tr.data, tr.target, te.data, te.target, ["atheism", "christian"]


def load_religion():
    """Religion dataset built by Ribeiro et al. from DMOZ (819 pages/class).

    Label order matches load_20ng: 0 = atheism, 1 = christian.
    """
    path = download("religion")
    docs = {"atheism": {}, "christianity": {}}
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if m.isfile() and len(parts) == 2 and parts[0] in docs and parts[1].endswith(".txt"):
                docs[parts[0]][int(parts[1][:-4])] = tar.extractfile(m).read().decode("utf-8", errors="ignore")
    # The authors balanced the classes to 819 pages each (README.txt).
    n = min(len(docs["atheism"]), len(docs["christianity"]))
    texts, labels = [], []
    for label, cls in ((0, "atheism"), (1, "christianity")):
        for i in sorted(docs[cls])[:n]:
            texts.append(docs[cls][i])
            labels.append(label)
    return texts, np.array(labels)


# ---------------------------------------------------------- Multi-domain reviews

def load_multi_domain(domain):
    """Books / DVD reviews (Blitzer et al., 2007), processed exactly as in
    the authors' lime-experiments/load_datasets.py: unigrams only, each
    word repeated by its count, shuffled with seed 1, 80/20 split."""
    path = download("multi_domain")
    import re
    pos, neg = [], []

    def words(line):
        z = [tuple(x.split(":")) for x in re.findall(r"\w*?:\d", line)]
        return " ".join(" ".join([w] * int(c)) for w, c in z if "_" not in w)

    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            parts = m.name.split("/")
            if len(parts) >= 2 and parts[-2] == domain and parts[-1] in ("positive.review", "negative.review"):
                lines = tar.extractfile(m).read().decode("utf-8", errors="ignore").splitlines()
                (pos if parts[-1].startswith("positive") else neg).extend(words(l) for l in lines if l.strip())
    if not pos:
        raise FileNotFoundError(f"Domain '{domain}' not found in {path}")
    random.seed(1)
    random.shuffle(pos)
    random.shuffle(neg)
    sp, sn = int(len(pos) * 0.8), int(len(neg) * 0.8)
    train = pos[:sp] + neg[:sn]
    test = pos[sp:] + neg[sn:]
    ytr = np.array([1] * sp + [0] * sn)
    yte = np.array([1] * (len(pos) - sp) + [0] * (len(neg) - sn))
    return train, ytr, test, yte, ["neg", "pos"]


# ------------------------------------------------------------------ Tabular data

HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_heart():
    import pandas as pd
    df = pd.read_csv(download("heart"), header=None, names=HEART_COLS, na_values="?")
    return df


ADULT_COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
              "marital-status", "occupation", "relationship", "race", "sex",
              "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]


def load_adult():
    import pandas as pd
    tr = pd.read_csv(download("adult_train"), header=None, names=ADULT_COLS,
                     na_values="?", skipinitialspace=True)
    te = pd.read_csv(download("adult_test"), header=None, names=ADULT_COLS,
                     na_values="?", skipinitialspace=True, skiprows=1)
    df = pd.concat([tr, te], ignore_index=True).dropna()
    df["income"] = df["income"].str.replace(".", "", regex=False).map({"<=50K": 0, ">50K": 1})
    return df


# ------------------------------------------------------------------ Movie reviews

def load_movie_reviews():
    """Pang & Lee (2004) polarity v2.0: 1000 positive + 1000 negative real
    film reviews, from the NLTK data mirror."""
    path = download("movie_reviews")
    texts, labels, ids = [], [], []
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            if n.endswith(".txt") and ("/pos/" in n or "/neg/" in n):
                texts.append(z.read(n).decode("utf-8", errors="ignore"))
                labels.append(1 if "/pos/" in n else 0)
                ids.append(n.rsplit("/", 1)[-1])
    return texts, np.array(labels), ids

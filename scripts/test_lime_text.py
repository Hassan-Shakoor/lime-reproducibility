"""20 Newsgroups text feasibility test for LIME.

Downloads the atheism vs Christian subset, trains a TF-IDF random forest,
and explains one test document with LimeTextExplainer.

Requires internet on the first run. On some macOS Python installs, set
SSL certificates via certifi (already handled below).
"""

import os

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
except ImportError:
    pass

import sklearn
import sklearn.ensemble
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

categories = ["alt.atheism", "soc.religion.christian"]
print("Downloading 20 Newsgroups (atheism vs christian)...")
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
class_names = ["atheism", "christian"]
print(f"Train docs: {len(newsgroups_train.data)}, test docs: {len(newsgroups_test.data)}")

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(train_vectors, newsgroups_train.target)
pred = rf.predict(test_vectors)
f1 = sklearn.metrics.f1_score(newsgroups_test.target, pred, average="binary")
print(f"Test F1: {f1:.4f}")

pipeline = make_pipeline(vectorizer, rf)
explainer = LimeTextExplainer(class_names=class_names)

idx = 83
doc = newsgroups_test.data[idx]
proba = pipeline.predict_proba([doc])[0]
exp = explainer.explain_instance(doc, pipeline.predict_proba, num_features=6)

print(f"\nDocument id: {idx}")
print(f"True class: {class_names[newsgroups_test.target[idx]]}")
print(f"Predicted probabilities: atheism={proba[0]:.3f}, christian={proba[1]:.3f}")
print("\nLIME explanation (word, weight):")
for word, weight in exp.as_list():
    print(f"  {word}: {weight:.4f}")

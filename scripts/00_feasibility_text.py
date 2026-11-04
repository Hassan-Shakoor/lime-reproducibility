"""Feasibility check (Assignment 1): LIME on 20 Newsgroups, Naive Bayes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
from src.common import load_20ng

Xtr, ytr, Xte, yte, classes = load_20ng()
print(f"Train docs: {len(Xtr)}, test docs: {len(Xte)}")
model = make_pipeline(TfidfVectorizer(lowercase=False), MultinomialNB(alpha=0.01)).fit(Xtr, ytr)
print(f"Test F1: {f1_score(yte, model.predict(Xte)):.4f}")
idx = 83
exp = LimeTextExplainer(class_names=classes).explain_instance(Xte[idx], model.predict_proba, num_features=6)
p = model.predict_proba([Xte[idx]])[0]
print(f"Document id: {idx}\nTrue class: {classes[yte[idx]]}")
print(f"Predicted probabilities: atheism={p[0]:.3f}, christian={p[1]:.3f}")
for w, v in exp.as_list():
    print(f"  {w}: {v:.4f}")

"""
Apply LIME to the Heart Disease UCI dataset (Cleveland subset, 303 rows),
downloaded directly from the UCI Machine Learning Repository.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import lime.lime_tabular

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "heart-disease/processed.cleveland.data")
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

print("Downloading Heart Disease (Cleveland) dataset from UCI...")
df = pd.read_csv(URL, header=None, names=COLUMNS, na_values="?")
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum()[df.isna().sum() > 0].to_dict()}")

X = df.drop(columns="num")
y = (df["num"] > 0).astype(int)
print(f"Class balance (0 = no disease, 1 = disease): {y.value_counts().to_dict()}")

X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns)
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train, y_train)
print(f"\nTest accuracy: {rf.score(X_test, y_test):.4f}")

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train, feature_names=feature_names,
    class_names=["no disease", "disease"],
    discretize_continuous=True, random_state=42,
)

i = 3
exp = explainer.explain_instance(X_test[i], rf.predict_proba, num_features=6, top_labels=1)
label = rf.predict([X_test[i]])[0]

print("\nTrue class:", "disease" if y_test[i] == 1 else "no disease")
print("Predicted class:", "disease" if label == 1 else "no disease")
print("Predicted probabilities (no disease, disease):", rf.predict_proba([X_test[i]])[0])
print("\nLIME explanation (feature, weight):")
for feature, weight in exp.as_list(label=label):
    print(f"  {feature}: {weight:.4f}")

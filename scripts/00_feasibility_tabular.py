"""Feasibility check (Assignment 1): LIME on Iris with a random forest."""
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from lime.lime_tabular import LimeTabularExplainer

iris = sklearn.datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size=0.8, random_state=42)
rf = RandomForestClassifier(n_estimators=500, random_state=42).fit(X_train, y_train)
print("Test accuracy:", rf.score(X_test, y_test))
explainer = LimeTabularExplainer(X_train, feature_names=iris.feature_names,
                                 class_names=iris.target_names, discretize_continuous=True)
i = 5
exp = explainer.explain_instance(X_test[i], rf.predict_proba, num_features=4, top_labels=1)
label = rf.predict([X_test[i]])[0]
print("True class:", iris.target_names[y_test[i]])
print("Predicted probabilities:", rf.predict_proba([X_test[i]])[0])
for feature, weight in exp.as_list(label=label):
    print(f"  {feature}: {weight:.4f}")

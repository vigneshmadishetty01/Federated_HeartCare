import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load data
data = pd.concat([
    pd.read_csv("typical.csv"),
    pd.read_csv("athletic.csv"),
    pd.read_csv("diver.csv")
])

X = data.drop("target", axis=1)
y = data["target"]

# Encode categorical data
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col])

# Pipeline: Scaling + Random Forest
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("✔ Level-1 Centralized Accuracy:", accuracy)

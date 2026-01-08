import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

datasets = {
    "Typical": "typical.csv",
    "Athletic": "athletic.csv",
    "Diver": "diver.csv"
}

for name, file in datasets.items():
    data = pd.read_csv(file)
    X = data.drop("target", axis=1)
    y = data["target"]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, f"model_{name.lower()}.pkl")
    print(f"✔ Saved {name} model")

print("Models Saved")

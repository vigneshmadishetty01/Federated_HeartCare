import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# STEP 1: LOAD MODELS
# -------------------------------------------------

models = {
    "Typical": joblib.load("model_typical.pkl"),
    "Athletic": joblib.load("model_athletic.pkl"),
    "Diver": joblib.load("model_diver.pkl")
}

# -------------------------------------------------
# STEP 2: LOAD DATASETS
# -------------------------------------------------

datasets = {
    "Typical": "typical.csv",
    "Athletic": "athletic.csv",
    "Diver": "diver.csv"
}

# -------------------------------------------------
# STEP 3: DATA PREPROCESSING
# -------------------------------------------------

def preprocess(data):
    X = data.drop("target", axis=1)
    y = data["target"]

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col])

    return X, y

# -------------------------------------------------
# STEP 4: CONFIDENCE CALCULATION
# -------------------------------------------------

def prediction_confidence(model, X):
    probabilities = model.predict_proba(X)
    return np.mean(np.max(probabilities, axis=1))

# -------------------------------------------------
# STEP 5: TRUST SCORE
# -------------------------------------------------

def compute_trust(confidence, accuracy):
    return 0.5 * confidence + 0.5 * accuracy

# -------------------------------------------------
# STEP 6: TRUST-AWARE MODEL SWITCHING
# -------------------------------------------------

current_state = "Typical"                     # Current model
incoming_state = "Athletic"                   # New user behavior

data = pd.read_csv(datasets[incoming_state])
X_test, y_test = preprocess(data)

current_model = models[current_state]

confidence = prediction_confidence(current_model, X_test)
accuracy = accuracy_score(y_test, current_model.predict(X_test))
trust = compute_trust(confidence, accuracy)

print("\n🔍 TRUST EVALUATION")
print(f"Prediction Confidence : {confidence:.2f}")
print(f"Recent Accuracy       : {accuracy:.2f}")
print(f"Trust Score           : {trust:.2f}")

# -------------------------------------------------
# STEP 7: DECISION
# -------------------------------------------------

TRUST_THRESHOLD = 0.65

if trust < TRUST_THRESHOLD:
    print("\n⚠ Trust Degraded → Switching Model")
    current_state = incoming_state
else:
    print("\n✔ Trust Stable → Retaining Current Model")

print(f"\n✅ ACTIVE MODEL : {current_state}")

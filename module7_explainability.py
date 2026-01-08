import pandas as pd
import shap
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("model_typical.pkl")

# Load dataset
data = pd.read_csv("typical.csv")

X = data.drop("target", axis=1)

# Encode categorical columns
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# SHAP Explainer
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Show explanation for first instance
shap.plots.waterfall(shap_values[0])

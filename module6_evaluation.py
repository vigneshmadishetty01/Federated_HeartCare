import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load test dataset (Athletic user after drift)
test_data = pd.read_csv("athletic.csv")

X_test = test_data.drop("target", axis=1)
y_test = test_data["target"]

# Encode categorical features
for col in X_test.columns:
    if X_test[col].dtype == "object":
        le = LabelEncoder()
        X_test[col] = le.fit_transform(X_test[col])

# Load models
typical_model = joblib.load("model_typical.pkl")
athletic_model = joblib.load("model_athletic.pkl")

# Accuracy BEFORE model swapping
y_pred_before = typical_model.predict(X_test)
accuracy_before = accuracy_score(y_test, y_pred_before)

# Accuracy AFTER model swapping
y_pred_after = athletic_model.predict(X_test)
accuracy_after = accuracy_score(y_test, y_pred_after)

print("Accuracy before model swapping:", accuracy_before)
print("Accuracy after model swapping:", accuracy_after)

# -----------------------------
# Enhanced Visualization
# -----------------------------
plt.figure(figsize=(7, 5))

bars = plt.bar(
    ["Before Swap", "After Swap"],
    [accuracy_before, accuracy_after],
    width=0.5
)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.01,
        f"{height:.2f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

# Styling
plt.ylim(0, 1.05)
plt.ylabel("Accuracy", fontsize=11)
plt.xlabel("Model State", fontsize=11)
plt.title("Accuracy Improvement After Model Swapping", fontsize=13)

plt.grid(axis="y", linestyle="--", alpha=0.6)

# Annotation explaining improvement
plt.annotate(
    "Model adapted\nto new user profile",
    xy=(1, accuracy_after),
    xytext=(0.5, accuracy_after - 0.25),
    arrowprops=dict(arrowstyle="->"),
    ha="center"
)

plt.tight_layout()
plt.show()

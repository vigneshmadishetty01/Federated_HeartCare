import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("heart.csv")

print("Available columns:", list(data.columns))

# Priority-based physiological column detection
heart_rate_cols = ["thalach", "MaxHR", "max_heart_rate", "Max_heart_rate"]
bp_cols = ["trestbps", "resting_blood_pressure"]
chol_cols = ["chol", "cholesterol", "cholestoral"]

phys_col = None

for col in heart_rate_cols:
    if col in data.columns:
        phys_col = col
        break

if phys_col is None:
    for col in bp_cols:
        if col in data.columns:
            phys_col = col
            break

if phys_col is None:
    for col in chol_cols:
        if col in data.columns:
            phys_col = col
            break

if phys_col is None:
    raise ValueError("No suitable physiological column found")

print(f"✔ Using column for simulation: {phys_col}")

def simulate_users(data, phys_col):
    typical = data.copy()
    athletic = data.copy()
    diver = data.copy()

    typical["user_type"] = "Typical"

    # Athletic users – improved cardiovascular efficiency
    athletic[phys_col] = athletic[phys_col] - np.random.randint(5, 15, size=len(athletic))
    athletic["user_type"] = "Athletic"

    # Diver users – stronger physiological adaptation
    diver[phys_col] = diver[phys_col] - np.random.randint(10, 25, size=len(diver))
    diver["user_type"] = "Diver"

    return typical, athletic, diver

typical, athletic, diver = simulate_users(data, phys_col)

# Save datasets
typical.to_csv("typical.csv", index=False)
athletic.to_csv("athletic.csv", index=False)
diver.to_csv("diver.csv", index=False)

print("✔ Module 1 Completed Successfully")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load client datasets
clients = {
    "Typical": pd.read_csv("typical.csv"),
    "Athletic": pd.read_csv("athletic.csv"),
    "Diver": pd.read_csv("diver.csv")
}

def preprocess_data(data):
    X = data.drop("target", axis=1)
    y = data["target"]

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])

    return X, y

# 🔐 Differential Privacy Noise Function
def add_dp_noise(weights, epsilon=0.5):
    noise = np.random.laplace(0, 1/epsilon, weights.shape)
    return weights + noise

def train_local_model(data):
    X, y = preprocess_data(data)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # 🔐 Apply Differential Privacy
    noisy_weights = add_dp_noise(model.coef_)
    return noisy_weights, model.intercept_

# Federated rounds
rounds = 3

for r in range(rounds):
    print(f"\n🔄 Federated Round {r+1} (DP Enabled)")
    weights, biases = [], []

    for name, data in clients.items():
        w, b = train_local_model(data)
        weights.append(w)
        biases.append(b)
        print(f"✔ Client {name} trained with DP")

    global_weights = np.mean(weights, axis=0)
    global_bias = np.mean(biases, axis=0)

print("\n✔ Federated Learning with Differential Privacy Completed")

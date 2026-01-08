import joblib

models = {
    "Typical": joblib.load("model_typical.pkl"),
    "Athletic": joblib.load("model_athletic.pkl"),
    "Diver": joblib.load("model_diver.pkl")
}

current_state = "Typical"
current_model = models[current_state]

def swap_model(new_state):
    global current_state, current_model
    current_state = new_state
    current_model = models[new_state]
    print(f"🔁 Model swapped to {new_state}")

# Simulated drift response
swap_model("Athletic")

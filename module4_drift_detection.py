from river.drift import ADWIN
import numpy as np

drift_detector = ADWIN()

# Simulated heart rate stream
heart_rate_stream = np.random.normal(70, 2, 100)

# Introduce concept drift
heart_rate_stream[50:] += 20

for i, rate in enumerate(heart_rate_stream):
    drift = drift_detector.update(rate)
    if drift:
        print(f"⚠ Concept Drift Detected at index {i}")
        break
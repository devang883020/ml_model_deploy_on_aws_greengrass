import pickle
import numpy as np

# Load model
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample input (4 features)
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

# Predict
prediction = model.predict(sample)
print("Prediction:", prediction)


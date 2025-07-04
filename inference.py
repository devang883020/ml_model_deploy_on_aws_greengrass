import pickle
import numpy as np
import os

def load_model(model_path="iris_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def run_inference(model):
    # Sample input: sepal length, sepal width, petal length, petal width
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = model.predict(sample)
    return prediction

if __name__ == "__main__":
    try:
        print("📦 Loading model...")
        model = load_model()
        print("🤖 Running prediction...")
        result = run_inference(model)
        print(f"✅ Prediction: {result}")
    except Exception as e:
        print(f"❌ Error occurred during inference: {str(e)}")


import pickle
import json
import numpy as np


# =========================
# 1. Load model
# =========================
def model_fn(model_dir):
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


# =========================
# 2. Parse input
# =========================
def input_fn(request_body, content_type):
    print("🔥 INPUT RECEIVED:", request_body)

    # Convert bytes → string
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")

    # Handle empty input
    if not request_body or request_body.strip() == "":
        raise ValueError("Empty request body")

    # Parse JSON
    try:
        data = json.loads(request_body)
    except Exception:
        raise ValueError(f"Invalid JSON: {request_body}")

    # Validate format
    if "inputs" not in data:
        raise ValueError(f"Expected 'inputs' key, got: {data}")

    arr = np.array(data["inputs"], dtype=float)

    # Ensure correct shape
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    return arr


# =========================
# 3. Predict
# =========================
def predict_fn(input_data, model):
    preds = model.predict(input_data)
    probs = model.predict_proba(input_data)

    return {
        "prediction": preds.tolist(),
        "probability": probs.tolist()
    }


# =========================
# 4. Return output
# =========================
def output_fn(prediction, accept):
    return json.dumps(prediction)
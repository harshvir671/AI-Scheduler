

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np

from ml.train import FEATURE_COLUMNS, MODEL_PATH, ENCODER_PATH
from ml.generate_dataset import compute_features

_model = None
_encoder = None


def _load():
    global _model, _encoder
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python ml/train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _encoder


def predict_from_features(features: dict) -> dict:
    """features must contain all keys in FEATURE_COLUMNS."""
    model, encoder = _load()
    x = np.array([[features[c] for c in FEATURE_COLUMNS]])
    proba = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    predicted_label = encoder.inverse_transform([pred_idx])[0]

    ranked = sorted(
        zip(encoder.classes_, proba), key=lambda kv: -kv[1]
    )

    return {
        "recommended_algorithm": predicted_label,
        "confidence": float(proba[pred_idx]),
        "all_probabilities": {cls: float(p) for cls, p in ranked},
        "features_used": features,
    }


def predict_from_processes(processes) -> dict:
    """processes: list of scheduler.process_generator.Process"""
    features = compute_features(processes)
    return predict_from_features(features)


def model_is_available() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH)


if __name__ == "__main__":
    from scheduler.process_generator import generate_processes

    procs = generate_processes(30, max_burst=15, seed=123)
    result = predict_from_processes(procs)
    print("Recommended:", result["recommended_algorithm"])
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print("All probabilities:")
    for cls, p in result["all_probabilities"].items():
        print(f"  {cls:10s} {p*100:.1f}%")

"""
==============================================================================
UniSage - University Query Management System (UQMS)
Machine Learning Query Classifier Helper (query_classifier.py)
==============================================================================
Provides helper functions to:
1. Load and cache the trained Scikit-learn Pipeline in memory.
2. Avoid retraining or reloading from disk on every incoming request.
3. Accept a university query string and return its predicted category.
4. Gracefully handle missing model files with clear setup instructions
   instead of crashing the Flask application.
==============================================================================
"""

import os
import joblib

# Default path to the saved model artifact
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "query_classifier.joblib"
)

# In-memory cached model instance (loaded once, reused for all requests)
_CACHED_MODEL = None
_CACHED_MODEL_PATH = None


def get_model_path(custom_path=None):
    """Returns the absolute path to the query classifier joblib file."""
    if custom_path:
        return os.path.abspath(custom_path)
    return DEFAULT_MODEL_PATH


def check_model_status(model_path=None):
    """
    Checks if the trained model artifact exists on disk.
    Returns a dictionary describing the current status and setup instructions.
    """
    path = get_model_path(model_path)
    if os.path.exists(path):
        size_bytes = os.path.getsize(path)
        return {
            "available": True,
            "path": path,
            "size_bytes": size_bytes,
            "message": f"Trained classifier model found ({size_bytes / 1024:.1f} KB)."
        }
    return {
        "available": False,
        "path": path,
        "size_bytes": 0,
        "message": (
            f"Classifier model artifact not found at '{path}'. "
            "Please train the model first by executing: python train_model.py"
        )
    }


def load_model(model_path=None, force_reload=False):
    """
    Loads the trained model pipeline into memory once and caches it.
    
    If the model file does not exist, returns None and logs a clear setup message
    without crashing the application.
    """
    global _CACHED_MODEL, _CACHED_MODEL_PATH
    target_path = get_model_path(model_path)

    # Return cached instance if already loaded and path hasn't changed
    if _CACHED_MODEL is not None and _CACHED_MODEL_PATH == target_path and not force_reload:
        return _CACHED_MODEL

    if not os.path.exists(target_path):
        print(f"[!] UniSage ML Setup Notice: Model file '{target_path}' does not exist.")
        print("    -> To create the model, please run: python train_model.py")
        _CACHED_MODEL = None
        _CACHED_MODEL_PATH = None
        return None

    try:
        _CACHED_MODEL = joblib.load(target_path)
        _CACHED_MODEL_PATH = target_path
        print(f"[*] UniSage ML: Successfully loaded model from '{target_path}'.")
        return _CACHED_MODEL
    except Exception as exc:
        print(f"[!] UniSage ML Setup Notice: Could not load model from '{target_path}': {exc}")
        print("    -> Try re-training the model by running: python train_model.py")
        _CACHED_MODEL = None
        _CACHED_MODEL_PATH = None
        return None


def predict_category(question, model_path=None):
    """
    Accepts a question string and returns the predicted university category.

    Categories include:
    - Scholarship
    - Fees
    - Examination
    - Admission
    - Department
    - General

    Returns:
        str: The predicted category name (e.g. 'Scholarship', 'Fees'),
             or a descriptive setup message if the model is not yet trained.
    """
    if not question or not str(question).strip():
        return "General"

    model = load_model(model_path)
    if model is None:
        return "Model Not Trained (Please run python train_model.py)"

    cleaned_q = str(question).strip()
    prediction = model.predict([cleaned_q])[0]
    return str(prediction)


def predict_category_details(question, model_path=None):
    """
    Accepts a question string and returns a detailed classification result,
    including confidence score and status information.
    """
    cleaned_q = str(question).strip() if question else ""
    if not cleaned_q:
        return {
            "success": False,
            "question": "",
            "category": "General",
            "confidence": 0.0,
            "message": "Question was empty. Defaulted to General."
        }

    model = load_model(model_path)
    if model is None:
        status_info = check_model_status(model_path)
        return {
            "success": False,
            "question": cleaned_q,
            "category": "Unknown",
            "confidence": 0.0,
            "message": status_info["message"]
        }

    prediction = model.predict([cleaned_q])[0]
    
    # Calculate confidence probability if classifier supports predict_proba
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_q])[0]
        confidence = float(max(probabilities))

    return {
        "success": True,
        "question": cleaned_q,
        "category": str(prediction),
        "confidence": round(confidence, 4),
        "message": f"Successfully classified question under '{prediction}'."
    }

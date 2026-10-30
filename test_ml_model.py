"""
==============================================================================
UniSage - Machine Learning Model Verification Suite (test_ml_model.py)
==============================================================================
Comprehensive automated tests verifying:
1. Dataset loading, schema validation, and sample labels in data/all_data.csv.
2. Presence and distribution of all required categories.
3. Model artifact existence and Scikit-learn Pipeline architecture
   (TfidfVectorizer + LogisticRegression).
4. Helper functions predict_category() and predict_category_details().
5. Correct classification for core benchmark and prompt example questions.
6. Graceful missing model handling without crashing.
7. Flask in-memory model loading, caching, and diagnostic API endpoints.
==============================================================================
"""

import os
import sys
import tempfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import query_classifier
import app


def run_ml_tests():
    print("=" * 65)
    print("      UniSage Machine Learning Model Verification Suite")
    print("=" * 65)

    passed_count = 0
    total_tests = 8

    # ------------------------------------------------------------------------
    # TEST 1: Dataset Existence & Schema Validation
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Verifying dataset existence and required columns in data/all_data.csv...")
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_data.csv")
    assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"

    df = pd.read_csv(dataset_path)
    assert "question" in df.columns, "Missing 'question' column in dataset!"
    assert "category" in df.columns, "Missing 'category' column in dataset!"
    assert "answer" in df.columns, "Missing 'answer' column in dataset!"
    assert len(df) > 100, f"Dataset too small ({len(df)} rows)"

    # Verify no empty questions or categories
    assert df["question"].dropna().str.strip().ne("").all(), "Found empty questions in dataset!"
    assert df["category"].dropna().str.strip().ne("").all(), "Found empty categories in dataset!"
    print(f"  -> PASS: Dataset validated ({len(df)} rows, columns: {list(df.columns)})")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 2: Category Distribution
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Verifying all required categories in dataset...")
    expected_categories = {"Scholarship", "Fees", "Examination", "Admission", "Department", "General"}
    present_categories = set(df["category"].unique())
    missing_categories = expected_categories - present_categories
    assert not missing_categories, f"Missing categories: {missing_categories}"
    print(f"  -> PASS: All 6 categories present: {present_categories}.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 3: Model Artifact & Pipeline Structure
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Verifying saved model artifact and Scikit-learn Pipeline components...")
    model_path = query_classifier.get_model_path()
    assert os.path.exists(model_path), f"Model artifact not found at {model_path}!"

    model = joblib.load(model_path)
    assert isinstance(model, Pipeline), f"Model is not a Scikit-learn Pipeline (got {type(model)})"
    assert "tfidf" in model.named_steps, "Pipeline missing 'tfidf' vectorizer step!"
    assert isinstance(model.named_steps["tfidf"], TfidfVectorizer), "First step must be TfidfVectorizer"
    assert "clf" in model.named_steps, "Pipeline missing 'clf' classifier step!"
    assert isinstance(model.named_steps["clf"], LogisticRegression), "Classifier step must be LogisticRegression"

    size_kb = os.path.getsize(model_path) / 1024
    print(f"  -> PASS: Valid Pipeline saved at '{model_path}' ({size_kb:.1f} KB).")
    print(f"  -> PASS: Pipeline steps = {list(model.named_steps.keys())} (TfidfVectorizer -> LogisticRegression)")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 4: Helper Function predict_category() on Benchmark Examples
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Testing helper function predict_category() on key benchmark questions...")
    test_cases = [
        ("What is the scholarship application process?", "Scholarship"),
        ("What is the examination fee?", "Fees"),
        ("When will the semester exam begin?", "Examination"),
        ("How can I apply for admission?", "Admission"),
        ("Where is the computer science department?", "Department"),
        ("What are the university library timings and rules?", "General"),
        ("How do I pay semester tuition fees online?", "Fees"),
        ("Can I apply for a merit scholarship with 95% marks?", "Scholarship"),
    ]

    for query, expected_cat in test_cases:
        predicted = query_classifier.predict_category(query)
        assert predicted == expected_cat, f"For '{query}': expected '{expected_cat}', got '{predicted}'"
        print(f"  -> PASS: \"{query[:40]}...\" -> [{predicted}]")

    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 5: Detailed Prediction Helper Function
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing predict_category_details() with confidence score...")
    details = query_classifier.predict_category_details("What is the examination fee?")
    assert details["success"] is True
    assert details["category"] == "Fees"
    assert 0.0 <= details["confidence"] <= 1.0
    assert "Fees" in details["message"]
    print(f"  -> PASS: Category='{details['category']}', Confidence={details['confidence'] * 100:.1f}%")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 6: Graceful Handling of Missing Model File (No Crashes!)
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Testing graceful handling of missing model file...")
    dummy_non_existent = os.path.join(tempfile.gettempdir(), "non_existent_model_xyz.joblib")
    if os.path.exists(dummy_non_existent):
        os.remove(dummy_non_existent)

    # 1. check_model_status reports unavailable
    status_info = query_classifier.check_model_status(dummy_non_existent)
    assert status_info["available"] is False
    assert "python train_model.py" in status_info["message"]

    # 2. load_model returns None without exception
    loaded = query_classifier.load_model(dummy_non_existent)
    assert loaded is None

    # 3. predict_category returns fallback message without exception
    pred_msg = query_classifier.predict_category("Sample query", model_path=dummy_non_existent)
    assert "Model Not Trained" in pred_msg

    print("  -> PASS: Missing model file handled gracefully with clear setup instructions; zero crashes.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 7: In-Memory Model Caching in Flask Backend
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing Flask backend in-memory caching (no retraining per request)...")
    # Verify that query_classifier_model in app is already loaded in memory
    assert app.query_classifier_model is not None, "Model was not preloaded in app.py!"
    
    # Check helper in app.py
    app_pred = app.predict_question_category("What is the scholarship application process?")
    assert app_pred == "Scholarship"

    # Subsequent loads return identical object reference without disk reload
    cached_ref_1 = query_classifier.load_model()
    cached_ref_2 = query_classifier.load_model()
    assert cached_ref_1 is cached_ref_2, "Model instance was not properly cached in memory!"
    print("  -> PASS: Model cached in memory; subsequent queries reuse cached instance.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 8: Flask Diagnostic Routes /api/model-status and /api/predict-category
    # ------------------------------------------------------------------------
    print("\n[TEST 8] Testing Flask diagnostic endpoints /api/model-status and /api/predict-category...")
    client = app.app.test_client()

    # GET /api/model-status
    resp_status = client.get("/api/model-status")
    assert resp_status.status_code == 200
    json_status = resp_status.get_json()
    assert json_status["available"] is True

    # GET /api/predict-category?question=What+is+the+examination+fee%3F
    resp_pred_get = client.get("/api/predict-category?question=What+is+the+examination+fee%3F")
    assert resp_pred_get.status_code == 200
    json_pred_get = resp_pred_get.get_json()
    assert json_pred_get["success"] is True
    assert json_pred_get["category"] == "Fees"

    # POST /api/predict-category with JSON payload
    resp_pred_post = client.post("/api/predict-category", json={
        "question": "Where is the computer science department?"
    })
    assert resp_pred_post.status_code == 200
    json_pred_post = resp_pred_post.get_json()
    assert json_pred_post["success"] is True
    assert json_pred_post["category"] == "Department"

    print("  -> PASS: Flask endpoints /api/model-status and /api/predict-category functional.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"     ALL {passed_count}/{total_tests} ML VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_ml_tests()

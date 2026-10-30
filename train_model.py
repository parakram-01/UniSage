"""
==============================================================================
UniSage - University Query Management System (UQMS)
Machine Learning Model Training Script (train_model.py)
==============================================================================
This script trains a lightweight, beginner-friendly Scikit-learn classification
model to categorize university queries into topics:
  - Scholarship
  - Fees
  - Examination
  - Admission
  - Department
  - General

PIPELINE ARCHITECTURE:
  1. Text Preprocessing & Feature Extraction:
     - TfidfVectorizer (Word unigrams & bigrams, sublinear TF scaling)
  2. Classification:
     - LogisticRegression (L2 regularization, balanced class weights)
  3. Artifact Persistence:
     - Saved using Joblib to 'models/query_classifier.joblib'

IMPORTANT NOTICE:
  This ML model performs statistical category prediction based on text patterns.
  It does NOT independently understand university policies or replace official
  university notices. All sample answers are for illustrative purposes and
  must be verified against official university resources.
==============================================================================
"""

import os
import sys
import argparse
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


# ----------------------------------------------------------------------------
# 1. Helper Function: Single Question Prediction
# ----------------------------------------------------------------------------
def predict_category(question, model):
    """
    Accepts a single question string and returns the predicted category.

    Args:
        question (str): The university inquiry text.
        model (Pipeline): The trained Scikit-learn Pipeline.

    Returns:
        str: Predicted category label (e.g., 'Fees', 'Scholarship').
    """
    if not question or not str(question).strip():
        return "General"
    cleaned = str(question).strip()
    prediction = model.predict([cleaned])[0]
    return str(prediction)


# ----------------------------------------------------------------------------
# 2. Data Loading & Validation
# ----------------------------------------------------------------------------
def load_and_validate_dataset(csv_path="data/all_data.csv"):
    """
    Loads dataset from CSV, validates required columns, and cleans missing/empty rows.

    Args:
        csv_path (str): Path to CSV dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing 'question' and 'category'.
    """
    print(f"\n[1/5] Loading dataset from: {csv_path}")

    # Fallback check if default all_data.csv does not exist
    if not os.path.exists(csv_path):
        data_dir = os.path.dirname(csv_path) or "data"
        fallback_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".csv") and f != "all_data.csv"
        ] if os.path.exists(data_dir) else []

        if fallback_files:
            print(f"  -> '{csv_path}' not found. Combining {len(fallback_files)} category CSVs from '{data_dir}'...")
            dfs = [pd.read_csv(f) for f in fallback_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError(
                f"Dataset file not found at '{csv_path}'. Please ensure data files exist in the 'data/' folder."
            )
    else:
        df = pd.read_csv(csv_path)

    initial_count = len(df)
    print(f"  -> Initial rows loaded: {initial_count}")

    # Validate required columns exist
    required_columns = {"question", "category"}
    actual_columns = set(df.columns)
    missing = required_columns - actual_columns
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    # Clean rows: remove NaN or whitespace-only questions and categories
    df = df.dropna(subset=["question", "category"]).copy()
    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()

    # Drop empty strings
    df = df[(df["question"] != "") & (df["category"] != "")]

    # Remove exact duplicate questions to prevent data leakage between train/test
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)
    cleaned_count = len(df)
    dropped_count = initial_count - cleaned_count

    print(f"  -> Cleaned rows: {cleaned_count} (removed {dropped_count} invalid or duplicate rows)")
    print("\n[2/5] Category Distribution:")
    counts = df["category"].value_counts()
    for cat, count in counts.items():
        percentage = (count / cleaned_count) * 100
        print(f"  - {cat:<15}: {count:>5} samples ({percentage:>5.1f}%)")

    return df


# ----------------------------------------------------------------------------
# 3. Model Pipeline Factory
# ----------------------------------------------------------------------------
def build_pipeline(random_state=42):
    """
    Constructs a Scikit-learn Pipeline with:
      - TfidfVectorizer: extracts term frequency-inverse document frequency features
      - LogisticRegression: classifies questions into university topics
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),        # Captures single words and two-word combinations
            stop_words="english",      # Removes common filler words like 'the', 'is'
            max_features=5000,         # Caps vocabulary for compact model size
            sublinear_tf=True          # Sublinear TF scaling (1 + log(tf))
        )),
        ("clf", LogisticRegression(
            max_iter=1000,             # Ensures convergence
            class_weight="balanced",   # Compensates for varying category sizes
            random_state=random_state
        ))
    ])


# ----------------------------------------------------------------------------
# 4. Model Training & Evaluation
# ----------------------------------------------------------------------------
def train_and_evaluate(df, test_size=0.2, random_state=42):
    """
    Splits data when possible, trains the pipeline, and evaluates test performance.

    Returns:
        tuple: (trained_pipeline, accuracy_score_or_none)
    """
    X = df["question"]
    y = df["category"]

    class_counts = y.value_counts()
    min_class_count = class_counts.min()

    # Check whether the dataset has enough examples per category for a stratified train/test split
    can_stratify = min_class_count >= 2 and len(df) >= 10

    if can_stratify:
        print(f"\n[3/5] Splitting data into Training ({(1-test_size)*100:.0f}%) and Testing ({test_size*100:.0f}%) sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        print(f"  -> Training samples: {len(X_train)}")
        print(f"  -> Testing samples:  {len(X_test)}")
    else:
        print("\n[3/5] Dataset size or class counts too small for stratified split.")
        print("  -> Training on all available data.")
        X_train, y_train = X, y
        X_test, y_test = None, None

    # Instantiate pipeline and fit
    print("\n[4/5] Training Scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)...")
    pipeline = build_pipeline(random_state=random_state)
    pipeline.fit(X_train, y_train)
    print("  -> Training completed successfully!")

    # Evaluation
    acc = None
    if X_test is not None and len(X_test) > 0:
        print("\n" + "=" * 65)
        print("                 MODEL EVALUATION METRICS")
        print("=" * 65)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Overall Test Accuracy: {acc * 100:.2f}%\n")
        report = classification_report(y_test, y_pred, digits=4, zero_division=0)
        print(report)
        print("=" * 65)

    return pipeline, acc


# ----------------------------------------------------------------------------
# 5. Save Artifact & Verification Test
# ----------------------------------------------------------------------------
def save_model(pipeline, output_path="models/query_classifier.joblib"):
    """
    Saves the trained pipeline using Joblib to disk.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(pipeline, output_path)
    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"\n[5/5] Model successfully saved to: {output_path} ({file_size_kb:.1f} KB)")


def run_sample_predictions(pipeline):
    """
    Tests the trained model on key benchmark questions, including prompt examples.
    """
    test_queries = [
        "What is the scholarship application process?",
        "What is the examination fee?",
        "When will the semester exam begin?",
        "How can I apply for admission?",
        "Where is the computer science department?",
        "What are the university library timings and rules?",
        "How do I pay my semester tuition fee via UPI?",
        "Can I get a fee waiver for high marks?",
        "Where is the office of the controller of examinations?"
    ]

    print("\n" + "=" * 65)
    print("             SAMPLE PREDICTIONS VERIFICATION")
    print("=" * 65)
    for query in test_queries:
        pred = predict_category(query, pipeline)
        # Confidence score
        probs = pipeline.predict_proba([query])[0]
        confidence = max(probs) * 100
        print(f"  Question  : \"{query}\"")
        print(f"  Prediction: [{pred}] (Confidence: {confidence:.1f}%)\n")
    print("=" * 65)


# ----------------------------------------------------------------------------
# Main Execution Entry Point
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train the UniSage Machine Learning Query Classification Model."
    )
    parser.add_argument(
        "--data",
        default="data/all_data.csv",
        help="Path to the training CSV file (default: data/all_data.csv)"
    )
    parser.add_argument(
        "--output",
        default="models/query_classifier.joblib",
        help="Path to save the trained model (default: models/query_classifier.joblib)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction for evaluation (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("     UniSage (UQMS) - Query Classifier Model Training")
    print("=" * 65)

    try:
        # Step 1 & 2: Load and validate dataset
        df = load_and_validate_dataset(csv_path=args.data)

        # Step 3 & 4: Train and evaluate
        pipeline, accuracy = train_and_evaluate(
            df,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Step 5: Save model
        save_model(pipeline, output_path=args.output)

        # Run sample predictions
        run_sample_predictions(pipeline)

        print("\n[SUCCESS] Model training and verification complete!")
        print("Note: The model predicts categories from text patterns.")
        print("Official university policies and records must always be consulted.")

    except Exception as exc:
        print(f"\n[ERROR] Model training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

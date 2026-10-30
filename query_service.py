"""
==============================================================================
UniSage - University Query Management System (UQMS)
Chatbot Query & Answer Service (query_service.py)
==============================================================================
This module bridges the Scikit-learn Machine Learning classifier and the
university CSV dataset to provide relevant sample answers:

1. Uses the trained TF-IDF + LogisticRegression model to predict category.
2. Checks term activation and confidence to detect unrelated / unreliable queries.
3. Finds the most appropriate sample answer from the CSV dataset using
   cosine similarity matching within the predicted category.
4. Provides curated category fallback guidance when similarity is low.
5. Attaches an official disclaimer reminding students that answers are sample
   data that must be confirmed via official university notices.
==============================================================================
"""

import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import query_classifier

# Paths to data and model
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "all_data.csv")

# Standard fallback message for unrelated or unconfident queries
UNRELIABLE_FALLBACK_MESSAGE = (
    "I am not confident that this question relates to our university information. "
    "Please try rephrasing your question using university keywords (e.g., fees, exams, "
    "scholarships, admission, departments), or contact the appropriate university "
    "administrative office for official assistance."
)

# Official verification disclaimer attached to all query responses
OFFICIAL_DISCLAIMER = (
    "This response is based on sample data for illustrative purposes. "
    "Please consult official university notices and administrative offices for definitive policy information."
)

# Curated fallback answers for each university category
CATEGORY_DEFAULT_ANSWERS = {
    "Scholarship": (
        "Please check the university scholarship portal for active application dates, "
        "eligibility criteria, and required documents. Inquiries can also be made at "
        "the Scholarship & Student Welfare Cell."
    ),
    "Fees": (
        "Tuition, examination, and hostel fees can be confirmed through the latest "
        "university fee circular or your student ERP dashboard. You may also contact "
        "the Accounts & Finance Office."
    ),
    "Examination": (
        "Please consult the latest examination timetable and circulars published on "
        "the student portal by the Controller of Examinations for official schedules "
        "and hall ticket dates."
    ),
    "Admission": (
        "Please visit the official university admission portal or contact the Central "
        "Admission Cell for eligibility requirements, prospectus details, and application deadlines."
    ),
    "Department": (
        "Please consult the campus directory for department office locations, faculty contacts, "
        "and Head of Department (HOD) consultation hours."
    ),
    "General": (
        "For general university facilities, library hours, identity card renewal, or campus "
        "holidays, please refer to the student handbook or visit the University Helpdesk."
    )
}

# Module-level cached knowledge base
_CACHED_DF = None
_CACHED_QUESTION_MATRIX = None
_CACHED_TFIDF = None


def get_knowledge_base():
    """
    Loads and caches the university CSV dataset and pre-computes TF-IDF vectors
    for all questions in memory. This avoids reading the CSV from disk on every query.
    """
    global _CACHED_DF, _CACHED_QUESTION_MATRIX, _CACHED_TFIDF

    # Return cached data if already loaded
    if _CACHED_DF is not None and _CACHED_QUESTION_MATRIX is not None:
        return _CACHED_DF, _CACHED_QUESTION_MATRIX, _CACHED_TFIDF

    if not os.path.exists(DATA_PATH):
        print(f"[!] Warning: Knowledge base CSV not found at '{DATA_PATH}'.")
        return None, None, None

    try:
        df = pd.read_csv(DATA_PATH)
        # Ensure string types and clean whitespace
        df["question"] = df["question"].astype(str).str.strip()
        df["category"] = df["category"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()

        # Load the model to retrieve its trained TfidfVectorizer
        model = query_classifier.load_model()
        if model is not None and "tfidf" in model.named_steps:
            tfidf = model.named_steps["tfidf"]
            # Precompute TF-IDF vector matrix for fast cosine similarity matching
            q_matrix = tfidf.transform(df["question"])
            _CACHED_DF = df
            _CACHED_QUESTION_MATRIX = q_matrix
            _CACHED_TFIDF = tfidf
            print(f"[*] UniSage Query Service: Loaded {len(df)} knowledge base rows into memory.")
        else:
            _CACHED_DF = df
            _CACHED_QUESTION_MATRIX = None
            _CACHED_TFIDF = None

        return _CACHED_DF, _CACHED_QUESTION_MATRIX, _CACHED_TFIDF
    except Exception as exc:
        print(f"[!] Error loading knowledge base: {exc}")
        return None, None, None


def answer_question(question_text):
    """
    Core function that processes a user question:
    1. Validates the question text.
    2. Uses Scikit-learn Pipeline to predict the question's category.
    3. Detects unrelated or zero-term questions and returns a fallback message.
    4. Finds the closest matching sample answer from the CSV dataset.
    5. Returns a structured dictionary with category, answer, and official disclaimer.
    """
    cleaned_q = str(question_text).strip() if question_text else ""
    if not cleaned_q:
        return {
            "success": False,
            "error": "Question cannot be empty. Please enter your query.",
            "category": "Unknown",
            "answer": ""
        }

    # Step 1: Ensure ML model is loaded
    model = query_classifier.load_model()
    if model is None:
        return {
            "success": False,
            "error": "The Machine Learning model is not trained yet. Run 'python train_model.py' first.",
            "category": "Model Not Ready",
            "answer": "System setup in progress. Please try again shortly."
        }

    # Step 2: Check for term overlap (does the question have any recognized vocabulary?)
    tfidf = model.named_steps.get("tfidf")
    query_vec = tfidf.transform([cleaned_q]) if tfidf else None

    # If the user typed random gibberish or symbols where no vocabulary terms match:
    if query_vec is not None and query_vec.nnz == 0:
        return {
            "success": True,
            "category": "Uncertain",
            "is_fallback": True,
            "answer": UNRELIABLE_FALLBACK_MESSAGE,
            "confidence": 0.0,
            "disclaimer": OFFICIAL_DISCLAIMER
        }

    # Step 3: Predict category and confidence using Scikit-learn Logistic Regression
    pred_details = query_classifier.predict_category_details(cleaned_q)
    predicted_category = pred_details.get("category", "General")
    confidence = pred_details.get("confidence", 0.0)

    # Step 4: Retrieve best matching sample answer from CSV dataset
    df, q_matrix, _ = get_knowledge_base()
    selected_answer = None
    similarity_score = 0.0
    matched_sample_question = ""

    if df is not None and q_matrix is not None and query_vec is not None:
        # Filter knowledge base rows that match the predicted category
        cat_mask = (df["category"] == predicted_category).values
        candidate_indices = [i for i, is_cat in enumerate(cat_mask) if is_cat]

        if candidate_indices:
            candidate_matrix = q_matrix[candidate_indices]
            # Calculate cosine similarities between query and candidates in the category
            similarities = cosine_similarity(query_vec, candidate_matrix)[0]
            best_local_idx = similarities.argmax()
            similarity_score = float(similarities[best_local_idx])

            # If there is a reasonable match in the dataset, use its sample answer
            if similarity_score >= 0.18:
                best_global_idx = candidate_indices[best_local_idx]
                selected_answer = df.iloc[best_global_idx]["answer"]
                matched_sample_question = df.iloc[best_global_idx]["question"]

    # Step 5: If no specific question matched well, use the curated category guide answer
    if not selected_answer:
        # Check if prediction itself is very low confidence and similarity is near zero
        if confidence < 0.25 and similarity_score < 0.10:
            return {
                "success": True,
                "category": "Uncertain",
                "is_fallback": True,
                "answer": UNRELIABLE_FALLBACK_MESSAGE,
                "confidence": confidence,
                "disclaimer": OFFICIAL_DISCLAIMER
            }

        # Otherwise use the category default guidance
        selected_answer = CATEGORY_DEFAULT_ANSWERS.get(
            predicted_category,
            CATEGORY_DEFAULT_ANSWERS["General"]
        )

    return {
        "success": True,
        "category": predicted_category,
        "is_fallback": False,
        "answer": selected_answer,
        "confidence": confidence,
        "similarity": round(similarity_score, 4),
        "matched_sample": matched_sample_question,
        "disclaimer": OFFICIAL_DISCLAIMER
    }

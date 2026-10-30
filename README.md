# UniSage - University Query Management System (UQMS)
## Backend & Machine Learning Implementation (Flask, SQLite & Scikit-learn)

UniSage is an automated university assistant web application built with a Python Flask backend, SQLite database (`uqms.db`), responsive academic frontend, and an in-house **Machine Learning query classification engine** powered by **Pandas** and **Scikit-learn**.

---

### ⚠️ Important Notice on Machine Learning & Policies

> [!IMPORTANT]
> - **Category Classification, Not Policy Knowledge:** The ML model predicts *topics/categories* (e.g., Scholarship, Fees, Examination, Admission, Department, General) based on statistical language patterns. It does **not** independently understand, create, or enforce university policies.
> - **Sample Data Only:** All questions and answers provided in the dataset are sample illustrations and cannot guarantee accurate predictions for every university question. Official university announcements, notices, and portal records must always be consulted.
> - **Self-Contained & Private:** No external AI APIs (like OpenAI ChatGPT or Google Gemini) are used. The model and knowledge base run 100% locally on your machine.

---

### 📁 Project Structure

```text
UniSage/
├── app.py                       # Main Flask app, authentication, & /api/chat endpoint
├── train_model.py               # ML training script: loads CSV, trains TF-IDF + LogisticRegression
├── query_classifier.py          # ML helper module: cached model loader & category predictor
├── query_service.py             # Knowledge retrieval: matches predicted category to sample answer
├── test_chatbot_api.py          # Automated test suite for /api/chat, validation & disclaimers
├── test_ml_model.py             # Automated test suite for ML dataset, model & helpers
├── test_backend.py              # Automated test suite for database, auth, & session security
├── requirements.txt             # Python dependencies (Flask, Pandas, Scikit-learn, Joblib)
├── uqms.db                      # SQLite database storing registered users securely
├── data/                        # Training datasets (CSV format)
│   ├── all_data.csv             # Unified dataset across all 6 categories (2,548+ rows)
│   ├── scholarship.csv          # Scholarship & financial aid queries
│   ├── fees.csv                 # Tuition, exam fees, caution deposits, & payment queries
│   ├── examination.csv          # Semester exam, timetables, & re-evaluation queries
│   ├── admission.csv            # Admissions, eligibility criteria, & counseling queries
│   ├── department.csv           # Faculty directories, HODs, & building queries
│   └── general.csv              # Campus holidays, libraries, Wi-Fi, & helpdesk queries
├── models/
│   └── query_classifier.joblib  # Trained, serialized Scikit-learn Pipeline (~176 KB)
├── templates/
│   ├── login.html               # User sign-in page with flash alerts
│   ├── signup.html              # User registration page
│   └── chatbot.html             # Protected University Assistant chat dashboard
└── static/
    ├── css/
    │   └── style.css            # Responsive academic blue-and-white stylesheet
    └── js/
        ├── auth.js              # Client-side form validation before POST submission
        └── chatbot.js           # Fetch-based chat UI with loading indicator & error handling
```

---

### 🔄 How the System Works Together (Architecture Workflow)

```mermaid
flowchart TD
    User["Student in Browser"] -->|1. Type Question & Click Send| JS["chatbot.js"]
    JS -->|2. Show Loading Indicator & Disable Input| JS
    JS -->|3. fetch POST /api/chat (JSON)| Flask["Flask Server (app.py)"]
    Flask -->|4. Check Session (Authentication)| Auth{"Is Student Logged In?"}
    Auth -->|No| Reject["Return 401 Unauthorized -> Redirect to /login"]
    Auth -->|Yes| Valid{"Validate Question (Non-empty, <= 500 chars)"}
    Valid -->|No| Err400["Return 400 Bad Request with Validation Error"]
    Valid -->|Yes| ML["Scikit-learn Pipeline (TfidfVectorizer + LogisticRegression)"]
    ML -->|5. Predict Category & Confidence| QS["query_service.py"]
    QS -->|6. Term Overlap Check| ZeroTerms{"Known Terms Found?"}
    ZeroTerms -->|No / Gibberish| Fallback["Return Fallback Message: Rephrase or Contact Dept"]
    ZeroTerms -->|Yes| KB["CSV Knowledge Base (data/all_data.csv)"]
    KB -->|7. Cosine Similarity Match in Category| Answer["Retrieve Sample Answer + Official Disclaimer"]
    Answer --> FlaskResp["Return JSON {success, category, answer, disclaimer}"]
    Fallback --> FlaskResp
    FlaskResp -->|8. HTTP 200 JSON| JS
    JS -->|9. Remove Loading Indicator & Render Bubble| UI["Chatbot Conversation Window"]
```

1. **Browser (`chatbot.js`):**
   - Captures the query, validates length, renders the user's chat bubble, displays an animated typing indicator, and sends an asynchronous `fetch` POST request with `{"question": "..."}`.
2. **Flask Backend (`app.py`):**
   - Strictly enforces session authentication (unauthenticated calls return HTTP 401).
   - Validates JSON format, non-empty text, and maximum 500-character limits (invalid requests return HTTP 400).
3. **Machine Learning Pipeline (`query_classifier.py`):**
   - Extracted features using `TfidfVectorizer` (unigrams and bigrams).
   - Classifies question into one of the 6 categories using `LogisticRegression` (with balanced class weights).
   - Pre-loaded in memory on server launch (no disk reloading or retraining per request).
4. **CSV Knowledge Base & Retrieval (`query_service.py`):**
   - Matches the user's question against candidate sample questions in that category from `data/all_data.csv` using TF-IDF cosine similarity.
   - If terms are unrecognized or unrelated (e.g., recipe questions, random strings), returns a friendly fallback guidance message asking the student to rephrase or contact the department.
   - Always attaches the official confirmation disclaimer.
5. **DOM Rendering:**
   - The browser removes the typing indicator, formats the category tag, displays the answer, and appends the official notice box without refreshing the page.

---

### 🏷️ Target Categories & Prompt Benchmark Examples

| Category | Example Question | Predicted Category | Sample Answer Summary |
| :--- | :--- | :--- | :--- |
| **Scholarship** | *"What is the scholarship application process?"* | `Scholarship` | Consult student scholarship portal for eligibility and application dates. |
| **Fees** | *"What is the examination fee?"* | `Fees` | Check latest university fee circular or portal for applicable exam fee. |
| **Examination** | *"When will the semester exam begin?"* | `Examination` | Check latest examination timetable published on portal. |
| **Admission** | *"How can I apply for admission?"* | `Admission` | Register on portal, fill form, submit entrance score, attend counseling. |
| **Department** | *"Where is the computer science department?"* | `Department` | Check campus directory for building and department room location. |
| **General** | *"What are the university library timings and rules?"* | `General` | Check library notice board or handbook for hours and borrowing rules. |

---

### 🛠️ Exact Commands to Run and Test the Project

#### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

#### 2. Train the Machine Learning Model (if not already trained)
```powershell
python train_model.py
```
*(Evaluates on test set and saves `models/query_classifier.joblib`)*.

#### 3. Run Automated Verification Test Suites
Run all three test suites to verify backend security, ML classifier accuracy, and chatbot API integration:
```powershell
# Suite 1: SQLite database, Werkzeug password hashing, & session authentication
python test_backend.py

# Suite 2: ML pipeline structure, categories, predictions, & missing-model handling
python test_ml_model.py

# Suite 3: Chatbot POST /api/chat endpoint, authentication, validation, categories, & disclaimers
python test_chatbot_api.py
```

#### 4. Run the Flask Web Application
```powershell
python app.py
```

#### 5. Access the Web Application
Open your browser and visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**
1. Register a student account on the **Sign Up** page.
2. Sign in on the **Login** page.
3. Access the **University Assistant Chatbot** at `/chatbot`.
4. Ask university queries or click any of the suggested inquiry chips!

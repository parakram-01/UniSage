"""
==============================================================================
UniSage - University Query Management System (UQMS)
Flask Backend Application (app.py)
==============================================================================
This module implements the core backend for UniSage:
- SQLite database initialization and management (uqms.db)
- User registration with password hashing (Werkzeug)
- Secure user authentication and session management
- Route protection for the chatbot interface
- Machine Learning query classification and answer retrieval API
==============================================================================
"""

import os
import re
import sqlite3
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import query_classifier
import query_service

# ----------------------------------------------------------------------------
# Flask App Configuration
# ----------------------------------------------------------------------------
app = Flask(__name__)

# Preload ML query classifier into memory once at application launch
query_classifier_model = query_classifier.load_model()

# Preload university CSV knowledge base into memory for fast answer retrieval
query_service.get_knowledge_base()

# Database file path in project directory
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uqms.db')

# Secret Key — set SECRET_KEY env var in production
app.secret_key = os.environ.get('SECRET_KEY', 'unisage-dev-secret-key-change-in-production')

# Secure Session Cookie Settings
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

is_production = (
    os.environ.get('FLASK_ENV') == 'production' or
    os.environ.get('USE_SECURE_COOKIE', '').lower() == 'true'
)
app.config['SESSION_COOKIE_SECURE'] = is_production


# ----------------------------------------------------------------------------
# Database Helper Functions
# ----------------------------------------------------------------------------
def get_db_connection():
    """Returns a SQLite connection with row-by-name access."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates the users table if it doesn't exist.
    Columns: id, full_name, email, password_hash, created_at
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name     TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()


init_db()


# ----------------------------------------------------------------------------
# Authentication Decorator
# ----------------------------------------------------------------------------
def login_required(view_func):
    """Redirects unauthenticated users to the login page."""
    @wraps(view_func)
    def decorated_view(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please sign in to access the University Assistant.", "error")
            return redirect(url_for('login'))
        return view_func(*args, **kwargs)
    return decorated_view


# ----------------------------------------------------------------------------
# Input Validation Helper
# ----------------------------------------------------------------------------
def is_valid_email(email):
    """Validates email format using a standard regex."""
    return re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', email) is not None


# ----------------------------------------------------------------------------
# Application Routes
# ----------------------------------------------------------------------------

@app.route('/')
def home():
    """Redirect logged-in users to chatbot, others to login."""
    if 'user_id' in session:
        return redirect(url_for('chatbot'))
    return redirect(url_for('login'))


# ── Signup ──────────────────────────────────────────────────────────────────
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """
    Registration flow:
      GET  → show signup form
      POST → validate → save user → redirect to login
    """
    if 'user_id' in session:
        return redirect(url_for('chatbot'))

    if request.method == 'POST':
        full_name        = request.form.get('fullname', '').strip()
        email            = request.form.get('email', '').strip().lower()
        password         = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # --- Validation ---
        if not full_name or not email or not password or not confirm_password:
            flash("All fields are required. Please fill in the entire form.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        if not is_valid_email(email):
            flash("Please enter a valid email address.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        if len(password) < 6:
            flash("Password must be at least 6 characters long.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        if password != confirm_password:
            flash("Passwords do not match. Please re-enter.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        # --- Check existing account ---
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        existing = cursor.fetchone()
        conn.close()

        if existing:
            flash("An account with this email already exists. Please sign in.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        # --- Create account ---
        hashed_password = generate_password_hash(password)
        try:
            conn   = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO users (full_name, email, password_hash) VALUES (?, ?, ?)',
                (full_name, email, hashed_password)
            )
            conn.commit()
            conn.close()
        except sqlite3.Error:
            conn.close()
            flash("A database error occurred. Please try again later.", "error")
            return render_template('signup.html', fullname=full_name, email=email)

        flash("Account created successfully! Please sign in.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')


# ── Login ────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    """
    Login flow:
      GET  → show login form
      POST → validate credentials → establish session → redirect to chatbot
    """
    if 'user_id' in session:
        return redirect(url_for('chatbot'))

    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            flash("Please provide both email and password.", "error")
            return render_template('login.html', email=email)

        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()

        # Validate password hash
        if user is None or not check_password_hash(user['password_hash'], password):
            flash("Invalid email or password. Please check your credentials.", "error")
            return render_template('login.html', email=email)

        # Valid credentials — establish session
        session.clear()
        session['user_id']    = user['id']
        session['user_name']  = user['full_name']
        session['user_email'] = user['email']

        flash(f"Welcome back, {user['full_name']}!", "success")
        return redirect(url_for('chatbot'))

    return render_template('login.html')


# ── Logout ───────────────────────────────────────────────────────────────────
@app.route('/logout')
def logout():
    """Clears session and redirects to login."""
    session.clear()
    flash("You have been signed out successfully.", "success")
    return redirect(url_for('login'))


# ── Chatbot ──────────────────────────────────────────────────────────────────
@app.route('/chatbot')
@login_required
def chatbot():
    """Protected chatbot page — only accessible to logged-in users."""
    return render_template('chatbot.html', user_name=session.get('user_name', 'Student'))


# ----------------------------------------------------------------------------
# Machine Learning Query Classifier Helper & Diagnostic Routes
# ----------------------------------------------------------------------------
def predict_question_category(question):
    """Returns the predicted category for a given question string."""
    return query_classifier.predict_category(question)


@app.route('/api/predict-category', methods=['GET', 'POST'])
def api_predict_category():
    """
    Test the ML query classifier independently.
    GET:  /api/predict-category?question=...
    POST: JSON {"question": "..."}
    """
    if request.method == 'POST':
        data     = request.get_json(silent=True) or request.form
        question = data.get('question', '')
    else:
        question = request.args.get('question', '')

    return jsonify(query_classifier.predict_category_details(question))


@app.route('/api/model-status')
def api_model_status():
    """Returns classifier model status."""
    return jsonify(query_classifier.check_model_status())


@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Authenticated chatbot query endpoint.
    Requires active session, validates JSON payload, runs ML pipeline.
    """
    if 'user_id' not in session:
        return jsonify({
            'success': False,
            'error':   'Authentication required. Please sign in to ask questions.',
            'redirect': url_for('login')
        }), 401

    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Invalid JSON payload.'}), 400

    raw_question = data.get('question', '')
    if not isinstance(raw_question, str):
        return jsonify({'success': False, 'error': 'Question must be a text string.'}), 400

    question = raw_question.strip()

    if not question:
        return jsonify({'success': False, 'error': 'Question cannot be empty.'}), 400

    if len(question) > 500:
        return jsonify({
            'success': False,
            'error':   f'Question too long (max 500 chars). Current: {len(question)}.'
        }), 400

    return jsonify(query_service.answer_question(question)), 200


# ----------------------------------------------------------------------------
# Main Application Entry Point
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    print(" * Starting UniSage Flask Application...")
    print(" * Database: uqms.db initialized.")

    model_info = query_classifier.check_model_status()
    if model_info['available']:
        print(f" * ML Model: Ready ({model_info['message']})")
    else:
        print(f" * ML Model Notice: {model_info['message']}")

    print(" * Visit: http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)

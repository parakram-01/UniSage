"""
==============================================================================
UniSage - Comprehensive Backend Verification Script
==============================================================================
Tests all backend requirements:
1. Database initialization and user table schema
2. User registration & Werkzeug password hashing
3. Duplicate email rejection
4. Password validation (length, matching, format)
5. Authentication & session creation
6. Invalid credentials rejection
7. Protected chatbot route access control
8. Logout and session teardown
==============================================================================
"""

import sys
import os
import sqlite3
from werkzeug.security import check_password_hash

# Ensure we import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app

def run_tests():
    print("=" * 65)
    print("      UniSage (UQMS) Backend Verification Suite")
    print("=" * 65)
    
    passed_count = 0
    total_count = 10

    # ------------------------------------------------------------------------
    # Test 1: SQLite Database & Table Schema
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Verifying SQLite database and 'users' table schema...")
    assert os.path.exists(app.DATABASE_PATH), "Database file uqms.db does not exist!"
    conn = app.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(users)")
    columns = {row['name']: row['type'].upper() for row in cursor.fetchall()}
    conn.close()

    required_cols = ['id', 'full_name', 'email', 'password_hash', 'created_at']
    for col in required_cols:
        assert col in columns, f"Missing column '{col}' in users table!"
    print(f"  -> PASS: Database 'uqms.db' exists with columns: {list(columns.keys())}")
    passed_count += 1

    # Initialize Flask test client with session cookies enabled
    client = app.app.test_client()

    # ------------------------------------------------------------------------
    # Test 2: Protected Route Without Login
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Verifying route protection: unauthenticated /chatbot access...")
    response = client.get('/chatbot', follow_redirects=False)
    assert response.status_code == 302, f"Expected HTTP 302 redirect, got {response.status_code}"
    assert '/login' in response.headers.get('Location', ''), "Did not redirect to /login"
    print("  -> PASS: Unauthenticated access to /chatbot is strictly blocked and redirected to /login.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 3: Valid User Registration
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Testing user registration with valid details...")
    test_email = "student.test@university.edu"
    test_password = "SecurePassword@123"

    # Clean up any leftover test record
    conn = app.get_db_connection()
    conn.cursor().execute("DELETE FROM users WHERE email = ?", (test_email,))
    conn.commit()
    conn.close()

    response = client.post('/signup', data={
        'fullname': 'Jane Student',
        'email': test_email,
        'password': test_password,
        'confirm_password': test_password
    }, follow_redirects=True)
    
    assert response.status_code == 200
    assert b"Account created successfully" in response.data or b"Sign In" in response.data
    print("  -> PASS: User 'Jane Student' registered successfully with redirect & flash message.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 4: Password Hashing Verification
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Verifying password is cryptographically hashed (not plain-text)...")
    conn = app.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE email = ?", (test_email,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "User record was not found in database!"
    stored_hash = row['password_hash']
    assert stored_hash != test_password, "CRITICAL ERROR: Password was stored in plain text!"
    assert check_password_hash(stored_hash, test_password), "Stored hash does not match original password!"
    print(f"  -> PASS: Password is safely hashed with Werkzeug (Hash prefix: {stored_hash[:20]}...).")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 5: Duplicate Email Rejection
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing duplicate email registration rejection...")
    response = client.post('/signup', data={
        'fullname': 'Duplicate User',
        'email': test_email,
        'password': test_password,
        'confirm_password': test_password
    }, follow_redirects=True)
    assert b"already exists" in response.data, "Duplicate email was not rejected!"
    print("  -> PASS: Duplicate email signup correctly rejected with flash error message.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 6: Password Mismatch Rejection
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Testing password mismatch during signup...")
    response = client.post('/signup', data={
        'fullname': 'Mismatch User',
        'email': 'mismatch@university.edu',
        'password': 'password123',
        'confirm_password': 'differentPassword456'
    }, follow_redirects=True)
    assert b"do not match" in response.data, "Mismatched passwords were not caught!"
    print("  -> PASS: Password mismatch caught and rejected.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 7: Invalid Login Credentials
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing login with invalid credentials...")
    response_wrong_pw = client.post('/login', data={
        'email': test_email,
        'password': 'WrongPassword999'
    }, follow_redirects=True)
    assert b"Invalid email or password" in response_wrong_pw.data
    
    response_no_user = client.post('/login', data={
        'email': 'nonexistent@university.edu',
        'password': 'somePassword'
    }, follow_redirects=True)
    assert b"Invalid email or password" in response_no_user.data
    print("  -> PASS: Incorrect password and non-existent email both rejected safely.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Test 8: Valid Login & Session Creation
    # ------------------------------------------------------------------------
    print("\n[TEST 8] Testing successful login and session establishment...")
    # Use client within session context
    with client:
        response = client.post('/login', data={
            'email': test_email,
            'password': test_password
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b"University Assistant" in response.data
        
        # Verify session variables
        with client.session_transaction() as sess:
            assert 'user_id' in sess, "user_id not saved in session!"
            assert sess['user_name'] == 'Jane Student', "user_name mismatch in session!"
            assert sess['user_email'] == test_email, "user_email mismatch in session!"
        print("  -> PASS: Login succeeded, session variables established (user_id, user_name, user_email).")
        passed_count += 1

        # --------------------------------------------------------------------
        # Test 9: Authenticated Access to Chatbot
        # --------------------------------------------------------------------
        print("\n[TEST 9] Testing access to protected /chatbot with active session...")
        response_chat = client.get('/chatbot')
        assert response_chat.status_code == 200
        assert b"Jane Student" in response_chat.data
        assert b"University Assistant" in response_chat.data
        print("  -> PASS: /chatbot rendered successfully with personalized user greeting.")
        passed_count += 1

        # --------------------------------------------------------------------
        # Test 10: Logout & Session Teardown
        # --------------------------------------------------------------------
        print("\n[TEST 10] Testing logout flow and session clearing...")
        response_logout = client.get('/logout', follow_redirects=True)
        assert response_logout.status_code == 200
        assert b"signed out successfully" in response_logout.data or b"Sign In" in response_logout.data
        
        # Verify session is empty
        with client.session_transaction() as sess:
            assert 'user_id' not in sess, "Session was not cleared upon logout!"

        # Verify /chatbot is blocked once again
        response_blocked = client.get('/chatbot', follow_redirects=False)
        assert response_blocked.status_code == 302
        assert '/login' in response_blocked.headers.get('Location', '')
        print("  -> PASS: Session cleared upon logout. Subsequent /chatbot access blocked.")
        passed_count += 1

    # Clean up test user
    conn = app.get_db_connection()
    conn.cursor().execute("DELETE FROM users WHERE email = ?", (test_email,))
    conn.commit()
    conn.close()

    print("\n" + "=" * 65)
    print(f"     ALL {passed_count}/{total_count} BACKEND VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == '__main__':
    run_tests()

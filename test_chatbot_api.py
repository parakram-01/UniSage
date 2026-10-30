"""
==============================================================================
UniSage - Chatbot API & ML Integration Verification Suite (test_chatbot_api.py)
==============================================================================
Tests the end-to-end chatbot integration:
1. Route protection: Unauthenticated requests rejected with 401.
2. Input validation: Empty strings and excessively long questions rejected with 400.
3. Invalid payload formats: Non-JSON / malformed bodies rejected with 400.
4. Classification & answer retrieval across all 6 categories:
   - Scholarship
   - Fees
   - Examination
   - Admission
   - Department
   - General
5. Fallback detection for unrelated queries / gibberish.
6. Inclusion of official verification disclaimers on all responses.
==============================================================================
"""

import sys
import os
import json

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import app


def run_chatbot_api_tests():
    print("=" * 65)
    print("      UniSage Chatbot & ML API Verification Suite")
    print("=" * 65)

    passed_count = 0
    total_tests = 10

    client = app.app.test_client()

    # ------------------------------------------------------------------------
    # TEST 1: Unauthenticated Access to /api/chat
    # ------------------------------------------------------------------------
    print("\n[TEST 1] Testing unauthenticated POST /api/chat access...")
    resp = client.post("/api/chat", json={"question": "What is the exam fee?"})
    assert resp.status_code == 401, f"Expected 401 Unauthorized, got {resp.status_code}"
    data = resp.get_json()
    assert data["success"] is False
    assert "Authentication required" in data["error"]
    print("  -> PASS: Unauthenticated access strictly rejected with HTTP 401.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # Create Authenticated Session for Subsequent Tests
    # ------------------------------------------------------------------------
    # Register/login user via test client
    client.post("/signup", data={
        "fullname": "Alex Student",
        "email": "alex.student@university.edu",
        "password": "Password123!",
        "confirm_password": "Password123!"
    })
    client.post("/login", data={
        "email": "alex.student@university.edu",
        "password": "Password123!"
    })

    # ------------------------------------------------------------------------
    # TEST 2: Empty Question Validation
    # ------------------------------------------------------------------------
    print("\n[TEST 2] Testing empty and whitespace-only query validation...")
    resp_empty = client.post("/api/chat", json={"question": ""})
    assert resp_empty.status_code == 400, f"Expected 400 for empty input, got {resp_empty.status_code}"
    data_empty = resp_empty.get_json()
    assert data_empty["success"] is False
    assert "cannot be empty" in data_empty["error"]

    resp_spaces = client.post("/api/chat", json={"question": "     "})
    assert resp_spaces.status_code == 400, f"Expected 400 for whitespace, got {resp_spaces.status_code}"
    print("  -> PASS: Empty and whitespace questions correctly rejected with HTTP 400.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 3: Invalid Request Formats
    # ------------------------------------------------------------------------
    print("\n[TEST 3] Testing malformed payloads and non-string inputs...")
    # Missing 'question' key
    resp_missing = client.post("/api/chat", json={"other_key": "hello"})
    assert resp_missing.status_code == 400

    # Non-string question
    resp_non_str = client.post("/api/chat", json={"question": 12345})
    assert resp_non_str.status_code == 400

    # Non-JSON content
    resp_non_json = client.post("/api/chat", data="not json", content_type="text/plain")
    assert resp_non_json.status_code == 400
    print("  -> PASS: Malformed requests, non-string, and non-JSON payloads safely rejected.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 4: Maximum Length Validation (> 500 characters)
    # ------------------------------------------------------------------------
    print("\n[TEST 4] Testing maximum length validation (> 500 characters)...")
    long_question = "What is the fee? " * 50  # ~850 characters
    resp_long = client.post("/api/chat", json={"question": long_question})
    assert resp_long.status_code == 400, f"Expected 400 for long input, got {resp_long.status_code}"
    data_long = resp_long.get_json()
    assert "too long" in data_long["error"]
    print(f"  -> PASS: Question with {len(long_question)} characters rejected with HTTP 400.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 5: Category Classification - Fees
    # ------------------------------------------------------------------------
    print("\n[TEST 5] Testing category 'Fees' query...")
    resp_fees = client.post("/api/chat", json={"question": "What is the examination fee?"})
    assert resp_fees.status_code == 200
    data_fees = resp_fees.get_json()
    assert data_fees["success"] is True
    assert data_fees["category"] == "Fees", f"Expected Fees, got {data_fees['category']}"
    assert len(data_fees["answer"]) > 10
    assert "disclaimer" in data_fees
    print(f"  -> PASS: Fees query classified correctly. Answer: \"{data_fees['answer'][:60]}...\"")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 6: Category Classification - Scholarship
    # ------------------------------------------------------------------------
    print("\n[TEST 6] Testing category 'Scholarship' query...")
    resp_sch = client.post("/api/chat", json={"question": "What is the scholarship application process?"})
    assert resp_sch.status_code == 200
    data_sch = resp_sch.get_json()
    assert data_sch["success"] is True
    assert data_sch["category"] == "Scholarship", f"Expected Scholarship, got {data_sch['category']}"
    assert len(data_sch["answer"]) > 10
    print(f"  -> PASS: Scholarship query classified correctly. Answer: \"{data_sch['answer'][:60]}...\"")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 7: Category Classification - Examination
    # ------------------------------------------------------------------------
    print("\n[TEST 7] Testing category 'Examination' query...")
    resp_exam = client.post("/api/chat", json={"question": "When will the semester exam begin?"})
    assert resp_exam.status_code == 200
    data_exam = resp_exam.get_json()
    assert data_exam["success"] is True
    assert data_exam["category"] == "Examination", f"Expected Examination, got {data_exam['category']}"
    assert len(data_exam["answer"]) > 10
    print(f"  -> PASS: Examination query classified correctly. Answer: \"{data_exam['answer'][:60]}...\"")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 8: Category Classification - Admission & Department & General
    # ------------------------------------------------------------------------
    print("\n[TEST 8] Testing remaining categories ('Admission', 'Department', 'General')...")
    # Admission
    resp_adm = client.post("/api/chat", json={"question": "How can I apply for admission?"})
    assert resp_adm.status_code == 200
    assert resp_adm.get_json()["category"] == "Admission"
    print("  -> PASS: Admission query classified correctly.")

    # Department
    resp_dept = client.post("/api/chat", json={"question": "Where is the computer science department?"})
    assert resp_dept.status_code == 200
    assert resp_dept.get_json()["category"] == "Department"
    print("  -> PASS: Department query classified correctly.")

    # General
    resp_gen = client.post("/api/chat", json={"question": "What are the university library timings and rules?"})
    assert resp_gen.status_code == 200
    assert resp_gen.get_json()["category"] == "General"
    print("  -> PASS: General query classified correctly.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 9: Unrelated / Unreliable Query Fallback Handling
    # ------------------------------------------------------------------------
    print("\n[TEST 9] Testing unrelated / gibberish query fallback handling...")
    resp_unrel = client.post("/api/chat", json={"question": "How to bake a chocolate cake at home?"})
    assert resp_unrel.status_code == 200
    data_unrel = resp_unrel.get_json()
    assert data_unrel["is_fallback"] is True
    assert "not confident" in data_unrel["answer"]

    resp_gib = client.post("/api/chat", json={"question": "asdfghjkl zxcvbnm 123456"})
    assert resp_gib.status_code == 200
    data_gib = resp_gib.get_json()
    assert data_gib["is_fallback"] is True
    assert "not confident" in data_gib["answer"]
    print("  -> PASS: Unrelated queries caught and assigned helpful fallback guidance.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # TEST 10: Official Verification Disclaimer on All Responses
    # ------------------------------------------------------------------------
    print("\n[TEST 10] Verifying official verification disclaimers are present on all responses...")
    for resp_obj in [data_fees, data_sch, data_exam, data_unrel, data_gib]:
        assert "disclaimer" in resp_obj
        assert "sample data" in resp_obj["disclaimer"].lower()
        assert "official university notices" in resp_obj["disclaimer"].lower()
    print("  -> PASS: 100% of responses include mandatory official verification disclaimers.")
    passed_count += 1

    # ------------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print(f"     ALL {passed_count}/{total_tests} CHATBOT API VERIFICATION TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    run_chatbot_api_tests()

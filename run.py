"""
run.py — UniSage launcher
Starts the Flask development server.
Run: python run.py
"""
import app

if __name__ == '__main__':
    app.app.run(debug=True, host='127.0.0.1', port=5000)

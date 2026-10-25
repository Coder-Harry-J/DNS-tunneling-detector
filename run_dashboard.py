"""
DNS Tunneling Detector — Dashboard Launcher
Run this from the project root to start the web dashboard.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'dashboard'))

from app import app, load_models

if __name__ == '__main__':
    load_models()
    print("\n" + "="*55)
    print("  DNS Tunneling NIDS Dashboard")
    print("  Open in browser: http://127.0.0.1:5000")
    print("="*55 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')

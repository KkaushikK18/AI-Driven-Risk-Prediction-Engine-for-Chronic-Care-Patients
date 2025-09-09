#!/usr/bin/env python3
"""
Startup Script for ML-Enhanced Chronic Care Risk Engine
======================================================

This script starts the Flask API server and opens the dashboard.

Usage:
    python start_ml_dashboard.py
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import os

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_cors
        import pandas
        import numpy
        import sklearn
        import joblib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("📦 Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "api_requirements.txt"])
            print("✅ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")
            return False

def check_models():
    """Check if ML models exist"""
    models_dir = Path("../models")
    required_models = ["enhanced_best_model.pkl", "enhanced_scaler.pkl"]
    
    missing_models = []
    for model in required_models:
        if not (models_dir / model).exists():
            missing_models.append(model)
    
    if missing_models:
        print(f"⚠️  Missing models: {missing_models}")
        print("📊 You can still run the dashboard with clinical fallback calculations")
        return False
    else:
        print("✅ All ML models found")
        return True

def start_server():
    """Start the Flask API server"""
    print("🚀 Starting ML Model API Server...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Dashboard will be available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/predict")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Start the Flask server
        subprocess.run([sys.executable, "ml_api_server.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    print("🏥 ML-Enhanced Chronic Care Risk Engine")
    print("="*50)
    
    # Check if we're in the right directory
    if not Path("ml_api_server.py").exists():
        print("❌ Please run this script from the HACKATHON_DEMO directory")
        return
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check models
    models_available = check_models()
    
    if not models_available:
        print("\n⚠️  ML models not found. The dashboard will use clinical fallback calculations.")
        print("   To train the ML models, run: python ../src/engines/enhanced_chronic_risk_engine.py")
    
    print("\n🎯 Starting the dashboard...")
    
    # Wait a moment then open browser
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()

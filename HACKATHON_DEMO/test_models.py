#!/usr/bin/env python3
"""
Quick Model Test Script
=======================

Test if the ML models can be loaded properly.

Usage:
    python test_models.py
"""

import joblib
from pathlib import Path
import sys

def test_model_loading():
    """Test loading the ML models"""
    print("🧪 Testing ML Model Loading")
    print("=" * 40)
    
    # Define model paths
    models_dir = Path("..") / "models"
    model_files = [
        "enhanced_best_model.pkl",
        "enhanced_scaler.pkl", 
        "feature_selector.pkl"
    ]
    
    print(f"📁 Models directory: {models_dir.absolute()}")
    print()
    
    # Check if models directory exists
    if not models_dir.exists():
        print("❌ Models directory not found!")
        print(f"   Expected: {models_dir.absolute()}")
        return False
    
    # Check each model file
    models_loaded = {}
    for model_file in model_files:
        model_path = models_dir / model_file
        print(f"🔍 Checking: {model_file}")
        
        if model_path.exists():
            try:
                model = joblib.load(model_path)
                models_loaded[model_file] = model
                print(f"   ✅ Loaded successfully: {type(model).__name__}")
            except Exception as e:
                print(f"   ❌ Failed to load: {e}")
        else:
            print(f"   ❌ File not found: {model_path}")
    
    print()
    print("📊 Summary:")
    print("=" * 20)
    
    if len(models_loaded) == len(model_files):
        print("🎉 All models loaded successfully!")
        print("   Your ML model integration should work perfectly.")
        return True
    else:
        print(f"⚠️  Only {len(models_loaded)}/{len(model_files)} models loaded")
        print("   The dashboard will use clinical fallback calculations.")
        return False

if __name__ == "__main__":
    test_model_loading()

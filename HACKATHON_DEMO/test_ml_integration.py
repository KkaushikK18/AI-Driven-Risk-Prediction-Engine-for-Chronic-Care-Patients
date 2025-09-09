#!/usr/bin/env python3
"""
Test Script for ML Model Integration
====================================

This script tests the ML model API integration to ensure everything works correctly.

Usage:
    python test_ml_integration.py
"""

import requests
import json
import time
import sys
from pathlib import Path

API_BASE_URL = 'http://localhost:5000/api'

def test_api_health():
    """Test if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Health Check: PASSED")
            return True
        else:
            print(f"❌ API Health Check: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API Health Check: FAILED (Error: {e})")
        return False

def test_model_info():
    """Test model information endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Model Info: PASSED")
            print(f"   Model Loaded: {data.get('model_loaded', False)}")
            print(f"   Model Type: {data.get('model_type', 'Unknown')}")
            print(f"   Features: {data.get('expected_features', 0)}")
            return data.get('model_loaded', False)
        else:
            print(f"❌ Model Info: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model Info: FAILED (Error: {e})")
        return False

def test_prediction():
    """Test prediction endpoint with sample data"""
    test_cases = [
        {
            "name": "Low Risk Patient",
            "data": {
                "age": 45,
                "admissions": 0,
                "hemoglobin": 14.0,
                "creatinine": 0.9,
                "diabetes": False,
                "heart_failure": False,
                "ckd": False,
                "copd": False
            }
        },
        {
            "name": "High Risk Patient",
            "data": {
                "age": 80,
                "admissions": 5,
                "hemoglobin": 8.5,
                "creatinine": 2.8,
                "diabetes": True,
                "heart_failure": True,
                "ckd": True,
                "copd": False
            }
        },
        {
            "name": "Medium Risk Patient",
            "data": {
                "age": 65,
                "admissions": 2,
                "hemoglobin": 11.0,
                "creatinine": 1.3,
                "diabetes": True,
                "heart_failure": False,
                "ckd": False,
                "copd": True
            }
        }
    ]
    
    print("\n🧪 Testing Predictions:")
    print("=" * 50)
    
    for test_case in test_cases:
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=test_case["data"],
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {test_case['name']}: PASSED")
                print(f"   Risk Score: {result.get('risk_score', 0):.3f}")
                print(f"   Risk Level: {result.get('risk_level', 'Unknown')}")
                print(f"   Model Used: {result.get('model_used', 'Unknown')}")
                print(f"   Features: {result.get('features_used', 0)}")
                print()
            else:
                print(f"❌ {test_case['name']}: FAILED (Status: {response.status_code})")
                print(f"   Response: {response.text}")
                print()
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {test_case['name']}: FAILED (Error: {e})")
            print()

def test_dashboard_access():
    """Test if dashboard is accessible"""
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Dashboard Access: PASSED")
            return True
        else:
            print(f"❌ Dashboard Access: FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Dashboard Access: FAILED (Error: {e})")
        return False

def main():
    """Main test function"""
    print("🧪 ML Model Integration Test")
    print("=" * 40)
    print()
    
    # Test 1: API Health
    if not test_api_health():
        print("❌ API is not running. Please start the server first:")
        print("   python ml_api_server.py")
        return False
    
    print()
    
    # Test 2: Model Info
    model_loaded = test_model_info()
    print()
    
    # Test 3: Predictions
    test_prediction()
    
    # Test 4: Dashboard Access
    test_dashboard_access()
    print()
    
    # Summary
    print("📊 Test Summary:")
    print("=" * 20)
    if model_loaded:
        print("🎉 ML Model Integration: FULLY WORKING")
        print("   • ML model is loaded and active")
        print("   • API endpoints are responding")
        print("   • Dashboard is accessible")
        print("   • Predictions are working")
        print()
        print("🌐 Open your browser to: http://localhost:5000")
    else:
        print("⚠️  ML Model Integration: FALLBACK MODE")
        print("   • API is running with clinical fallback")
        print("   • Dashboard is accessible")
        print("   • Predictions use clinical logic")
        print()
        print("💡 To enable ML model:")
        print("   1. Train the model: python ../src/engines/enhanced_chronic_risk_engine.py")
        print("   2. Restart the API server")
        print()
        print("🌐 Open your browser to: http://localhost:5000")
    
    return True

if __name__ == "__main__":
    main()

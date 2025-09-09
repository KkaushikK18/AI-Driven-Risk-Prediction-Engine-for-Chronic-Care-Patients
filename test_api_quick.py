#!/usr/bin/env python3
"""
Quick API Test
"""

import requests
import json

def test_api():
    try:
        # Test health endpoint
        response = requests.get('http://localhost:5000/api/health')
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test prediction endpoint
        test_data = {
            "age": 75,
            "admissions": 3,
            "hemoglobin": 9.5,
            "creatinine": 2.1,
            "diabetes": True,
            "heart_failure": True,
            "ckd": False,
            "copd": False
        }
        
        response = requests.post('http://localhost:5000/api/predict', 
                               json=test_data,
                               headers={'Content-Type': 'application/json'})
        
        print(f"\nPrediction test: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Risk Score: {result.get('risk_percentage', 'N/A')}%")
            print(f"Risk Level: {result.get('risk_level', 'N/A')}")
            print(f"Model: {result.get('model_used', 'N/A')}")
            print(f"Features: {result.get('features_used', 'N/A')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
#!/usr/bin/env python3
"""
ML Model API Server for Chronic Care Risk Engine
===============================================

Flask API server that serves predictions from the trained ML model
to the interactive dashboard.

Usage:
    python ml_api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys
import os

# Models will be loaded from the models directory

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

class MLModelAPI:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.feature_names = None
        self.load_models()
    
    def load_models(self):
        """Load the trained ML models"""
        try:
            # Define model paths directly (more robust)
            models_dir = Path(__file__).parent.parent / "models"
            model_path = models_dir / "enhanced_best_model.pkl"
            scaler_path = models_dir / "enhanced_scaler.pkl"
            feature_selector_path = models_dir / "feature_selector.pkl"
            
            print(f"🔍 Looking for models in: {models_dir}")
            print(f"   Model path: {model_path}")
            print(f"   Scaler path: {scaler_path}")
            print(f"   Feature selector path: {feature_selector_path}")
            
            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"✅ Loaded enhanced model: {type(self.model).__name__}")
            else:
                print("❌ Enhanced model not found, using fallback")
                return
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print("✅ Loaded enhanced scaler")
                
                # Get actual feature names from the scaler
                if hasattr(self.scaler, 'feature_names_in_'):
                    self.feature_names = list(self.scaler.feature_names_in_)
                    print(f"✅ Got feature names from scaler: {len(self.feature_names)} features")
                else:
                    print("⚠️  Scaler doesn't have feature names, using fallback")
                    self.feature_names = self._get_expected_features()
            else:
                self.feature_names = self._get_expected_features()
            
            if feature_selector_path.exists():
                self.feature_selector = joblib.load(feature_selector_path)
                print("✅ Loaded feature selector")
            
            print(f"✅ Total features expected: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def _get_expected_features(self):
        """Define the expected feature names for the enhanced model"""
        return [
            # Demographics
            'age', 'gender_M', 'ethnicity_WHITE', 'ethnicity_BLACK', 'ethnicity_HISPANIC', 'ethnicity_OTHER',
            
            # Lab values (recent)
            'hemoglobin_mean', 'hemoglobin_min', 'hemoglobin_max', 'hemoglobin_trend',
            'creatinine_mean', 'creatinine_min', 'creatinine_max', 'creatinine_trend',
            'sodium_mean', 'sodium_min', 'sodium_max', 'sodium_trend',
            'potassium_mean', 'potassium_min', 'potassium_max', 'potassium_trend',
            'glucose_mean', 'glucose_min', 'glucose_max', 'glucose_trend',
            'bun_mean', 'bun_min', 'bun_max', 'bun_trend',
            'wbc_mean', 'wbc_min', 'wbc_max', 'wbc_trend',
            'platelet_mean', 'platelet_min', 'platelet_max', 'platelet_trend',
            
            # Admission history
            'prior_admissions_6m', 'prior_admissions_12m', 'prior_admissions_total',
            'avg_los_6m', 'avg_los_12m', 'readmission_30d_flag',
            
            # Comorbidities
            'diabetes_flag', 'heart_failure_flag', 'ckd_flag', 'copd_flag',
            'hypertension_flag', 'coronary_artery_disease_flag', 'stroke_flag',
            'comorbidity_count', 'comorbidity_burden_score',
            
            # Medication complexity
            'medication_count', 'high_risk_med_count', 'polypharmacy_flag',
            'anticoagulant_flag', 'diuretic_flag', 'ace_inhibitor_flag',
            
            # ICU utilization
            'icu_admissions_6m', 'icu_admissions_12m', 'total_icu_days',
            'mechanical_ventilation_flag', 'vasopressor_flag',
            
            # Vital signs
            'systolic_bp_mean', 'diastolic_bp_mean', 'heart_rate_mean',
            'respiratory_rate_mean', 'temperature_mean', 'oxygen_saturation_mean',
            
            # Clinical scores
            'charlson_score', 'elixhauser_score', 'sofa_score',
            
            # Additional features
            'emergency_admission_flag', 'weekend_admission_flag',
            'discharge_disposition_home', 'discharge_disposition_snf',
            'insurance_medicare', 'insurance_medicaid', 'insurance_private',
            
            # Lab abnormalities
            'anemia_flag', 'hyperkalemia_flag', 'hyponatremia_flag',
            'hyperglycemia_flag', 'leukocytosis_flag', 'thrombocytopenia_flag',
            
            # Time-based features
            'days_since_last_admission', 'admission_month', 'admission_day_of_week',
            
            # Risk scores
            'baseline_risk_score'
        ]
    
    def prepare_features(self, patient_data):
        """Convert patient input to feature vector matching training data"""
        try:
            print(f"🔧 Preparing features: {len(self.feature_names)} expected")
            
            # Extract input values
            age = int(patient_data.get('age', 65))
            admissions = int(patient_data.get('admissions', 0))
            hemoglobin = float(patient_data.get('hemoglobin', 12.0))
            creatinine = float(patient_data.get('creatinine', 1.0))
            
            # Comorbidities
            diabetes = patient_data.get('diabetes', False)
            heart_failure = patient_data.get('heart_failure', False)
            ckd = patient_data.get('ckd', False)
            copd = patient_data.get('copd', False)
            
            # Initialize feature vector with zeros
            feature_vector = np.zeros(len(self.feature_names))
            
            # Create a mapping of feature names to indices
            feature_map = {name: i for i, name in enumerate(self.feature_names)}
            
            # Set features based on what's available in the model
            if 'age' in feature_map:
                feature_vector[feature_map['age']] = age
            
            # Set lab values
            lab_features = {
                'hemoglobin_mean': hemoglobin,
                'hemoglobin_min': hemoglobin * 0.9,
                'hemoglobin_max': hemoglobin * 1.1,
                'hemoglobin_trend': 0,
                'creatinine_mean': creatinine,
                'creatinine_min': creatinine * 0.9,
                'creatinine_max': creatinine * 1.1,
                'creatinine_trend': 0
            }
            
            for feature_name, value in lab_features.items():
                if feature_name in feature_map:
                    feature_vector[feature_map[feature_name]] = value
            
            # Set admission history
            admission_features = {
                'prior_admissions_6m': admissions,
                'prior_admissions_12m': admissions * 1.5,
                'prior_admissions_total': admissions * 2
            }
            
            for feature_name, value in admission_features.items():
                if feature_name in feature_map:
                    feature_vector[feature_map[feature_name]] = value
            
            # Set comorbidities
            comorbidity_features = {
                'diabetes_flag': 1 if diabetes else 0,
                'heart_failure_flag': 1 if heart_failure else 0,
                'ckd_flag': 1 if ckd else 0,
                'copd_flag': 1 if copd else 0
            }
            
            for feature_name, value in comorbidity_features.items():
                if feature_name in feature_map:
                    feature_vector[feature_map[feature_name]] = value
            
            # Calculate derived features
            comorbidity_count = sum([diabetes, heart_failure, ckd, copd])
            
            derived_features = {
                'comorbidity_count': comorbidity_count,
                'comorbidity_burden_score': comorbidity_count * 0.2,
                'medication_count': 5 + comorbidity_count * 2,
                'polypharmacy_flag': 1 if (5 + comorbidity_count * 2) > 5 else 0,
                'charlson_score': max(0, (age - 50) / 10) + comorbidity_count * 0.5,
                'elixhauser_score': comorbidity_count * 0.3,
                'anemia_flag': 1 if hemoglobin < 10 else 0,
                'hyperkalemia_flag': 1 if creatinine > 1.5 else 0,
                'baseline_risk_score': self._calculate_baseline_risk(age, admissions, hemoglobin, creatinine, comorbidity_count)
            }
            
            for feature_name, value in derived_features.items():
                if feature_name in feature_map:
                    feature_vector[feature_map[feature_name]] = value
            
            # Set default values for common features
            default_features = {
                'gender_M': 1,
                'ethnicity_WHITE': 1,
                'emergency_admission_flag': 0,
                'weekend_admission_flag': 0,
                'discharge_disposition_home': 1,
                'insurance_medicare': 1
            }
            
            for feature_name, value in default_features.items():
                if feature_name in feature_map:
                    feature_vector[feature_map[feature_name]] = value
            
            print(f"✅ Prepared feature vector: {feature_vector.shape}")
            return feature_vector.reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_baseline_risk(self, age, admissions, hemoglobin, creatinine, comorbidity_count):
        """Calculate baseline risk score"""
        risk_score = 0
        
        # Age risk
        if age >= 85:
            risk_score += 4
        elif age >= 75:
            risk_score += 3
        elif age >= 65:
            risk_score += 2
        elif age >= 50:
            risk_score += 1
        
        # Lab risk
        if hemoglobin < 7:
            risk_score += 4
        elif hemoglobin < 8:
            risk_score += 3
        elif hemoglobin < 10:
            risk_score += 2
        elif hemoglobin < 11:
            risk_score += 1
        
        if creatinine > 4:
            risk_score += 4
        elif creatinine > 3:
            risk_score += 3
        elif creatinine > 2:
            risk_score += 2
        elif creatinine > 1.5:
            risk_score += 1
        
        # Admission risk
        if admissions >= 5:
            risk_score += 3
        elif admissions >= 3:
            risk_score += 2
        elif admissions >= 1:
            risk_score += 1
        
        # Comorbidity risk
        risk_score += comorbidity_count
        
        return risk_score
    
    def predict_risk(self, patient_data):
        """Make prediction using the ML model"""
        try:
            if self.model is None:
                return {"error": "Model not loaded"}
            
            # Prepare features
            features = self.prepare_features(patient_data)
            if features is None:
                return {"error": "Failed to prepare features"}
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
                print(f"✅ Scaled features: {features_scaled.shape}")
            else:
                features_scaled = features
            
            # Skip feature selection - the enhanced model was trained on all 173 features
            features_selected = features_scaled
            print(f"✅ Using all scaled features for enhanced model: {features_selected.shape}")
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                risk_score = self.model.predict_proba(features_selected)[:, 1][0]
            else:
                risk_score = self.model.predict(features_selected)[0]
            
            # Classify risk level
            if risk_score >= 0.7:
                risk_level = "HIGH"
                recommendation = "🚨 Immediate clinical attention required"
            elif risk_score >= 0.4:
                risk_level = "MEDIUM"
                recommendation = "⚠️ Enhanced monitoring recommended"
            else:
                risk_level = "LOW"
                recommendation = "✅ Standard care protocol recommended"
            
            return {
                "risk_score": float(risk_score),
                "risk_percentage": round(risk_score * 100, 1),
                "risk_level": risk_level,
                "recommendation": recommendation,
                "model_used": type(self.model).__name__,
                "features_used": len(features_selected[0])
            }
            
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {"error": str(e)}

# Initialize the ML model API
ml_api = MLModelAPI()

@app.route('/')
def index():
    """Serve the main dashboard"""
    dashboard_path = Path(__file__).parent / 'MAIN_DASHBOARD.html'
    if dashboard_path.exists():
        return dashboard_path.read_text(encoding='utf-8')
    else:
        return "Dashboard file not found", 404

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for risk prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Make prediction
        result = ml_api.predict_risk(data)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        "model_loaded": ml_api.model is not None,
        "model_type": type(ml_api.model).__name__ if ml_api.model else None,
        "scaler_loaded": ml_api.scaler is not None,
        "feature_selector_loaded": ml_api.feature_selector is not None,
        "expected_features": len(ml_api.feature_names) if ml_api.feature_names else 0
    })

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_ready": ml_api.model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting ML Model API Server...")
    print("📊 Model status:", "✅ Loaded" if ml_api.model is not None else "❌ Not loaded")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Dashboard will be available at: http://localhost:5000")
    print("🔗 API endpoint: http://localhost:5000/api/predict")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

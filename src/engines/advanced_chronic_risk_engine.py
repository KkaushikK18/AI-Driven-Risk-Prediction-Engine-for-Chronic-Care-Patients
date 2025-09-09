#!/usr/bin/env python3
"""
Advanced Chronic Care Risk Prediction Engine
===========================================

Clean, working version for 90-day deterioration prediction.

Usage:
    python advanced_chronic_risk_engine.py --mode synthetic
    python advanced_chronic_risk_engine.py --mode mimic
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib

# Import configuration
import sys
sys.path.append('../../')
from src.utils.config import config

class AdvancedChronicRiskEngine:
    def __init__(self, use_real_data=False):
        self.use_real_data = use_real_data
        self.output_dir = config.paths.results
        self.models_dir = config.paths.models
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🏥 Advanced Chronic Risk Engine initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
        
    def load_mimic_data(self):
        """Load and preprocess real MIMIC-IV data"""
        print("📊 Loading MIMIC-IV data...")
        
        mimic_path = config.paths.raw_mimic
        
        # Load core tables
        print("   Loading patients...")
        patients = pd.read_csv(mimic_path / "patients.csv")
        
        print("   Loading admissions...")
        admissions = pd.read_csv(mimic_path / "admissions.csv")
        
        print("   Loading lab events (this may take a moment)...")
        # Load labevents in chunks for memory efficiency
        lab_chunks = []
        chunk_size = 100000
        for chunk in pd.read_csv(mimic_path / "labevents.csv", chunksize=chunk_size):
            lab_chunks.append(chunk)
        labevents = pd.concat(lab_chunks, ignore_index=True)
        
        print(f"   Raw data: {len(patients)} patients, {len(admissions)} admissions, {len(labevents)} lab events")
        
        # Preprocess dates
        date_columns = {
            'patients': ['dob', 'dod'],
            'admissions': ['admittime', 'dischtime', 'deathtime'],
            'labevents': ['charttime']
        }
        
        for table_name, cols in date_columns.items():
            table = locals()[table_name]
            for col in cols:
                if col in table.columns:
                    table[col] = pd.to_datetime(table[col], errors='coerce')
        
        # Filter to patients with sufficient data for meaningful analysis
        print("   Filtering to patients with sufficient data...")
        
        # Patients with at least 2 admissions and some lab data
        patients_with_multiple_admissions = admissions['subject_id'].value_counts()
        valid_patients = patients_with_multiple_admissions[patients_with_multiple_admissions >= 2].index
        
        # Patients with lab data
        patients_with_labs = set(labevents['subject_id'].unique())
        
        # Intersection of both criteria
        final_patients = set(valid_patients) & patients_with_labs
        
        # Filter all tables to these patients
        patients = patients[patients['subject_id'].isin(final_patients)]
        admissions = admissions[admissions['subject_id'].isin(final_patients)]
        labevents = labevents[labevents['subject_id'].isin(final_patients)]
        
        # Focus on key lab values for chronic care
        key_lab_itemids = [
            50811,  # Hemoglobin
            50912,  # Creatinine
            50824, 50983,  # Sodium
            50822, 50971,  # Potassium
            50809, 50931,  # Glucose
        ]
        
        labevents = labevents[labevents['itemid'].isin(key_lab_itemids)]
        
        # Remove rows with missing critical values
        admissions = admissions.dropna(subset=['admittime', 'dischtime'])
        labevents = labevents.dropna(subset=['charttime', 'valuenum'])
        
        # Filter out extreme outliers in lab values (likely data errors)
        labevents = labevents[
            (labevents['valuenum'] > 0) & 
            (labevents['valuenum'] < 1000)  # Remove extreme outliers
        ]
        
        print(f"   Filtered data: {len(patients)} patients, {len(admissions)} admissions, {len(labevents)} lab events")
        
        # Add some derived features for chronic care focus
        # Calculate age at each admission
        patients_dict = patients.set_index('subject_id').to_dict('index')
        
        def calculate_age_at_admission(row):
            patient_info = patients_dict.get(row['subject_id'])
            if patient_info and pd.notna(patient_info.get('dob')):
                return (row['admittime'] - patient_info['dob']).days / 365.25
            return patient_info.get('anchor_age', 65) if patient_info else 65
        
        admissions['age_at_admission'] = admissions.apply(calculate_age_at_admission, axis=1)
        
        # Calculate length of stay
        admissions['length_of_stay'] = (admissions['dischtime'] - admissions['admittime']).dt.days
        admissions['length_of_stay'] = admissions['length_of_stay'].fillna(1).clip(lower=1)  # Minimum 1 day
        
        # Focus on adult patients (18+) for chronic care
        adult_admissions = admissions[admissions['age_at_admission'] >= 18]
        adult_patients = set(adult_admissions['subject_id'].unique())
        
        patients = patients[patients['subject_id'].isin(adult_patients)]
        admissions = adult_admissions
        labevents = labevents[labevents['subject_id'].isin(adult_patients)]
        
        print(f"   Final dataset: {len(patients)} adult patients, {len(admissions)} admissions, {len(labevents)} lab events")
        
        # Save processed data
        patients.to_csv(config.paths.processed / "mimic_patients.csv", index=False)
        admissions.to_csv(config.paths.processed / "mimic_admissions.csv", index=False)
        labevents.to_csv(config.paths.processed / "mimic_labevents.csv", index=False)
        
        return patients, admissions, labevents
    
    def create_synthetic_cohort(self, n_patients=1000):
        """Create realistic synthetic patient cohort"""
        print(f"🏥 Creating synthetic cohort of {n_patients} patients...")
        
        np.random.seed(42)
        
        # Generate patients with realistic characteristics
        patients = []
        for i in range(n_patients):
            age = np.random.normal(65, 15)
            age = max(18, min(95, age))
            
            # Correlated risk factors
            high_risk = age > 70
            diabetes_prob = 0.4 if high_risk else 0.2
            hf_prob = 0.25 if high_risk else 0.1
            
            patients.append({
                'subject_id': i + 1,
                'age': age,
                'gender': np.random.choice(['M', 'F']),
                'diabetes': np.random.choice([0, 1], p=[1-diabetes_prob, diabetes_prob]),
                'hypertension': np.random.choice([0, 1], p=[0.6, 0.4]),
                'heart_failure': np.random.choice([0, 1], p=[1-hf_prob, hf_prob]),
                'ckd': np.random.choice([0, 1], p=[0.8, 0.2]),
                'dob': datetime(1950, 1, 1) + timedelta(days=int(age * 365.25)),
                'dod': None if np.random.random() > 0.1 else datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
            })
        
        patients_df = pd.DataFrame(patients)
        
        # Generate admissions
        admissions = []
        admission_id = 1
        
        for _, patient in patients_df.iterrows():
            # Risk-based admission patterns
            base_rate = 3 if patient['age'] > 75 else 2
            n_admissions = np.random.poisson(base_rate) + 1
            
            start_date = datetime(2022, 1, 1)
            
            for adm in range(n_admissions):
                admit_date = start_date + timedelta(days=np.random.randint(0, 300))
                
                # Length of stay influenced by patient factors
                base_los = 5
                if patient['heart_failure']: base_los += 2
                if patient['ckd']: base_los += 1
                if patient['age'] > 80: base_los += 1
                
                los = max(1, np.random.exponential(base_los))
                discharge_date = admit_date + timedelta(days=int(los))
                
                # Emergency admission probability
                emergency_prob = 0.6 if patient['age'] > 75 else 0.3
                emergency = np.random.choice([0, 1], p=[1-emergency_prob, emergency_prob])
                
                admissions.append({
                    'hadm_id': admission_id,
                    'subject_id': patient['subject_id'],
                    'admittime': admit_date,
                    'dischtime': discharge_date,
                    'length_of_stay': los,
                    'admission_type': 'EMERGENCY' if emergency else 'ELECTIVE',
                    'insurance': np.random.choice(['Medicare', 'Medicaid', 'Private'], p=[0.5, 0.2, 0.3])
                })
                
                admission_id += 1
                start_date = discharge_date + timedelta(days=np.random.randint(7, 90))
        
        admissions_df = pd.DataFrame(admissions)
        
        # Generate lab events
        labs = []
        
        for _, admission in admissions_df.iterrows():
            patient = patients_df[patients_df['subject_id'] == admission['subject_id']].iloc[0]
            
            # Labs during admission
            n_labs = max(1, int(admission['length_of_stay'] / 2))
            
            for lab_day in range(n_labs):
                lab_date = admission['admittime'] + timedelta(days=lab_day)
                
                # Patient-specific lab baselines
                hgb_base = 11 if patient['gender'] == 'F' else 13
                if patient['ckd']: hgb_base -= 2
                
                creat_base = 1.0
                if patient['ckd']: creat_base = np.random.uniform(2.0, 4.0)
                elif patient['diabetes']: creat_base = np.random.uniform(1.2, 2.0)
                
                glucose_base = 180 if patient['diabetes'] else 100
                
                # Add some temporal trends (deterioration over time)
                trend_factor = lab_day / max(1, n_labs - 1)  # 0 to 1
                
                labs.append({
                    'subject_id': admission['subject_id'],
                    'hadm_id': admission['hadm_id'],
                    'charttime': lab_date,
                    'itemid': '50811',  # Hemoglobin
                    'valuenum': max(6, np.random.normal(hgb_base - trend_factor, 1.5)),
                    'valueuom': 'g/dL'
                })
                
                labs.append({
                    'subject_id': admission['subject_id'],
                    'hadm_id': admission['hadm_id'],
                    'charttime': lab_date,
                    'itemid': '50912',  # Creatinine
                    'valuenum': max(0.5, np.random.normal(creat_base + trend_factor * 0.5, 0.3)),
                    'valueuom': 'mg/dL'
                })
                
                labs.append({
                    'subject_id': admission['subject_id'],
                    'hadm_id': admission['hadm_id'],
                    'charttime': lab_date,
                    'itemid': '50931',  # Glucose
                    'valuenum': max(50, np.random.normal(glucose_base + trend_factor * 50, 30)),
                    'valueuom': 'mg/dL'
                })
        
        labs_df = pd.DataFrame(labs)
        
        # Save datasets
        patients_df.to_csv(config.paths.processed / "patients.csv", index=False)
        admissions_df.to_csv(config.paths.processed / "admissions.csv", index=False)
        labs_df.to_csv(config.paths.processed / "labevents.csv", index=False)
        
        print(f"✅ Created {len(patients_df)} patients, {len(admissions_df)} admissions, {len(labs_df)} lab results")
        
        return patients_df, admissions_df, labs_df
    
    def create_deterioration_labels(self, patients_df, admissions_df, labs_df):
        """Create 90-day deterioration labels using clinical evidence"""
        print("🏷️ Creating deterioration labels...")
        
        labels = []
        criteria = config.get_deterioration_criteria()
        
        # Sort admissions by patient and date
        admissions_sorted = admissions_df.sort_values(['subject_id', 'dischtime'])
        
        for _, admission in admissions_sorted.iterrows():
            if pd.isna(admission['dischtime']):
                continue
                
            patient_id = admission['subject_id']
            index_date = admission['dischtime']
            future_end = index_date + timedelta(days=config.features.prediction_window_days)
            
            deteriorated = False
            triggers = []
            days_to_event = None
            
            # 1. Death within 90 days
            patient = patients_df[patients_df['subject_id'] == patient_id].iloc[0]
            if pd.notna(patient.get('dod')):
                if index_date < patient['dod'] <= future_end:
                    deteriorated = True
                    triggers.append('mortality')
                    days_to_event = (patient['dod'] - index_date).days
            
            # 2. Readmission within specified window
            future_admissions = admissions_df[
                (admissions_df['subject_id'] == patient_id) &
                (admissions_df['admittime'] > index_date) &
                (admissions_df['admittime'] <= future_end)
            ]
            
            if not future_admissions.empty:
                deteriorated = True
                triggers.append('readmission')
                readmit_days = (future_admissions['admittime'].min() - index_date).days
                if days_to_event is None or readmit_days < days_to_event:
                    days_to_event = readmit_days
            
            # 3. Critical lab values
            future_labs = labs_df[
                (labs_df['subject_id'] == patient_id) &
                (labs_df['charttime'] > index_date) &
                (labs_df['charttime'] <= future_end)
            ]
            
            if not future_labs.empty:
                lab_triggers = self._check_critical_labs(future_labs, criteria['lab_thresholds'])
                if lab_triggers:
                    deteriorated = True
                    triggers.extend(lab_triggers)
            
            labels.append({
                'subject_id': patient_id,
                'hadm_id': admission.get('hadm_id'),
                'index_date': index_date,
                'deterioration_90d': int(deteriorated),
                'triggers': ';'.join(triggers) if triggers else 'none',
                'days_to_event': days_to_event,
                'risk_score': self._calculate_baseline_risk(patient, admission)
            })
        
        labels_df = pd.DataFrame(labels)
        labels_df.to_csv(self.output_dir / "deterioration_labels.csv", index=False)
        
        deterioration_rate = labels_df['deterioration_90d'].mean()
        print(f"✅ Created {len(labels_df)} labels")
        print(f"   Deterioration rate: {deterioration_rate:.1%}")
        
        return labels_df
    
    def _check_critical_labs(self, labs_df, thresholds):
        """Check for critical lab values indicating deterioration"""
        triggers = []
        
        # Map MIMIC itemids to lab concepts (including alternative itemids)
        itemid_mapping = {
            # Hemoglobin
            '50811': 'hemoglobin',
            # Creatinine  
            '50912': 'creatinine',
            # Sodium
            '50824': 'sodium', '50983': 'sodium',
            # Potassium
            '50822': 'potassium', '50971': 'potassium',
            # Glucose
            '50809': 'glucose', '50931': 'glucose'
        }
        
        for _, lab in labs_df.iterrows():
            itemid = str(lab['itemid'])
            if itemid in itemid_mapping:
                lab_name = itemid_mapping[itemid]
                value = lab['valuenum']
                
                if pd.notna(value) and lab_name in thresholds:
                    thresh = thresholds[lab_name]
                    
                    # Check critical thresholds
                    if 'critical_low' in thresh and value < thresh['critical_low']:
                        triggers.append(f'{lab_name}_critical_low')
                    elif 'critical_high' in thresh and value > thresh['critical_high']:
                        triggers.append(f'{lab_name}_critical_high')
                    elif 'low' in thresh and value < thresh['low']:
                        triggers.append(f'{lab_name}_low')
                    elif 'high' in thresh and value > thresh['high']:
                        triggers.append(f'{lab_name}_high')
        
        return list(set(triggers))  # Remove duplicates
    
    def _calculate_baseline_risk(self, patient, admission):
        """Calculate baseline risk score based on patient characteristics"""
        risk_score = 0
        
        # Age risk - use age_at_admission from admission or anchor_age from patient
        age = admission.get('age_at_admission') or patient.get('anchor_age', 65)
        if age > config.clinical.age_high_risk:
            risk_score += 2
        elif age > 65:
            risk_score += 1
            
        # Comorbidity risk
        if patient.get('diabetes', 0):
            risk_score += 1
        if patient.get('heart_failure', 0):
            risk_score += 2
        if patient.get('ckd', 0):
            risk_score += 1
            
        # Admission type risk
        if admission.get('admission_type') == 'EMERGENCY':
            risk_score += 1
            
        # Length of stay risk
        if admission.get('length_of_stay', 0) > config.clinical.long_stay_days:
            risk_score += 1
            
        return risk_score    
 
    def engineer_features(self, patients_df, admissions_df, labs_df, labels_df):
        """Create comprehensive ML features"""
        print("🔧 Engineering features...")
        
        features_list = []
        
        for _, label in labels_df.iterrows():
            patient_id = label['subject_id']
            index_date = label['index_date']
            
            # Lookback windows
            lookback_start = index_date - timedelta(days=config.features.lookback_days_max)
            recent_start = index_date - timedelta(days=config.features.recent_labs_days)
            
            # Patient demographics
            patient = patients_df[patients_df['subject_id'] == patient_id].iloc[0]
            
            # Calculate age at admission
            if pd.notna(patient.get('dob')):
                age = (index_date - patient['dob']).days / 365.25
            else:
                # For MIMIC data, use anchor_age or age_at_admission
                age = patient.get('anchor_age', 65)
                # Try to get age from admission if available
                current_admission = admissions_df[
                    (admissions_df['subject_id'] == patient_id) &
                    (admissions_df['dischtime'] == index_date)
                ]
                if not current_admission.empty and 'age_at_admission' in current_admission.columns:
                    age = current_admission.iloc[0]['age_at_admission']
            
            features = {
                'subject_id': patient_id,
                'age': age,
                'gender_M': 1 if patient.get('gender', 'M') == 'M' else 0,
                'elderly': 1 if age > config.clinical.age_high_risk else 0
            }
            
            # Comorbidities
            comorbidities = ['diabetes', 'hypertension', 'heart_failure', 'ckd']
            for comorbidity in comorbidities:
                features[comorbidity] = patient.get(comorbidity, 0)
            
            features['comorbidity_count'] = sum(patient.get(c, 0) for c in comorbidities)
            
            # Historical admissions
            hist_admissions = admissions_df[
                (admissions_df['subject_id'] == patient_id) &
                (admissions_df['dischtime'] >= lookback_start) &
                (admissions_df['dischtime'] < index_date)
            ]
            
            features.update({
                'prior_admissions_6m': len(hist_admissions),
                'avg_length_of_stay': hist_admissions['length_of_stay'].mean() if not hist_admissions.empty else 0,
                'emergency_admissions': len(hist_admissions[hist_admissions['admission_type'] == 'EMERGENCY']),
                'total_hospital_days': hist_admissions['length_of_stay'].sum(),
                'frequent_flyer': 1 if len(hist_admissions) >= config.clinical.multiple_admissions_threshold else 0
            })
            
            # Lab-based features
            patient_labs = labs_df[
                (labs_df['subject_id'] == patient_id) &
                (labs_df['charttime'] >= lookback_start) &
                (labs_df['charttime'] <= index_date)
            ]
            
            # Recent labs (last 7 days)
            recent_labs = patient_labs[patient_labs['charttime'] >= recent_start]
            
            lab_features = self._extract_lab_features(patient_labs, recent_labs)
            features.update(lab_features)
            
            # Add baseline risk score
            features['baseline_risk_score'] = label.get('risk_score', 0)
            
            # Add target
            features['deterioration_90d'] = label['deterioration_90d']
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle missing values intelligently
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
        
        features_df.to_csv(self.output_dir / "ml_features.csv", index=False)
        print(f"✅ Created {len(features_df)} feature vectors with {len(features_df.columns)-2} features")
        
        return features_df
    
    def _extract_lab_features(self, all_labs, recent_labs):
        """Extract lab-based features"""
        features = {}
        
        # Lab availability
        features['total_labs'] = len(all_labs)
        features['recent_labs'] = len(recent_labs)
        
        # MIMIC itemid mappings (including alternative itemids)
        lab_mappings = {
            'hemoglobin': ['50811'],
            'creatinine': ['50912'],
            'sodium': ['50824', '50983'],
            'potassium': ['50822', '50971'],
            'glucose': ['50809', '50931']
        }
        
        for lab_name, itemids in lab_mappings.items():
            # Get lab values
            lab_values = all_labs[all_labs['itemid'].isin(itemids)]['valuenum'].dropna()
            recent_values = recent_labs[recent_labs['itemid'].isin(itemids)]['valuenum'].dropna()
            
            if not lab_values.empty:
                # Statistical features
                features[f'{lab_name}_mean'] = lab_values.mean()
                features[f'{lab_name}_std'] = lab_values.std() if len(lab_values) > 1 else 0
                features[f'{lab_name}_min'] = lab_values.min()
                features[f'{lab_name}_max'] = lab_values.max()
                
                # Recent values
                if not recent_values.empty:
                    features[f'{lab_name}_recent'] = recent_values.iloc[-1]
                    features[f'{lab_name}_trend'] = recent_values.diff().mean() if len(recent_values) > 1 else 0
                else:
                    features[f'{lab_name}_recent'] = lab_values.iloc[-1]
                    features[f'{lab_name}_trend'] = 0
                
                # Clinical thresholds
                thresholds = config.get_deterioration_criteria()['lab_thresholds'].get(lab_name, {})
                if thresholds:
                    if 'critical_low' in thresholds:
                        features[f'{lab_name}_critical_count'] = (lab_values < thresholds['critical_low']).sum()
                    elif 'critical_high' in thresholds:
                        features[f'{lab_name}_critical_count'] = (lab_values > thresholds['critical_high']).sum()
                    else:
                        features[f'{lab_name}_critical_count'] = 0
                
            else:
                # Default values when no labs available
                default_values = {
                    'hemoglobin': 12.0, 'creatinine': 1.0, 'glucose': 100.0,
                    'sodium': 140.0, 'potassium': 4.0
                }
                default_val = default_values.get(lab_name, 0)
                
                features[f'{lab_name}_mean'] = default_val
                features[f'{lab_name}_std'] = 0
                features[f'{lab_name}_min'] = default_val
                features[f'{lab_name}_max'] = default_val
                features[f'{lab_name}_recent'] = default_val
                features[f'{lab_name}_trend'] = 0
                features[f'{lab_name}_critical_count'] = 0
        
        return features
    
    def train_models(self, features_df):
        """Train and evaluate ML models"""
        print("🤖 Training ML models...")
        
        # Prepare data
        X = features_df.drop(['subject_id', 'deterioration_90d'], axis=1)
        y = features_df['deterioration_90d']
        
        print(f"   Dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"   Positive class rate: {y.mean():.1%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.model.test_size, 
            random_state=config.model.random_state, 
            stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Models
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=config.model.rf_n_estimators,
                max_depth=config.model.rf_max_depth,
                random_state=config.model.random_state,
                class_weight='balanced'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=config.model.random_state,
                max_iter=config.model.lr_max_iter,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"   Training {name}...")
            
            if name == 'Logistic Regression':
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            auroc = roc_auc_score(y_test, y_pred_proba)
            auprc = average_precision_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'auroc': auroc,
                'auprc': auprc,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'feature_names': X.columns.tolist()
            }
            
            print(f"     AUROC: {auroc:.3f}, AUPRC: {auprc:.3f}")
        
        # Save best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['auroc'])
        best_model = results[best_model_name]
        
        joblib.dump(best_model['model'], self.models_dir / f"best_model.pkl")
        joblib.dump(scaler, self.models_dir / "scaler.pkl")
        
        # Generate evaluation
        self.create_evaluation_plots(results, best_model_name)
        
        return results, best_model_name
    
    def create_evaluation_plots(self, results, best_model_name):
        """Create evaluation visualizations"""
        print("📊 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        axes[0, 0].set_title('ROC Curves', fontsize=14, fontweight='bold')
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
            axes[0, 0].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={result['auroc']:.3f})")
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        axes[0, 1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(result['y_test'], result['y_pred_proba'])
            axes[0, 1].plot(recall, precision, linewidth=2, label=f"{name} (AP={result['auprc']:.3f})")
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature Importance (Random Forest)
        rf_result = results['Random Forest']
        importances = rf_result['model'].feature_importances_
        feature_names = rf_result['feature_names']
        indices = np.argsort(importances)[::-1][:10]
        
        axes[1, 0].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        axes[1, 0].bar(range(10), importances[indices])
        axes[1, 0].set_xticks(range(10))
        axes[1, 0].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Importance')
        
        # Confusion Matrix
        best_result = results[best_model_name]
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues')
        axes[1, 1].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Evaluation plots saved to {self.output_dir / 'model_evaluation.png'}")

def main():
    """Main execution pipeline"""
    parser = argparse.ArgumentParser(description='Advanced Chronic Risk Engine')
    parser.add_argument('--mode', choices=['synthetic', 'mimic'], 
                       default='synthetic', help='Execution mode')
    parser.add_argument('--patients', type=int, default=1000, 
                       help='Number of synthetic patients to generate')
    
    args = parser.parse_args()
    
    print("🏥 Advanced Chronic Care Risk Prediction Engine")
    print("=" * 60)
    
    engine = AdvancedChronicRiskEngine(use_real_data=(args.mode == 'mimic'))
    
    if args.mode == 'synthetic':
        # Use synthetic data
        patients_df, admissions_df, labs_df = engine.create_synthetic_cohort(n_patients=args.patients)
    elif args.mode == 'mimic':
        # Use real MIMIC data
        patients_df, admissions_df, labs_df = engine.load_mimic_data()
    
    # Create deterioration labels
    labels_df = engine.create_deterioration_labels(patients_df, admissions_df, labs_df)
    
    # Engineer features
    features_df = engine.engineer_features(patients_df, admissions_df, labs_df, labels_df)
    
    # Train models
    results, best_model_name = engine.train_models(features_df)
    
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Results saved in: {engine.output_dir}")
    print(f"🏆 Best model: {best_model_name}")
    print(f"📊 AUROC: {results[best_model_name]['auroc']:.3f}")
    
    print("\n📋 Next steps:")
    print("1. Review model_evaluation.png for performance metrics")
    print("2. Check ml_features.csv for feature analysis")
    print("3. Use trained models for risk prediction")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Chronic Care Risk Prediction Engine - Hackathon Version
======================================================

A clean, minimal implementation for 90-day deterioration prediction.
Focuses on getting results quickly with explainable AI.

Usage:
    python chronic_risk_engine.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import joblib

class ChronicRiskEngine:
    def __init__(self):
        self.output_dir = Path("risk_engine_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Simple, evidence-based thresholds
        self.risk_thresholds = {
            'age_high_risk': 75,
            'readmission_days': 30,  # High risk if readmitted within 30 days
            'los_threshold': 7,      # Long stay indicator
            'emergency_threshold': 2  # Multiple emergency visits
        }
        
    def create_synthetic_cohort(self, n_patients=1000):
        """Create a realistic synthetic patient cohort for demonstration"""
        print(f"🏥 Creating synthetic cohort of {n_patients} patients...")
        
        np.random.seed(42)
        
        # Generate patients
        patients = []
        for i in range(n_patients):
            age = np.random.normal(65, 15)
            age = max(18, min(95, age))  # Clip to reasonable range
            
            # Risk factors influence each other
            high_risk = age > 70
            
            patients.append({
                'patient_id': f'P{i:04d}',
                'age': age,
                'gender': np.random.choice(['M', 'F']),
                'diabetes': np.random.choice([0, 1], p=[0.7, 0.3]),
                'hypertension': np.random.choice([0, 1], p=[0.6, 0.4]),
                'heart_failure': np.random.choice([0, 1], p=[0.85, 0.15]),
                'ckd': np.random.choice([0, 1], p=[0.8, 0.2]),
                'high_risk_baseline': int(high_risk)
            })
        
        patients_df = pd.DataFrame(patients)
        
        # Generate admissions with realistic patterns
        admissions = []
        admission_id = 1
        
        for _, patient in patients_df.iterrows():
            # Number of admissions (high-risk patients have more)
            base_rate = 2 if patient['high_risk_baseline'] else 1
            n_admissions = np.random.poisson(base_rate) + 1
            
            start_date = datetime(2023, 1, 1)
            
            for adm in range(n_admissions):
                admit_date = start_date + timedelta(days=np.random.randint(0, 300))
                los = np.random.exponential(4) + 1  # Length of stay
                discharge_date = admit_date + timedelta(days=int(los))
                
                # Emergency admission more likely for high-risk
                emergency = np.random.choice([0, 1], p=[0.7, 0.3] if not patient['high_risk_baseline'] else [0.4, 0.6])
                
                admissions.append({
                    'admission_id': admission_id,
                    'patient_id': patient['patient_id'],
                    'admit_date': admit_date,
                    'discharge_date': discharge_date,
                    'length_of_stay': los,
                    'emergency_admission': emergency,
                    'admission_type': 'Emergency' if emergency else 'Elective'
                })
                
                admission_id += 1
                start_date = discharge_date + timedelta(days=np.random.randint(1, 60))
        
        admissions_df = pd.DataFrame(admissions)
        
        # Generate lab values
        labs = []
        lab_id = 1
        
        for _, admission in admissions_df.iterrows():
            patient = patients_df[patients_df['patient_id'] == admission['patient_id']].iloc[0]
            
            # Number of lab draws during admission
            n_labs = max(1, int(admission['length_of_stay'] / 2))
            
            for lab in range(n_labs):
                lab_date = admission['admit_date'] + timedelta(days=lab)
                
                # Generate realistic lab values with patient-specific trends
                base_hgb = 12 if patient['gender'] == 'F' else 14
                if patient['ckd']:
                    base_hgb -= 2  # Anemia in CKD
                
                base_creat = 1.0
                if patient['ckd']:
                    base_creat = np.random.uniform(2.0, 4.0)
                elif patient['diabetes']:
                    base_creat = np.random.uniform(1.2, 2.0)
                
                labs.append({
                    'lab_id': lab_id,
                    'patient_id': admission['patient_id'],
                    'admission_id': admission['admission_id'],
                    'lab_date': lab_date,
                    'hemoglobin': max(6, np.random.normal(base_hgb, 1.5)),
                    'creatinine': max(0.5, np.random.normal(base_creat, 0.3)),
                    'sodium': np.random.normal(140, 3),
                    'potassium': np.random.normal(4.0, 0.4),
                    'glucose': np.random.normal(120 if not patient['diabetes'] else 180, 30)
                })
                
                lab_id += 1
        
        labs_df = pd.DataFrame(labs)
        
        # Save datasets
        patients_df.to_csv(self.output_dir / "patients.csv", index=False)
        admissions_df.to_csv(self.output_dir / "admissions.csv", index=False)
        labs_df.to_csv(self.output_dir / "labs.csv", index=False)
        
        print(f"✅ Created {len(patients_df)} patients, {len(admissions_df)} admissions, {len(labs_df)} lab results")
        
        return patients_df, admissions_df, labs_df
    
    def create_deterioration_labels(self, patients_df, admissions_df, labs_df):
        """Create 90-day deterioration labels using clinical criteria"""
        print("🏷️ Creating deterioration labels...")
        
        labels = []
        
        # Sort admissions by patient and date
        admissions_sorted = admissions_df.sort_values(['patient_id', 'discharge_date'])
        
        for _, admission in admissions_sorted.iterrows():
            patient_id = admission['patient_id']
            index_date = admission['discharge_date']
            future_end = index_date + timedelta(days=90)
            
            # Check for deterioration events
            deteriorated = False
            triggers = []
            days_to_event = None
            
            # 1. Readmission within 90 days
            future_admissions = admissions_df[
                (admissions_df['patient_id'] == patient_id) &
                (admissions_df['admit_date'] > index_date) &
                (admissions_df['admit_date'] <= future_end)
            ]
            
            if not future_admissions.empty:
                deteriorated = True
                triggers.append('readmission')
                days_to_event = (future_admissions['admit_date'].min() - index_date).days
            
            # 2. Critical lab values in future window
            future_labs = labs_df[
                (labs_df['patient_id'] == patient_id) &
                (labs_df['lab_date'] > index_date) &
                (labs_df['lab_date'] <= future_end)
            ]
            
            if not future_labs.empty:
                # Check for critical values
                if (future_labs['hemoglobin'] < 8).any():
                    deteriorated = True
                    triggers.append('severe_anemia')
                
                if (future_labs['creatinine'] > 3.0).any():
                    deteriorated = True
                    triggers.append('acute_kidney_injury')
                
                if (future_labs['potassium'] > 6.0).any() or (future_labs['potassium'] < 3.0).any():
                    deteriorated = True
                    triggers.append('electrolyte_imbalance')
            
            # 3. Simulated mortality (5% baseline risk, higher for high-risk patients)
            patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]
            mortality_risk = 0.05 if not patient['high_risk_baseline'] else 0.15
            
            if np.random.random() < mortality_risk:
                deteriorated = True
                triggers.append('mortality')
                if days_to_event is None:
                    days_to_event = np.random.randint(1, 90)
            
            labels.append({
                'patient_id': patient_id,
                'admission_id': admission['admission_id'],
                'index_date': index_date,
                'deterioration_90d': int(deteriorated),
                'triggers': ';'.join(triggers) if triggers else 'none',
                'days_to_event': days_to_event
            })
        
        labels_df = pd.DataFrame(labels)
        labels_df.to_csv(self.output_dir / "labels.csv", index=False)
        
        deterioration_rate = labels_df['deterioration_90d'].mean()
        print(f"✅ Created {len(labels_df)} labels with {deterioration_rate:.1%} deterioration rate")
        
        return labels_df
    
    def engineer_features(self, patients_df, admissions_df, labs_df, labels_df):
        """Create ML features from patient data"""
        print("🔧 Engineering features...")
        
        features_list = []
        
        for _, label in labels_df.iterrows():
            patient_id = label['patient_id']
            index_date = label['index_date']
            
            # Lookback window (30-180 days before discharge)
            lookback_start = index_date - timedelta(days=180)
            
            # Patient demographics
            patient = patients_df[patients_df['patient_id'] == patient_id].iloc[0]
            
            features = {
                'patient_id': patient_id,
                'age': patient['age'],
                'gender_M': 1 if patient['gender'] == 'M' else 0,
                'diabetes': patient['diabetes'],
                'hypertension': patient['hypertension'],
                'heart_failure': patient['heart_failure'],
                'ckd': patient['ckd'],
                'comorbidity_count': patient['diabetes'] + patient['hypertension'] + patient['heart_failure'] + patient['ckd']
            }
            
            # Historical admissions (6 months lookback)
            hist_admissions = admissions_df[
                (admissions_df['patient_id'] == patient_id) &
                (admissions_df['discharge_date'] >= lookback_start) &
                (admissions_df['discharge_date'] < index_date)
            ]
            
            features.update({
                'prior_admissions_6m': len(hist_admissions),
                'avg_length_of_stay': hist_admissions['length_of_stay'].mean() if not hist_admissions.empty else 0,
                'emergency_admissions': hist_admissions['emergency_admission'].sum(),
                'total_hospital_days': hist_admissions['length_of_stay'].sum()
            })
            
            # Recent lab trends (30 days before discharge)
            recent_labs = labs_df[
                (labs_df['patient_id'] == patient_id) &
                (labs_df['lab_date'] >= index_date - timedelta(days=30)) &
                (labs_df['lab_date'] <= index_date)
            ]
            
            if not recent_labs.empty:
                features.update({
                    'last_hemoglobin': recent_labs['hemoglobin'].iloc[-1],
                    'last_creatinine': recent_labs['creatinine'].iloc[-1],
                    'last_sodium': recent_labs['sodium'].iloc[-1],
                    'last_potassium': recent_labs['potassium'].iloc[-1],
                    'last_glucose': recent_labs['glucose'].iloc[-1],
                    'hemoglobin_trend': recent_labs['hemoglobin'].diff().mean(),
                    'creatinine_trend': recent_labs['creatinine'].diff().mean(),
                    'abnormal_labs': ((recent_labs['hemoglobin'] < 10) | 
                                    (recent_labs['creatinine'] > 1.5) |
                                    (recent_labs['potassium'] < 3.5) |
                                    (recent_labs['potassium'] > 5.0)).sum()
                })
            else:
                # Default values if no recent labs
                features.update({
                    'last_hemoglobin': 12, 'last_creatinine': 1.0, 'last_sodium': 140,
                    'last_potassium': 4.0, 'last_glucose': 100, 'hemoglobin_trend': 0,
                    'creatinine_trend': 0, 'abnormal_labs': 0
                })
            
            # Add target
            features['deterioration_90d'] = label['deterioration_90d']
            
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        
        # Handle any remaining missing values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].median())
        
        features_df.to_csv(self.output_dir / "features.csv", index=False)
        print(f"✅ Created {len(features_df)} feature vectors with {len(features_df.columns)-2} features")
        
        return features_df    

    def train_models(self, features_df):
        """Train and evaluate ML models"""
        print("🤖 Training ML models...")
        
        # Prepare data
        X = features_df.drop(['patient_id', 'deterioration_90d'], axis=1)
        y = features_df['deterioration_90d']
        
        print(f"   Dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"   Positive class rate: {y.mean():.1%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
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
        
        # Save models
        best_model_name = max(results.keys(), key=lambda k: results[k]['auroc'])
        joblib.dump(results[best_model_name]['model'], self.output_dir / "best_model.pkl")
        joblib.dump(scaler, self.output_dir / "scaler.pkl")
        
        # Generate evaluation plots
        self.create_evaluation_plots(results)
        
        return results, best_model_name
    
    def create_evaluation_plots(self, results):
        """Create comprehensive evaluation visualizations"""
        print("📊 Creating evaluation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        axes[0, 2].set_title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        axes[0, 2].bar(range(10), importances[indices])
        axes[0, 2].set_xticks(range(10))
        axes[0, 2].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        axes[0, 2].set_ylabel('Importance')
        
        # Confusion Matrix
        best_result = max(results.values(), key=lambda x: x['auroc'])
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix (Best Model)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Risk Score Distribution
        axes[1, 1].set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
        low_risk = best_result['y_pred_proba'][best_result['y_test'] == 0]
        high_risk = best_result['y_pred_proba'][best_result['y_test'] == 1]
        axes[1, 1].hist(low_risk, bins=20, alpha=0.7, label='No Deterioration', color='green')
        axes[1, 1].hist(high_risk, bins=20, alpha=0.7, label='Deterioration', color='red')
        axes[1, 1].set_xlabel('Risk Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()
        
        # Model Comparison
        model_names = list(results.keys())
        auroc_scores = [results[name]['auroc'] for name in model_names]
        auprc_scores = [results[name]['auprc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, auroc_scores, width, label='AUROC', alpha=0.8)
        axes[1, 2].bar(x + width/2, auprc_scores, width, label='AUPRC', alpha=0.8)
        axes[1, 2].set_xlabel('Models')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(model_names)
        axes[1, 2].legend()
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show for headless operation
    
    def create_dashboard_data(self, features_df, results, best_model_name):
        """Create dashboard-ready outputs"""
        print("📋 Creating dashboard data...")
        
        best_model = results[best_model_name]['model']
        
        # Patient risk scores
        X = features_df.drop(['patient_id', 'deterioration_90d'], axis=1)
        
        if best_model_name == 'Logistic Regression':
            scaler = joblib.load(self.output_dir / "scaler.pkl")
            X_scaled = scaler.transform(X)
            risk_scores = best_model.predict_proba(X_scaled)[:, 1]
        else:
            risk_scores = best_model.predict_proba(X)[:, 1]
        
        # Create cohort view
        cohort_df = features_df[['patient_id', 'age', 'gender_M', 'comorbidity_count', 'deterioration_90d']].copy()
        cohort_df['risk_score'] = risk_scores
        cohort_df['risk_category'] = pd.cut(risk_scores, 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        cohort_df['gender'] = cohort_df['gender_M'].map({1: 'Male', 0: 'Female'})
        
        cohort_df.to_csv(self.output_dir / "cohort_dashboard.csv", index=False)
        
        # Feature importance for explainability
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(best_model.coef_[0])
            }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.output_dir / "feature_importance.csv", index=False)
        
        # Summary statistics
        summary = {
            'total_patients': len(cohort_df),
            'deterioration_rate': cohort_df['deterioration_90d'].mean(),
            'high_risk_patients': len(cohort_df[cohort_df['risk_category'] == 'High']),
            'model_performance': {
                'auroc': results[best_model_name]['auroc'],
                'auprc': results[best_model_name]['auprc']
            },
            'top_risk_factors': feature_importance.head(5)['feature'].tolist()
        }
        
        # Save summary
        import json
        with open(self.output_dir / "dashboard_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Dashboard data created:")
        print(f"   - {summary['total_patients']} patients analyzed")
        print(f"   - {summary['high_risk_patients']} high-risk patients identified")
        print(f"   - Model AUROC: {summary['model_performance']['auroc']:.3f}")
        
        return cohort_df, feature_importance, summary

def main():
    """Main execution pipeline"""
    print("🏥 Chronic Care Risk Prediction Engine")
    print("=" * 50)
    
    engine = ChronicRiskEngine()
    
    # Step 1: Create synthetic data (or load your real data here)
    patients_df, admissions_df, labs_df = engine.create_synthetic_cohort(n_patients=1000)
    
    # Step 2: Create deterioration labels
    labels_df = engine.create_deterioration_labels(patients_df, admissions_df, labs_df)
    
    # Step 3: Engineer features
    features_df = engine.engineer_features(patients_df, admissions_df, labs_df, labels_df)
    
    # Step 4: Train models
    results, best_model_name = engine.train_models(features_df)
    
    # Step 5: Create dashboard outputs
    cohort_df, feature_importance, summary = engine.create_dashboard_data(features_df, results, best_model_name)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Results saved in: {engine.output_dir}")
    print("\nNext steps:")
    print("1. Review model_evaluation.png for performance metrics")
    print("2. Check cohort_dashboard.csv for patient risk scores")
    print("3. Use feature_importance.csv for explainability")

if __name__ == "__main__":
    main()
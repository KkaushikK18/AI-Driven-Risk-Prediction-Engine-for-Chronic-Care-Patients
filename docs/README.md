# 🏥 AI-Driven Risk Prediction Engine for Chronic Care Patients

A clean, hackathon-ready implementation for predicting 90-day patient deterioration using clinical data.

## 🎯 **Problem Statement**
Predict whether chronic care patients are at risk of deterioration within 90 days using:
- Patient demographics and medical history
- Lab results and vital signs  
- Admission patterns and medication adherence
- Provide explainable, clinician-friendly predictions

## 🚀 **Quick Start**

### 1. Clean Setup (First Time)
```bash
# Run cleanup (removes old complex files)
cleanup_commands.bat

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn joblib

# Run the engine
python chronic_risk_engine.py
```

### 2. Using Your Real Data
```python
# Edit chronic_risk_engine.py, replace synthetic data generation with:
patients_df = pd.read_csv("data/raw/mimic/patients.csv")
admissions_df = pd.read_csv("data/raw/mimic/admissions.csv") 
labs_df = pd.read_csv("data/raw/mimic/labevents.csv")
```

## 📁 **Project Structure**
```
📁 Chronic Risk Engine/
├── 📄 chronic_risk_engine.py     # Main prediction engine
├── 📄 config.py                  # Configuration & thresholds
├── 📄 README.md                  # This file
├── 📄 cleanup_commands.bat       # Project cleanup script
├── 📁 data/
│   ├── 📁 raw/mimic/             # Original MIMIC-IV data
│   ├── 📁 raw/synthea/           # Synthea synthetic data
│   ├── 📁 processed/             # Clean processed datasets
│   └── 📁 results/               # Model outputs & dashboards
├── 📁 models/                    # Trained ML models
└── 📁 dashboard/                 # Dashboard files
```

## 🔧 **Key Features**

### ✅ **Prediction Model**
- **Random Forest** + **Logistic Regression** ensemble
- **AUROC** and **AUPRC** evaluation metrics
- **Cross-validation** for robust performance assessment
- **Feature importance** for explainability

### ✅ **Clinical Deterioration Criteria**
- **Readmission** within 30-90 days
- **Critical lab values** (anemia, kidney injury, electrolyte imbalance)
- **Mortality risk** based on patient characteristics
- **Evidence-based thresholds** from clinical guidelines

### ✅ **Feature Engineering**
- **Demographics**: Age, gender, comorbidities
- **Admission history**: Frequency, length of stay, emergency visits
- **Lab trends**: Recent values, trends, abnormal counts
- **Risk factors**: Diabetes, hypertension, heart failure, CKD

### ✅ **Explainability**
- **Global importance**: Which features matter most overall
- **Local explanations**: Why each patient is high/low risk
- **Clinical interpretations**: Translate ML outputs to medical terms
- **Actionable insights**: Specific recommendations for care teams

## 📊 **Output Files**

After running, check `data/results/` for:
- `cohort_dashboard.csv` - Patient risk scores for dashboard
- `feature_importance.csv` - Feature explanations
- `model_evaluation.png` - Performance visualizations
- `dashboard_summary.json` - Key metrics and insights

## 🎨 **Dashboard Ready**

The engine produces dashboard-ready outputs:
- **Cohort View**: Risk scores for all patients
- **Patient Detail**: Individual risk factors and trends  
- **Risk Categories**: Low/Medium/High risk stratification
- **Actionable Alerts**: Patients needing immediate attention

## ⚙️ **Configuration**

Customize clinical thresholds in `config.py`:
```python
# Example: Adjust hemoglobin thresholds
config.clinical.hemoglobin_critical_low = 7.0  # Severe anemia
config.clinical.hemoglobin_low = 10.0           # Mild anemia
```

## 🏆 **Hackathon Advantages**

1. **Fast Results** - Works immediately with synthetic data
2. **Real Data Ready** - Easy to plug in MIMIC/Synthea datasets  
3. **Clinical Validity** - Evidence-based deterioration criteria
4. **Explainable AI** - Clear feature importance and interpretations
5. **Dashboard Ready** - Structured outputs for visualization
6. **Scalable Design** - Modular architecture for extensions

## 📈 **Performance Metrics**

The engine automatically calculates:
- **AUROC** (Area Under ROC Curve) - Overall discrimination
- **AUPRC** (Area Under Precision-Recall Curve) - Performance on imbalanced data
- **Confusion Matrix** - True/false positives and negatives
- **Feature Importance** - Which factors drive predictions
- **Calibration** - How well predicted probabilities match reality

## 🔬 **Clinical Validation**

Deterioration criteria based on:
- **AKI Guidelines** - Creatinine thresholds from KDIGO
- **Anemia Standards** - Hemoglobin levels from WHO/ASH
- **Electrolyte Management** - K+/Na+ ranges from clinical practice
- **Readmission Research** - 30-day readmission as quality indicator

## 🚀 **Next Steps for Hackathon**

1. **Run baseline** - Execute with synthetic data
2. **Integrate real data** - Load your MIMIC/Synthea datasets
3. **Tune thresholds** - Adjust clinical criteria if needed
4. **Build dashboard** - Create web interface for results
5. **Present solution** - Use visualizations and explanations

## 💡 **Extensions**

Easy to add:
- **More data sources** - EHR, wearables, social determinants
- **Advanced models** - XGBoost, neural networks, ensemble methods
- **Real-time scoring** - API endpoints for live predictions
- **Clinical workflows** - Integration with EHR systems
- **Outcome tracking** - Monitor prediction accuracy over time

---

**Ready to predict patient deterioration with explainable AI!** 🎯
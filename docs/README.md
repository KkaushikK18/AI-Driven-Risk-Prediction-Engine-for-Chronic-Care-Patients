# 🏥 AI-Driven Risk Prediction for Chronic Care Patients

An **AI-powered risk prediction engine** that forecasts the probability of **clinical deterioration within 90 days** for patients with chronic illnesses.  
Built on **MIMIC-IV clinical data** and validated with **synthetic Synthea patients**, this project combines advanced ML models with an interactive dashboard for **clinician-friendly insights**.

---

## 🚀 Key Highlights
- Predicts deterioration **90 days in advance** using 30–180 days of history  
- Uses **MIMIC-IV ICU cohort (100 patients, 275 admissions)**  
- Validated on **5,000 Synthea synthetic patients**  
- AUROC up to **0.99 on MIMIC-IV**, **0.75 on Synthea**  
- Transparent **feature importance** and patient-level explanations  
- Interactive **dashboard** for live risk simulation  

---

## 📂 Project Structure

📁 Chronic-Care-Risk-Engine/

├── 🏆 HACKATHON_DEMO/
│ ├── MAIN_DASHBOARD.html # Demo-ready interactive dashboard
│ ├── MAIN_ENGINE.py # Enhanced ML 
│ ├── PRESENTATION_GUIDE.md # Guide for presenting
│ └── PROJECT_OVERVIEW.md # Project summary


├── 📁 src/
│ ├── engines/ # ML engines (basic → advanced)
│ │ ├── enhanced_chronic_risk_engine.py
│ │ ├── advanced_chronic_risk_engine.py
│ │ └── chronic_risk_engine.py
│ ├── dashboard/ # Dashboard implementations
│ │ ├── static_dashboard.html
│ │ ├── chronic_care_dashboard.py
│ │ └── setup_dashboard.py
│ └── utils/ # Config & data loaders
│ ├── config.py
│ └── load_real_data.py


├── 📁 data/ # Data (not pushed to GitHub)
│ ├── raw/ # MIMIC-IV raw data
│ ├── processed/ # Preprocessed datasets
│ └── results/ # Model outputs & plots


├── 📁 models/ # Saved trained models


├── 📁 docs/ # Documentation


│ ├── README.md # This file
│ ├── DASHBOARD_GUIDE.md
│ └── CLEANUP_AND_RESTART_GUIDE.md


├── 📁 scripts/ # Utility scripts
└── requirements.txt # Dependencies


---

## 📊 Model Performance

| Dataset            | AUROC | AUPRC | Calibration | Notes                          |
|--------------------|-------|-------|-------------|--------------------------------|
| **MIMIC-IV (100 pts, 275 admits)** | 0.99  | 0.99  | Excellent  | Small but high-quality ICU subset |
| **Synthea (5000 pts)**             | 0.75  | 0.73  | Good       | Synthetic validation set          |

<img width="848" height="727" alt="Screenshot 2025-09-09 202957" src="https://github.com/user-attachments/assets/815005ad-9bb7-493b-953e-7ea5a895be33" />


<img width="399" height="433" alt="Screenshot 2025-09-09 234624" src="https://github.com/user-attachments/assets/4289ffe2-cdce-40b1-b5ad-f7f795674383" />




## 🧠 Features Used
- **Demographics**: Age, gender  
- **Clinical history**: Number of prior admissions, comorbidity burden  
- **Lab values**: Creatinine, Hemoglobin  
- **Comorbidities**: Diabetes, CKD, COPD, Heart Failure  
- **Scores & derived metrics**: Charlson index, SOFA score, lab ratios, interaction terms  

<img width="930" height="577" alt="Screenshot 2025-09-09 234649" src="https://github.com/user-attachments/assets/ea5aa80e-d6bd-475a-8d39-a597ef987e0b" />


## 🛠️ How It Works

1. **Data preprocessing** → Convert MIMIC-IV / Synthea into structured features  
2. **Model training** → Ensemble (XGBoost + RF + Logistic Regression)  
3. **Prediction** → Outputs patient-level deterioration probability  
4. **Explanation** → Global feature importance + Local explanations per patient  
5. **Dashboard integration** → Risk scores + clinical reasoning displayed interactively  
 

---

## 🖥️ Dashboard Demo

The dashboard lets clinicians:  
- Adjust **age, labs, admissions, comorbidities**  
- View **real-time risk score updates**  
- See **transparent explanations** for each prediction  

<img width="1564" height="920" alt="Screenshot 2025-09-09 234755" src="https://github.com/user-attachments/assets/1aba4580-5e51-4cad-95d2-197f3cbf8e0d" />


---

## 📈 Example Outputs

- **Risk prediction file:** `enhanced_deterioration_labels.csv`  
- **Performance plots:** `comprehensive_model_evaluation.png`, `enhanced_synthea_model_results.png`  
- **Feature importances:** `comprehensive_feature_importance.csv`  

<img width="751" height="542" alt="Screenshot 2025-09-09 234837" src="https://github.com/user-attachments/assets/007fa9ac-78a2-48e9-87b7-5b69420e8d87" />


<img width="1102" height="438" alt="Screenshot 2025-09-09 234905" src="https://github.com/user-attachments/assets/2290ee4a-b7b3-48ee-9d2e-19de85d3bc0f" />


---

## ⚡ Impact
- Earlier detection of deterioration → proactive care  
- Reduced hospital readmissions → cost savings  
- Explainable AI → clinician trust  

---

## ⚠️ Limitations
- Small MIMIC-IV cohort (100 patients)  
- Validation so far limited to synthetic patients  
- ICU-focused; needs broader population testing  

---

## 🔮 Next Steps
- Scale to larger, diverse real-world cohorts  
- Integrate into **hospital EHR systems**  
- Incorporate wearable device data + NLP on clinical notes  
- Real-world clinical validation  

---

## 📦 Installation

```bash
git clone https://github.com/<your-username>/Chronic-Care-Risk-Engine.git
cd Chronic-Care-Risk-Engine
pip install -r requirements.txt

python HACKATHON_DEMO\MAIN_ENGINE.py #for real MIMIC-IV Results
# Results in data/results

python enhanced_synthea_results_generator.py #for Synthea generated synthetic patients results
#Results in synthea_results

# For Dashbaord, open the HACKATHON_DEMO/MAIN_DASHBOARD.html in webpage for the Risk Prediction

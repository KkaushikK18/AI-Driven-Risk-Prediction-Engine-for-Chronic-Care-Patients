import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Load processed feature dataset (same as MAIN_ENGINE uses)
features_path = RESULTS_DIR / "enhanced_ml_features.csv"
labels_path = RESULTS_DIR / "enhanced_deterioration_labels.csv"

features_df = pd.read_csv(features_path)
labels_df = pd.read_csv(labels_path)

# Dashboard subset features
dashboard_features = [
    "age", "gender_M", "prior_admissions_6m",
    "diabetes_flag", "heart_failure_flag", "ckd_flag", "copd_flag",
    "hemoglobin_min", "creatinine_mean"  # adjust if your dataset uses different names
]

# Keep only columns that exist
available_features = [f for f in dashboard_features if f in features_df.columns]
X = features_df[available_features]
y = labels_df["deterioration_90d"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale + train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

print("✅ Dashboard model trained")
print("Train score:", model.score(X_train_scaled, y_train))
print("Test score:", model.score(X_test_scaled, y_test))

# Save artifacts
joblib.dump(model, MODELS_DIR / "dashboard_model.pkl")
joblib.dump(scaler, MODELS_DIR / "dashboard_scaler.pkl")

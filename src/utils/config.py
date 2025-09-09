"""
Configuration Management for Chronic Risk Engine
===============================================

Centralized configuration for data paths, model parameters, and clinical thresholds.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class DataPaths:
    """Data directory paths"""
    root: Path = Path("data")
    raw_mimic: Path = Path("data/raw/mimic")
    raw_synthea: Path = Path("data/raw/synthea")
    processed: Path = Path("data/processed")
    results: Path = Path("data/results")
    models: Path = Path("models")
    dashboard: Path = Path("dashboard")
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for path in [self.processed, self.results, self.models, self.dashboard]:
            path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """ML model configuration"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    rf_min_samples_split: int = 5
    
    # Logistic Regression parameters
    lr_max_iter: int = 1000
    lr_C: float = 1.0

@dataclass
class ClinicalThresholds:
    """Evidence-based clinical thresholds for deterioration detection"""
    
    # Lab value thresholds (critical values indicating deterioration)
    hemoglobin_critical_low: float = 7.0      # g/dL - Severe anemia
    hemoglobin_low: float = 10.0              # g/dL - Mild anemia
    
    creatinine_critical_high: float = 3.0     # mg/dL - Acute kidney injury
    creatinine_high: float = 1.5              # mg/dL - Kidney impairment
    
    sodium_critical_low: float = 125          # mEq/L - Severe hyponatremia
    sodium_critical_high: float = 155         # mEq/L - Severe hypernatremia
    sodium_low: float = 135                   # mEq/L - Mild hyponatremia
    sodium_high: float = 145                  # mEq/L - Mild hypernatremia
    
    potassium_critical_low: float = 2.8       # mEq/L - Severe hypokalemia
    potassium_critical_high: float = 6.2      # mEq/L - Severe hyperkalemia
    potassium_low: float = 3.5                # mEq/L - Mild hypokalemia
    potassium_high: float = 5.0               # mEq/L - Mild hyperkalemia
    
    glucose_critical_low: float = 50          # mg/dL - Severe hypoglycemia
    glucose_critical_high: float = 400        # mg/dL - Severe hyperglycemia
    glucose_high: float = 200                 # mg/dL - Hyperglycemia
    
    # Vital sign thresholds
    systolic_bp_critical_low: float = 80      # mmHg - Shock
    systolic_bp_critical_high: float = 200    # mmHg - Hypertensive crisis
    systolic_bp_low: float = 90               # mmHg - Hypotension
    systolic_bp_high: float = 160             # mmHg - Hypertension
    
    diastolic_bp_critical_low: float = 50     # mmHg - Severe hypotension
    diastolic_bp_critical_high: float = 120   # mmHg - Severe hypertension
    
    # Risk factor thresholds
    age_high_risk: int = 75                   # Age threshold for high risk
    readmission_window_days: int = 30         # Days for readmission risk
    long_stay_days: int = 7                   # Long hospital stay threshold
    multiple_admissions_threshold: int = 2     # Multiple admissions indicator

@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    lookback_days_min: int = 30               # Minimum lookback for features
    lookback_days_max: int = 180              # Maximum lookback for features
    prediction_window_days: int = 90          # Prediction window (target)
    
    # Lab aggregation windows
    recent_labs_days: int = 7                 # Recent lab values window
    trend_calculation_days: int = 30          # Trend calculation window
    
    # Missing value handling
    fill_missing_with_median: bool = True
    drop_high_missing_threshold: float = 0.8  # Drop features with >80% missing

class RiskEngineConfig:
    """Main configuration class"""
    
    def __init__(self):
        self.paths = DataPaths()
        self.model = ModelConfig()
        self.clinical = ClinicalThresholds()
        self.features = FeatureConfig()
        
        # MIMIC-IV specific mappings
        self.mimic_lab_mappings = {
            'hemoglobin': ['50811', '51222', '51248'],  # Hemoglobin itemids
            'creatinine': ['50912'],                     # Creatinine itemids
            'sodium': ['50824', '50983'],                # Sodium itemids
            'potassium': ['50822', '50971'],             # Potassium itemids
            'glucose': ['50809', '50931'],               # Glucose itemids
        }
        
        # Synthea LOINC mappings
        self.synthea_loinc_mappings = {
            'hemoglobin': ['718-7'],                     # Hemoglobin LOINC
            'creatinine': ['2160-0'],                    # Creatinine LOINC
            'sodium': ['2951-2'],                        # Sodium LOINC
            'potassium': ['2823-3'],                     # Potassium LOINC
            'glucose': ['2345-7', '33747-0'],            # Glucose LOINC codes
        }
    
    def get_lab_itemids(self, lab_name: str, data_source: str = 'both') -> List[str]:
        """Get itemids for a specific lab based on data source"""
        itemids = []
        
        if data_source in ['mimic', 'both']:
            itemids.extend(self.mimic_lab_mappings.get(lab_name, []))
        
        if data_source in ['synthea', 'both']:
            itemids.extend(self.synthea_loinc_mappings.get(lab_name, []))
        
        return itemids
    
    def get_deterioration_criteria(self) -> Dict:
        """Get structured deterioration criteria for labeling"""
        return {
            'readmission_days': self.clinical.readmission_window_days,
            'lab_thresholds': {
                'hemoglobin': {
                    'critical_low': self.clinical.hemoglobin_critical_low,
                    'low': self.clinical.hemoglobin_low
                },
                'creatinine': {
                    'critical_high': self.clinical.creatinine_critical_high,
                    'high': self.clinical.creatinine_high
                },
                'sodium': {
                    'critical_low': self.clinical.sodium_critical_low,
                    'critical_high': self.clinical.sodium_critical_high,
                    'low': self.clinical.sodium_low,
                    'high': self.clinical.sodium_high
                },
                'potassium': {
                    'critical_low': self.clinical.potassium_critical_low,
                    'critical_high': self.clinical.potassium_critical_high,
                    'low': self.clinical.potassium_low,
                    'high': self.clinical.potassium_high
                },
                'glucose': {
                    'critical_low': self.clinical.glucose_critical_low,
                    'critical_high': self.clinical.glucose_critical_high,
                    'high': self.clinical.glucose_high
                }
            },
            'vital_thresholds': {
                'systolic_bp': {
                    'critical_low': self.clinical.systolic_bp_critical_low,
                    'critical_high': self.clinical.systolic_bp_critical_high,
                    'low': self.clinical.systolic_bp_low,
                    'high': self.clinical.systolic_bp_high
                },
                'diastolic_bp': {
                    'critical_low': self.clinical.diastolic_bp_critical_low,
                    'critical_high': self.clinical.diastolic_bp_critical_high
                }
            }
        }

# Global configuration instance
config = RiskEngineConfig()
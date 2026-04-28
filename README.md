# 🏥 Diabetes Hospital Readmission Prediction

Predicting whether a diabetic patient will be readmitted to hospital 
within 30 days using real clinical data from 130 US hospitals.

## 📊 Dataset
- Source: UCI ML Repository — Diabetes 130-US Hospitals (1999-2008)
- Size: 101,766 patient encounters × 50 features
- Target: Early readmission within 30 days (binary classification)

## 🗂️ Project Structure
| File | Description |
|------|-------------|
| `01_EDA.ipynb` | Exploratory Data Analysis |
| `02_Feature_Engineering.ipynb` | Cleaning, encoding & feature creation |
| `diabetic_data.csv` | Raw dataset |
| `diabetic_ml_ready.csv` | ML-ready dataset (101K rows × 67 features) |

## 🔍 Key Findings So Far
- Only 11.2% of patients are readmitted within 30 days — significant class imbalance (8:1)
- weight (97%), max_glu_serum (95%), A1Cresult (83%) — dropped due to high missing %
- Dataset uses '?' for missing values — detected and replaced with NaN
- 700+ ICD-9 diagnosis codes simplified into 8 meaningful medical categories
- Circulatory conditions dominate as primary diagnosis (30,458 patients)
- Final dataset: 50 raw columns → 67 ML-ready features after encoding

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn

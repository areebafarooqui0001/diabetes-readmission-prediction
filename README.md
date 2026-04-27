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
| `diabetic_data.csv` | Raw dataset |

## 🔍 Key Findings So Far
- Only 11.2% of patients are readmitted within 30 days — significant class imbalance
- weight (97%), max_glu_serum (95%), A1Cresult (83%) — too many missing, will drop
- Dataset uses '?' for missing values — detected and replaced with NaN

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn
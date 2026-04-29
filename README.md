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
| `03_Model_Building.ipynb` | Model building & evaluation (all models) |
| `diabetic_data.csv` | Raw dataset |
| `diabetic_ml_ready.csv` | ML-ready dataset (101K rows × 67 columns) |

## 🔍 Key Findings
- Only 11.2% of patients are readmitted within 30 days — significant class imbalance (8:1)
- weight (97%), max_glu_serum (95%), A1Cresult (83%) — dropped due to high missing %
- Dataset uses '?' for missing values — detected and replaced with NaN
- 700+ ICD-9 diagnosis codes simplified into 8 meaningful medical categories
- Circulatory conditions dominate as primary diagnosis (30,458 patients)
- Applied SMOTE on training data only — balanced from 8:1 → 1:1 (prevents data leakage)
- Top predictive features: num_lab_procedures, num_medications, time_in_hospital, age
- Key insight: In healthcare, Recall > Precision — missing a high-risk patient is costlier than a false alarm

## 📈 Model Results
| Model | ROC-AUC | F1 Score | Recall |
|-------|---------|----------|--------|
| Logistic Regression | 0.545 | 0.165 | 0.206 |
| Random Forest | 0.599 | 0.112 | 0.089 |

> ⚠️ ROC-AUC is primary metric — accuracy is misleading with 8:1 class imbalance

## ▶️ How to Run
1. Clone the repo: `git clone https://github.com/areebafarooqui0001/diabetes-readmission-prediction`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. Run notebooks in order: `01_EDA` → `02_Feature_Engineering` → `03_Model_Building`

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn
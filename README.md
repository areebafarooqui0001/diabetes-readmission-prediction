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
| `03_Model_Building.ipynb` | Model building, HPT & evaluation (all models) |
| `diabetic_data.csv` | Raw dataset |
| `diabetic_ml_ready.csv` | ML-ready dataset (101K rows × 67 columns) |

## 🔍 Key Findings
- Only 11.2% of patients are readmitted within 30 days — significant class imbalance (8:1)
- weight (97%), max_glu_serum (95%), A1Cresult (83%) — dropped due to high missing %
- Dataset uses '?' for missing values — detected and replaced with NaN
- 700+ ICD-9 diagnosis codes simplified into 8 meaningful medical categories
- Circulatory conditions dominate as primary diagnosis (30,458 patients)
- Applied SMOTE on training data only — balanced from 8:1 → 1:1 (prevents data leakage)
- Top predictive feature: prior inpatient visits (number_inpatient) — strongest readmission signal
- HPT improved Recall by +9.78% — tuned model catches 221 more high-risk patients

## 📈 Model Results
| Model | ROC-AUC | Recall | F1 Score | Accuracy |
|-------|---------|--------|----------|----------|
| Logistic Regression | 0.545 | 0.206 | 0.165 | 76.8% |
| Random Forest | 0.599 | 0.089 | 0.112 | 84.2% |
| XGBoost (Base) | 0.659 | 0.517 | 0.266 | 68.2% |
| **XGBoost (Tuned)** 🏆 | **0.689** | **0.614** | **0.281** | 64.8% |

> ⚠️ ROC-AUC & Recall are primary metrics — accuracy is misleading with 8:1 class imbalance.
> Tuned XGBoost catches 6.9x more high-risk patients than Random Forest.

## ⚙️ Best Hyperparameters (RandomizedSearchCV — 50 iterations, 3-fold CV)
| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| learning_rate | 0.01 |
| max_depth | 6 |
| subsample | 0.6 |
| colsample_bytree | 0.6 |
| scale_pos_weight | 8 |

## ▶️ How to Run
1. Clone the repo: `git clone https://github.com/areebafarooqui0001/diabetes-readmission-prediction`
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. Run notebooks in order: `01_EDA` → `02_Feature_Engineering` → `03_Model_Building`

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Imbalanced-learn
# Disease Outcome Classifier

Predicting diabetes diagnosis from clinical patient data using machine learning.

## Overview
Built a full ML classification pipeline on the Pima Indians Diabetes Database (768 patients, 8 clinical features). Compared three models, tuned the best performer with GridSearchCV, and exported a production-ready sklearn Pipeline.

## Results
| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 75% | 0.82 |
| Decision Tree | 71% | 0.69 |
| Random Forest | 73% | 0.83 |
| **Random Forest (Tuned)** | **76%** | **0.84** |

## Key Findings
- Glucose level was the strongest predictor of diabetes outcome
- Hyperparameter tuning improved ROC-AUC from 0.83 to 0.84
- 21 false negatives highlight the importance of optimizing for recall in clinical settings

## Pipeline
```python
import joblib
pipeline = joblib.load('diabetes_pipeline.pkl')
pipeline.predict(new_patient_data)
```

## Tools
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

## Dataset
[Pima Indians Diabetes Database](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)

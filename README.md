# Credit Card Fraud Detection System

An end-to-end Machine Learning system for detecting fraudulent credit card transactions using supervised learning, business-driven threshold optimization, explainability, and a production-ready FastAPI backend.

---

## Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent a very small fraction of total transactions. This project builds a robust fraud detection pipeline that focuses not only on high accuracy but also on **business-aware decision-making** and **model explainability**.

The system includes:
- Exploratory Data Analysis (EDA)
- Feature preprocessing and scaling
- Multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Threshold optimization for fraud detection
- SHAP-based model explainability
- Real-time prediction API using FastAPI

---

## Dataset

- **Source**: European cardholders dataset (Kaggle)
- **Transactions**: 284,807
- **Fraud cases**: 492 (~0.17%)
- **Features**:
  - `V1`–`V28`: PCA-transformed features
  - `Time`: Transaction time
  - `Amount`: Transaction amount
  - `Class`: Target variable (0 = Legitimate, 1 = Fraud)

> Dataset is **not included** in the repository due to size and privacy constraints.

---

## Exploratory Data Analysis (EDA)

Implemented in: notebooks/01_eda.ipynb
EDA was performed to understand the structure and characteristics of the dataset.

Key insights from EDA:
- Severe class imbalance (fraud < 1%)
- No missing values
- Highly skewed transaction amounts with outliers
- PCA-transformed features require careful evaluation metrics

- Class distribution plot
<img width="566" height="413" alt="image" src="https://github.com/user-attachments/assets/c7ee077d-03d4-46ed-bd3a-6f136fbaa38e" />
 
- Transaction amount distribution
<img width="580" height="421" alt="image" src="https://github.com/user-attachments/assets/a7d1ea6c-608a-4eaa-87e2-eb027088db1b" />

---

##  Data Preprocessing
The preprocessing pipeline ensures data integrity and prevents leakage.

Steps performed:
- Feature–target separation
- Robust scaling applied to the `Amount` feature
- Stratified train–test split to preserve class distribution

Data preprocessing is implemented in: notebooks/02_preprocessing.ipynb

---

## Model Training & Evaluation

Multiple machine learning models were trained and evaluated:

### Models Implemented
- Logistic Regression (baseline)
- Random Forest
- XGBoost (**final selected model**)

### Evaluation Metrics
- ROC-AUC
- Precision, Recall, F1-score
- Confusion Matrix
- Precision–Recall Curve

XGBoost was selected due to its strong balance between fraud recall and precision on highly imbalanced data.

Model training, evaluation, threshold optimization, and SHAP explainability are implemented in: notebooks/03_model_training.ipynb

---

## Threshold Optimization

Instead of using the default classification threshold (0.5), multiple thresholds were evaluated to optimize business trade-offs.

Key considerations:
- Lower thresholds improve fraud recall
- Controlled false positives
- Threshold = **0.3** selected for inference

This approach reflects real-world fraud detection systems, where missing fraudulent transactions is more costly than flagging legitimate ones.

---

## Model Explainability (SHAP)

SHAP (SHapley Additive exPlanations) was used to interpret the XGBoost model.

Benefits:
- Identifies the most influential features driving fraud predictions
- Explains feature impact and direction
- Improves transparency and trust in the model


- SHAP summary plot
  
  <img width="567" height="708" alt="image" src="https://github.com/user-attachments/assets/d8adc58c-baca-4682-a5c1-46e0ab16be00" />

  <img width="574" height="683" alt="image" src="https://github.com/user-attachments/assets/971b80f5-404f-451c-8896-fd729e3059e0" />

  <img width="575" height="143" alt="image" src="https://github.com/user-attachments/assets/383de3cc-a3d6-4551-b4f9-81d8ade039ce" />
  
---

## FastAPI Backend (Local Deployment)

A FastAPI backend was developed to serve real-time fraud predictions locally.

### API Features
- REST-based inference
- Business-optimized decision threshold
- Structured request and response schema
- Interactive Swagger UI

### Run API Locally
```bash
uvicorn api.main:app --reload
```

Swagger UI: 
```http://127.0.0.1:8000/docs ```

Sample API Request: 
``` {
  "Time": 12345,
  "V1": -1.3598,
  "V2": -0.0727,
  "V3": 2.5363,
  "V4": 1.3781,
  "V5": -0.3383,
  "V6": 0.4623,
  "V7": 0.2395,
  "V8": 0.0986,
  "V9": 0.3637,
  "V10": 0.0907,
  "V11": -0.5516,
  "V12": -0.6178,
  "V13": -0.9913,
  "V14": -0.3111,
  "V15": 1.4681,
  "V16": -0.4704,
  "V17": 0.2079,
  "V18": 0.0257,
  "V19": 0.4039,
  "V20": 0.2514,
  "V21": -0.0183,
  "V22": 0.2778,
  "V23": -0.1104,
  "V24": 0.0669,
  "V25": 0.1285,
  "V26": -0.1891,
  "V27": 0.1335,
  "V28": -0.0210,
  "Amount": 149.62
}
 ```

## API Demo (FastAPI + Swagger)

Below is a live prediction made using the deployed FastAPI service via Swagger UI:

<img width="880" height="865" alt="Screenshot 2025-12-30 214827" src="https://github.com/user-attachments/assets/ada8ac10-cfa3-4fe4-b76e-38bf3e8c6862" />

---

## Project Structure

```
Credit-Card-Fraud-Detection/
│
├── api/
│   └── main.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
├── src/
│   ├── loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
├── models/
│   ├── xgboost_fraud_model.pkl
│   └── random_forest_fraud_model.pkl
├── requirements.txt
└── README.md
```

---

## Key Learnings

Handling extreme class imbalance

Business-driven decision threshold optimization

Model interpretability using SHAP

End-to-end ML pipeline design

Serving ML models using FastAPI

---

## Conclusion
This project demonstrates a complete and practical machine learning workflow for fraud detection, combining statistical rigor, business reasoning, explainability, and system-level thinking. It reflects real-world practices used in fintech and risk analytics systems.

---

## Author 
Palak Vastrakar


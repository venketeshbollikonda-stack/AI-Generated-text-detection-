# 🤖 AI vs Human Text Detector  
### Hybrid Ensemble Model (Logistic Regression + XGBoost + CatBoost)  
### Team PrimeV — Major Project  

---

## 📑 Table of Contents
- [Project Overview](#-project-overview)
- [Datasets Used](#-datasets-used)
- [Model Architecture](#-model-architecture)
- [Hybrid Ensemble Logic](#-hybrid-ensemble-logic)
- [Performance](#-performance)
- [Training Pipeline Summary](#-training-pipeline-summary)
- [Saved Model Files](#-saved-model-files)
- [Streamlit Application Features](#-streamlit-application-features)
- [Folder Structure](#-folder-structure)
- [How to Run](#-how-to-run)
- [Team](#-team)

---

## 📌 Project Overview

This project detects whether a given text is **AI-generated** or **Human-written** using a **Hybrid Ensemble Machine Learning Model**.  

Traditional deep learning methods like BERT require large memory/GPU, so we built a lightweight but powerful approach using:

- **TF-IDF Vectorization**  
- **Logistic Regression**  
- **XGBoost Classifier**  
- **CatBoost Classifier (raw text)**  

The combination produces a strong model that is:

✔ Highly Accurate  
✔ Efficient  
✔ Fast to Train  
✔ Fully Explainable using SHAP  
✔ Easy to Deploy with Streamlit  

A complete **Streamlit Web App** is also provided.

---

## 📂 Datasets Used

We used multiple open-source Kaggle datasets to create a robust training set.

### 1. LLM Detect AI-Generated Text  
https://www.kaggle.com/competitions/llm-detect-ai-generated-text

### 2. DAIGT v2 Dataset (Darek Kłeczek)  
https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

### 3. Additional curated datasets:
- `train_drcat_04.csv`  
- `train_v2_drcat_02.csv`  
- `train_essays_RDizzl3_seven_v1.csv`  
- `train_drcat_01.csv`  

All datasets were merged, cleaned, column-normalized, and duplicates were removed.

---

## 🧠 Model Architecture

Our hybrid model uses **three strong classifiers**:

### **1️⃣ Logistic Regression (TF-IDF)**
- Excellent baseline for sparse high-dimensional data  
- Fast and effective  

### **2️⃣ XGBoost Classifier (TF-IDF)**
- Captures non-linear relationships  
- Robust on structured text vector features  
- Supports SHAP explainability  

### **3️⃣ CatBoost Classifier (Raw Text)**
- Handles text natively  
- Learns semantic structure  
- Improves final ensemble generalization  

---

## 🔀 Hybrid Ensemble Logic

The final probability of AI-generated text is:

Final = 0.3 * LogisticRegression
+ 0.3 * XGBoost
+ 0.4 * CatBoost


Weights were selected after comparing ROC-AUC scores for each model.

---

## 📊 Performance

### ✔ Training ROC-AUC: **0.99994**

### ✔ Classification Metrics (Validation)
Accuracy: 98.6%
Precision: 0.99
Recall: 0.99
F1-Score: 0.99


These scores demonstrate that the model is extremely effective for AI text detection using classical ML.

---

## 🧪 Training Pipeline Summary

### Steps performed:

1. Load 4 Kaggle datasets  
2. Normalize column names (`text`, `label`)  
3. Remove duplicate essay entries  
4. Create **TF-IDF features** (50,000 dimensions)  
5. Train:
   - Logistic Regression  
   - XGBoost  
   - CatBoost (raw text)  
6. Build custom `HybridModel` class  
7. Evaluate using ROC-AUC and metrics  
8. Save all models using Joblib  

### Hybrid model scoring formula:

```python
final = (p_cat * 0.4) + (p_xgb * 0.3) + (p_lr * 0.3)
```

🗂 Saved Model Files

Training script automatically saves:
```
model/hybrid_model.pkl  
model/catboost_raw_text.pkl  
model/xgb_tfidf.pkl  
model/logreg_tfidf.pkl  
model/tfidf_vectorizer1.pkl  
```

The Streamlit app loads only:
```
hybrid_model.pkl
tfidf_vectorizer1.pkl
```

🖥 Streamlit Application Features

✔ Single Text Detection

Predict AI or Human
Confidence score (%)
Clean UI with color-coded result boxes
SHAP Explainability for XGBoost

Probability visualization bar chart

✔ Batch CSV Evaluation

Auto-detects text & label columns
Live progress bar
Per-row prediction with AI confidence
Evaluation Metrics:
Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion matrix visualization
Downloadable output CSV

📂 Folder Structure

AI-VS-Human-main/
│
├── app.py
├── model/
│   ├── hybrid_model.pkl
│   ├── catboost_raw_text.pkl
│   ├── xgb_tfidf.pkl
│   ├── logreg_tfidf.pkl
│   └── tfidf_vectorizer1.pkl
│
├── Datasets/
│   ├── train_drcat_04.csv
│   ├── train_v2_drcat_02.csv
│   ├── train_essays_RDizzl3_seven_v1.csv
│   ├── train_drcat_01.csv
│   └── test_essays.csv
│
├── submission_catboost_ensemble.csv
└── README.md


▶ How to Run

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start Streamlit App
streamlit run app.py

3️⃣ Visit in Browser
http://localhost:8501/

👨‍💻 Team

Team Batch_13
Department of Computer Science & Engineering
Hybrid Ensemble + Explainable AI — Final Year Major Project

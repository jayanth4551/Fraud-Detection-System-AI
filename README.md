# 🚨 Fraud Detection System using Machine Learning

An end-to-end Machine Learning project designed to detect fraudulent financial transactions using imbalanced classification techniques, explainable AI, and Streamlit deployment.

---

# 🚀 Live Demo

👉 [View Live App](https://fraud-detection-system-ai.streamlit.app)

---

# 📊 Project Overview

Financial fraud detection is a critical problem in the banking and fintech industry.
This project focuses on identifying fraudulent credit card transactions using Machine Learning techniques on a highly imbalanced dataset.

The system analyzes transaction patterns and predicts whether a transaction is fraudulent or genuine.

---

# 🧠 Machine Learning Workflow

## Steps Performed

* Exploratory Data Analysis (EDA)
* Handling Imbalanced Data using SMOTE
* Feature Scaling using StandardScaler
* Model Training and Comparison
* Threshold Tuning
* SHAP Explainability
* Streamlit Deployment

---

# 🤖 Models Used

| Model                    | Purpose              |
| ------------------------ | -------------------- |
| Logistic Regression      | Baseline model       |
| Random Forest Classifier | Final selected model |

---

# 📈 Evaluation Metrics

Since the dataset is highly imbalanced, accuracy alone is not reliable.

The following metrics were used:

* ROC-AUC
* Precision
* Recall
* F1-Score
* Precision-Recall Analysis

---

# 🔥 Key Insights

* Random Forest achieved the best balance between precision and recall
* Threshold tuning improved fraud detection sensitivity
* Fraud detection depends heavily on latent PCA-transformed features
* SHAP explainability helped interpret model behavior

---

# ⚙️ Techniques Used

## ✅ SMOTE

Handled severe class imbalance by generating synthetic fraud samples.

## ✅ Threshold Tuning

Lowered prediction threshold to improve fraud recall.

## ✅ SHAP Explainability

Used SHAP to identify important features contributing to fraud detection.

---

# 🖥️ Streamlit App Features

* Manual transaction input
* Fraud probability prediction
* Risk level classification
* Real fraud example loader
* Interactive UI

---

# 📊 Sample Screenshots

## 🔹 Application Interface

![App UI](images/app_ui.png)

---

## 🔹 Fraud Prediction Output

![Prediction](images/fraud_prediction.png)

---

## 🔹 SHAP Explainability

![SHAP](images/shap_summary.png)

---

# 🛠️ Tech Stack

## Programming Language

* Python

## Libraries & Tools

* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* SHAP
* Streamlit
* Joblib
* Matplotlib
* Seaborn

---

# 📁 Repository Structure

```text
Fraud-Detection-System-AI/
│
├── app.py
├── fraud_detection.ipynb
├── fraud_model.pkl
├── fraud_model_columns.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
│
├── images/
│   ├── app_ui.png
│   ├── fraud_prediction.png
│   └── shap_summary.png
```

---

# ▶️ How to Run Locally

## Clone Repository

```bash
git clone https://github.com/your-username/Fraud-Detection-System-AI.git
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 📊 Dataset

This project uses the Credit Card Fraud Detection dataset from Kaggle:

🔗 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> Note: Dataset is not included in this repository due to size constraints.

---

# 🚀 Future Improvements

* Add XGBoost and LightGBM models
* Real-time API integration
* Advanced anomaly detection techniques
* Improved dashboard visualizations

---

# 🙌 Acknowledgements

* Kaggle Credit Card Fraud Dataset
* Open-source Python ML ecosystem
* Streamlit Community Cloud

---

# 👨‍💻 Author

**Jayanth**
AI & Data Science Student
Machine Learning & Data Science Enthusiast

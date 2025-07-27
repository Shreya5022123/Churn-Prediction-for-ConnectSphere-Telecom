# Churn-Prediction-for-ConnectSphere-Telecom

Customer churn is a major challenge in the telecom industry. This project uses machine learning to **predict whether a customer is likely to leave** ConnectSphere Telecom, allowing for timely intervention and improved customer retention.

---

## 🧠 Problem Statement

Customer retention is more cost-effective than acquisition. By predicting churn early using customer demographics, service details, and billing history, ConnectSphere can proactively reduce churn and improve business outcomes.

---

## 📂 Dataset Overview

- **Size:** ~7,000 records
- **Target Variable:** `Churn` (Yes = customer left, No = customer stayed)
- **Features Include:**
  - Demographics (Gender, SeniorCitizen, Partner)
  - Services (Internet, Phone, Tech Support, Streaming)
  - Account Info (Tenure, Contract, Payment Method)
  - Billing (MonthlyCharges, TotalCharges)

---

## 🤖 Model Architecture

A simple yet effective **Artificial Neural Network (ANN)** built using TensorFlow/Keras.

- **Input Layer:** All numerical and encoded categorical features
- **Hidden Layers:** 2 Dense layers with ReLU activation
- **Output Layer:** 1 neuron with Sigmoid activation for binary classification
- **Loss Function:** `binary_crossentropy`
- **Optimizer:** `adam`
- **Epochs:** 50
- **Batch Size:** 32

---

## 📈 Results

- **Accuracy:** ~75–80%
- **Recall & Precision:** Balanced (good performance on both churned and retained customers)
- **Evaluation Tools:**
  - Confusion Matrix
  - Accuracy

---

## 🛠️ Technologies Used

- **Language:** Python
- **Libraries:**
  - `Pandas`, `NumPy` – Data processing
  - `Matplotlib`, `Seaborn` – Visualization
  - `Scikit-learn` – Preprocessing and evaluation
  - `TensorFlow`, `Keras` – Deep learning model
- **Platform:** Google Colab

---

## 📊 Workflow Summary

1. **Data Cleaning:**
   - Handle missing values in `TotalCharges`
   - Convert target `Churn` to binary values (Yes → 1, No → 0)

2. **Feature Engineering:**
   - One-hot encode categorical columns
   - Scale numerical features using `StandardScaler`

3. **Model Building & Training:**
   - Design ANN with Keras
   - Compile, train, and validate model
   - Plot accuracy/loss across epochs

4. **Model Evaluation:**
   - Analyze confusion matrix
   - Measure recall and precision

---



## ✅ Getting Started

1. Clone the repo:
```bash
git clone https://github.com/yourusername/connectsphere-churn-prediction.git
cd connectsphere-churn-prediction

1.Install dependencies:
bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow

2.Run the notebook:
Open churn_model.ipynb in Google Colab or Jupyter Notebook

3.Follow the cells step by step


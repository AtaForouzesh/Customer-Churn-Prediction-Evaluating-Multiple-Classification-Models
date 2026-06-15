# Customer-Churn-Prediction-Evaluating-Multiple-Classification-Models
Predicting customer churn using 6 different machine learning classification models in scikit-learn.

## 📊 Dataset
* **Features:** Customer tenure, monthly charges, total charges, etc.
* **Target Variable:** `Churn` (Binary: 0 = No Churn, 1 = Churn)

## 🧠 Models Trained
1. **Logistic Regression** (`Logistic Regression.py`)
2. **K-Nearest Neighbors (KNN)** (`KNN.py`)
3. **Support Vector Classifier (SVC)** (`SVC.py`)
4. **Gaussian Naive Bayes** (`Naive Bayes.py`)
5. **Decision Tree Classifier** (`Decision Tree.py`)
6. **Random Forest Classifier** (`Random Forest.py`)

## ⚙️ Preprocessing & Evaluation
* **Data Splitting:** Data is split into 80% training and 20% testing sets using `train_test_split` with `stratify=y` to maintain class proportions.
* **Scaling:** Features are standardized using `StandardScaler`.
* **Metrics:** Models are evaluated using Confusion Matrices, Accuracy Scores, and detailed Classification Reports (Precision, Recall, F1-Score).

# e-waste-classification-project
 A project on e-waste classification and data analysis.
# 📦 E-Waste Generation Classification

This project uses a machine learning model (Decision Tree Classifier) to classify e-waste items based on their weight and presence of hazardous materials like batteries and lead. It helps raise awareness about e-waste and its classification for better recycling and disposal.

---

## 📁 Files in this Project
- `classification_model.py` – Python code to train and evaluate the model
- `e_waste_data.csv` – Sample dataset with features and target class
- `README.md` – Project explanation with code and results

---

## 🚀 Technologies Used
- Python
- Pandas
- Scikit-learn

---

## 🔍 Model Overview

We use the following features to classify e-waste:
- **Weight**
- **Contains Battery** (Yes/No)
- **Contains Lead** (Yes/No)

The goal is to predict the **Category** (e.g., Small Device, Large Device, Accessory, etc.)

---

## ✅ Code Preview

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('e_waste_data.csv')

# Encode categorical values
df['Contains Battery'] = df['Contains Battery'].map({'Yes': 1, 'No': 0})
df['Contains Lead'] = df['Contains Lead'].map({'Yes': 1, 'No': 0})

# Define features and target
X = df[['Weight', 'Contains Battery', 'Contains Lead']]
y = df['Category']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

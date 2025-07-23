# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID (not useful for training)
df.drop("customerID", axis=1, inplace=True)

# Convert 'TotalCharges' to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].mean(), inplace=True)

# Encode target
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical features
cat_cols = df.select_dtypes(include="object").columns
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

# Features and labels
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Save column names for use in the app
joblib.dump(X.columns.tolist(), "columns.pkl")

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "churn_model.pkl")

print("✅ Model, scaler, and columns saved successfully.")

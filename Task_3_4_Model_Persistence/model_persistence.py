# Task 3.4 - Model Persistence
# ----------------------------

import numpy as np
import pickle
import joblib
import json
import os
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# ------Step 1: Create dataset
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# ------Step 2: Train model
model = LinearRegression()
model.fit(X, y)

# Create folder for saved models
os.makedirs("Task_3_4_Model_Persistence/saved_models", exist_ok=True)

# ------Step 3: Save using pickle
with open("Task_3_4_Model_Persistence/saved_models/model.pkl", "wb") as f:
    pickle.dump(model, f)

# ------Step 4: Save using joblib
joblib.dump(model, "saved_models/model.joblib")

# ------Step 5: Save weights as JSON
weights = {
    "coef": model.coef_.tolist(),
    "intercept": model.intercept_
}

with open("saved_models/weights.json", "w") as f:
    json.dump(weights, f)

print("Models saved successfully.")

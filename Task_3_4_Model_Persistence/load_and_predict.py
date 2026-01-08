# ------------ Step 3.4.6: Create load_and_predict.py
# This script loads saved models and makes predictions on new data

import pickle
import joblib
import json
import time
import os
import numpy as np
from sklearn.linear_model import LinearRegression


# ------------ Step 3.4.7: Load each model format

# New unseen data for prediction
X_new = np.array([[5], [10], [15]])

results = []


# ------------ Step 3.4.8: Make predictions with Pickle model

start = time.time()
with open("saved_models/model.pkl", "rb") as f:
    model_pickle = pickle.load(f)
pickle_load_time = time.time() - start

pickle_predictions = model_pickle.predict(X_new)
pickle_size = os.path.getsize("saved_models/model.pkl")

results.append(["Pickle", pickle_size, pickle_load_time, pickle_predictions.tolist()])


# ------------ Step 3.4.8 (continued): Make predictions with Joblib model

start = time.time()
model_joblib = joblib.load("saved_models/model.joblib")
joblib_load_time = time.time() - start

joblib_predictions = model_joblib.predict(X_new)
joblib_size = os.path.getsize("saved_models/model.joblib")

results.append(["Joblib", joblib_size, joblib_load_time, joblib_predictions.tolist()])


# ------------ Step 3.4.8 (continued): Make predictions with JSON weights

start = time.time()
with open("saved_models/weights.json", "r") as f:
    weights = json.load(f)
json_load_time = time.time() - start

# Manually rebuild model using weights
model_json = LinearRegression()
model_json.coef_ = np.array(weights["coef"])
model_json.intercept_ = weights["intercept"]

json_predictions = model_json.predict(X_new)
json_size = os.path.getsize("saved_models/weights.json")

results.append(["JSON", json_size, json_load_time, json_predictions.tolist()])


# ------------ Step 3.4.9: Measure file sizes using os.path.getsize()
# (Already measured above for each format)


# ------------ Step 3.4.10: Time loading with time module
# (Already measured using time.time())


# Print final comparison results
for r in results:
    print(r)

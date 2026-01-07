# Task 3.2 Multiple Linear Regression with sklearn

# ---------Step 3.2.1 Import Libraries
# Already Installed


# ---------Step 3.2.2 Import Libraries
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
X = housing.data  # Features (e.g., Median Inc, HouseAge, etc.)
y = housing.target  # Target (house prices)



# ---------Step 3.2.3 Split Data into Training and Testing Sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# ---------Step 3.2.4 & 5 import linear regression from sklearn

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# ---------Step 3.2.6 Predict y

y_pred = model.predict(X_test)

# ---------Step 3.2.7 Calculate MAE, MSE, RMSE, R² using sklearn.metrics

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np  # RMSE ke liye

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# ---------Step 3.2.8 Create Scatter Plot (Actual vs Predicted)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.savefig('Task_3_2_Multiple_Linear_regression_sklearn\plots/actual_vs_predicted.png') 
plt.show()

# ---------- Step 3.2.9 Plot Residuals

residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('Task_3_2_Multiple_Linear_regression_sklearn\plots/residuals.png')
plt.show()

# ------------ Step 3.2.10 Print Feature Coefficients

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# ------------ Step 3.2.11 Save the Model using joblib

import json
weights = {'coef': model.coef_.tolist(), 'intercept': float(model.intercept_)}
with open('weights.json', 'w') as f:
    json.dump(weights, f)
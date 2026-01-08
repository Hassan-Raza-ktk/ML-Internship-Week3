# Week 3 Task 3.3 - Polynomial Regression & Overfitting by Hassan Raza

# ----Step 3.3.1 Create Synthetic Data----
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X**3 - X**2 + 2 + np.random.randn(100, 1)

# ----Step 3.3.2 Polynomial Features from sklearn----

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

degrees = [1, 2, 3, 5, 10]

train_errors = []
test_errors = []

X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='black', alpha=0.3, label="Data")


# --Step 3.3.3 to 6 Fit Polynomial Regression Models and Plot----

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    X_plot_poly = poly.transform(X_plot)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_mse)
    test_errors.append(test_mse)

    y_plot = model.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, label=f"Degree {d}")


# --- Step 3.3.7 Plot all models together ----

plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression Models Comparison")
plt.legend()
plt.savefig(r"C:\Users\razak\OneDrive\Desktop\Neuro_App\week3\Task_3_3_Polynomial_Regression\plots\learning_curves.png")
plt.close()


# --- Step 3.3.8 Showing errors for degrees ----

import pandas as pd

error_table = pd.DataFrame({
    "Degree": degrees,
    "Train MSE": train_errors,
    "Test MSE": test_errors
})

print(error_table)


# --- Step 3.3.9 Learning Curves ----

plt.figure(figsize=(8, 6))
plt.plot(degrees, train_errors, marker='o', label="Train Error")
plt.plot(degrees, test_errors, marker='o', label="Test Error")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE")
plt.title("Training vs Testing Error")
plt.legend()
plt.savefig(r"C:\Users\razak\OneDrive\Desktop\Neuro_App\week3\Task_3_3_Polynomial_Regression\plots\learning_curves.png")
plt.close()

# --- Step 3.3.10 Documentation ----

# Overfitting Observation:
# Higher-degree polynomial models reduce training error but increase test error,
# indicating that the model starts fitting noise rather than the true data pattern.
# Degree 3 provides the best bias-variance tradeoff for this dataset.

# The Output

#    Degree  Train MSE   Test MSE
# 0       1  10.415755  16.352830
# 1       2   4.259944   5.266638
# 2       3   0.816024   0.602247 Best Degree
# 3       5   0.797184   0.649126
# 4      10   0.749306   0.817281
# week3 machine learning by Hassan Raza

import numpy as np
import matplotlib.pyplot as plt

# ------------Step 3.1.1 Create a synthetic dataset

# Fix seed so results are repeatable
np.random.seed(42)

# Create a synthetic dataset following y = 2x + 1 + noise
# Noise is added to simulate real world imperfect data

X = np.linspace(0, 10, 50)   # 50 points between 0 and 10
noise = np.random.randn(50) # random noise
y = 2 * X + 1 + noise       # true relationship (hidden from model)


plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Synthetic Dataset")
plt.savefig("plots/syntheticdata_plot.png")
plt.close()

# ------------Step 3.1.2 Define Linear Regression Class

class LinearRegression:
    """
    Simple Linear Regression implemented from scratch
    using gradient descent optimization.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        
# ------------Step 3.1.3 Initialize weights and bias
        
        self.w = 0
        self.b = 0
        self.costs = []

# ------------Step 3.1.4 Implement Cost Function

    def compute_cost(self, y_true, y_pred):
    # Mean Squared Error (MSE)
    # Measures the average squared difference between predictions and actual values
    # Squaring penalizes larger errors more heavily
        m = len(y_true)
        cost = (1 / (2 * m)) * np.sum((y_pred - y_true) ** 2)
        return cost

# ------------Step 3.1.5 Implement Fit Method

    def fit(self, X, y):
    # Gradient descent algorithm
    # Iteratively updates weight and bias to minimize the cost function
        m = len(y)

        for i in range(self.n_iterations):

            # Prediction
            y_pred = self.w * X + self.b

        # These indicate the direction of steepest increase in error
        # Compute gradients of the cost function
            dw = (1 / m) * np.sum((y_pred - y) * X)
            db = (1 / m) * np.sum(y_pred - y)

        # Update parameters in the opposite direction of the gradient
        # This moves the model toward lower error
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            # Compute and store cost
            cost = self.compute_cost(y, y_pred)
            self.costs.append(cost)

# ------------Step 3.1.6  predict Method

    def predict(self, X):
    # Predict output values using the learned linear equation
        return self.w * X + self.b

# ------------Step 3.1.7 Calculate R² Score

def r2_score(y_true, y_pred):
# R² score measures how well the model explains the variance in the data
# R² = 1 indicates perfect prediction, 0 indicates no improvement over mean

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# --------- Create and train model (REQUIRED before R²)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("R² Score:", r2)

# ------------ Step 3.1.8 Plot data points and regression line

# Plot actual data points and the learned regression line

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Data Points and Regression Line")
plt.legend()
plt.savefig("plots/linear_regression_plot.png")
plt.close()

# ------------ Step 3.1.9 Plot cost function vs iterations

# Plot cost versus iterations to visualize model convergence

plt.plot(model.costs)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.savefig("plots/cost_convergence.png")
plt.close()
ML Internship – Week 3
Regression Models, Overfitting & Model Persistence
This repository contains Week 3 tasks of my Machine Learning Internship.
The focus of this week is to deeply understand regression models, their behavior, common pitfalls like overfitting, and how trained models are saved and reused in real-world systems.

Overview of Tasks
Week 3 covers the following major concepts:
    • Linear Regression from scratch (without sklearn)
    • Multiple Linear Regression using sklearn
    • Polynomial Regression and Overfitting analysis
    • Model Persistence (Saving & Loading models)
Each task includes:
    • Clean Python implementation
    • Evaluation metrics
    • Visualizations
    • Conceptual understanding

Task 3.1: Linear Regression from Scratch (Gradient Descent)
Objective
To understand how linear regression works internally, without using any machine learning library.
What was implemented
    • Synthetic dataset generation using NumPy
    • Custom LinearRegression class
    • Manual implementation of:
        ◦ Weight and bias initialization
        ◦ Mean Squared Error (MSE) cost function
        ◦ Gradient Descent optimization
    • Manual calculation of R² score
    • Visualization of results
Key Outputs
    • Data points vs regression line
    • Cost (MSE) vs iterations graph
Visualizations
<img src="Task_3_1_Linear_Regression_From_Scratch/plots/linear_regression_plot.png" width="400"> 
<img src="Task_3_1_Linear_Regression_From_Scratch/plots/cost_convergence.png" width="400"> 
Learning Outcome
This task helped in understanding:
    • How gradient descent updates parameters
    • Why cost decreases over iterations
    • How a model learns from data mathematically

Task 3.2: Multiple Linear Regression using sklearn
Objective
To apply regression on real-world multi-feature data using an industry-standard library.
Dataset
    • California Housing Dataset (sklearn)
What was implemented
    • Data loading using sklearn
    • Train-test split
    • Model training using LinearRegression
    • Model evaluation using:
        ◦ MAE
        ◦ MSE
        ◦ RMSE
        ◦ R² Score
    • Visualization of predictions and residuals
    • Saving model weights as JSON
Visualizations
<img src="Task_3_2_Multiple_Linear_Regression_sklearn/plots/actual_vs_predicted.png" width="400"> 
<img src="Task_3_2_Multiple_Linear_Regression_sklearn/plots/residuals.png" width="400"> 
Learning Outcome
This task demonstrated:
    • How multiple features affect predictions
    • Why R² is often lower on real datasets
    • How residual plots help diagnose model behavior

Task 3.3: Polynomial Regression & Overfitting
Objective
To understand model complexity, underfitting, overfitting, and the bias–variance tradeoff.
What was implemented
    • Synthetic non-linear dataset
    • Polynomial regression models with degrees:
        ◦ 1, 2, 3, 5, and 10
    • Training and testing error comparison
    • Learning curves
    • Visual comparison of all models
Visualizations
<img src="Task_3_3_Polynomial_Regression/plots/models_comparison.png" width="400"> 
<img src="Task_3_3_Polynomial_Regression/plots/learning_curves.png" width="400"> 
Overfitting Analysis
    • Low-degree models underfit the data
    • High-degree models achieve very low training error but higher test error
    • Degree 3 provides the best balance between bias and variance
Learning Outcome
This task clearly demonstrated:
    • Why more complex models are not always better
    • How overfitting can be detected using metrics and plots
    • The importance of generalization

Task 3.4: Model Persistence – Saving & Loading Models
Objective
To learn how trained models are saved, loaded, and reused in production systems.
What was implemented
    • Training a regression model
    • Saving the model in three formats:
        ◦ Pickle (.pkl)
        ◦ Joblib (.joblib)
        ◦ JSON (weights only)
    • Loading each format and making predictions
    • Measuring:
        ◦ File sizes
        ◦ Model loading times
    • Comparing all formats
Model Persistence Comparison
Format
File Size (bytes)
Load Time (sec)
Notes
Pickle
(434)
(0.0)
Python-native
Joblib
(576)
(0.0)
Optimized for ML
JSON
(62)
(0.0)
Stores only Weights
Conclusion
    • Joblib offers the best balance of speed and usability for ML models
    • JSON is lightweight and suitable for cross-platform deployment
    • Pickle is simple but less flexible for production use

Tools & Libraries Used
    • Python
    • NumPy
    • Matplotlib
    • scikit-learn
    • Pickle, Joblib, JSON

Final Learning Summary
By completing Week 3, I gained a strong understanding of:
    • How regression models work internally
    • How to evaluate and visualize model performance
    • How overfitting occurs and how to detect it
    • How trained models are saved and reused in real applications
This week focused not only on implementation, but on thinking like a machine learning engineer.

Author
Hassan Raza
Machine Learning Intern

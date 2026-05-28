# Linear Regression from Scratch
This is an implementation of Linear Regression without sklearn.

## What This Project Does
Predicts salary based on years of experience using:
- Cost function (MSE)
- Gradient computation for both weight and bias
- Gradient Descent optimization loop
- Visualization of the regression line over the data

## How It Works

### Cost Function
Measures how wrong the model is:
```
J(w,b) = (1/2m) * Σ(f(xᵢ) - yᵢ)²
```
It's the difference between the model's prediction and actual value.
Square is used to:
- Penalize large errors and encourage the model to correct significant mistakes.
- Make the errors always positive to prevent negative errors from canceling positive errors.

### Gradient Descent
Iteratively updates weights to minimize the cost:
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```
The goal is to reach the optimal weight and bias that make the cost as low as possible.

## Dataset
**Salary_Data.csv** — contains two columns:
- `YearsExperience` — input feature
- `Salary` — target variable

## How To Run
```bash
pip install numpy pandas matplotlib
python linear_regression_scratch.py
```

## Results

![Regression Line](https://github.com/user-attachments/assets/b2a89516-88e3-4211-97a0-24360350fbd8)

The model fits a clean regression line through the salary data, converging after 10,000 iterations with learning rate α=0.01.

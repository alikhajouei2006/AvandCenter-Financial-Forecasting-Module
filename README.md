# AvandCenter Financial Forecasting Module

A machine learning component of the AvandCenter project for forecasting users' daily net balance based on historical transaction data.

This project focuses on designing a complete forecasting pipeline, including data processing, feature engineering, time-series validation, model evaluation, forecasting, and retraining strategies.

## Overview

Financial behavior is highly dynamic and difficult to predict due to irregular spending patterns, unexpected expenses, and limited historical information.

The goal of this project is to build a forecasting module capable of estimating future daily net balance:

**Net Balance = Income - Expense**

The system predicts the next seven days using a recursive multi-step forecasting strategy and evaluates different machine learning models to select the most reliable approach.

---

## Key Features

- Time-series based financial forecasting
- Leakage-safe feature engineering pipeline
- Calendar and historical transaction features
- Time-aware model validation using `TimeSeriesSplit`
- Hyperparameter optimization using `RandomizedSearchCV`
- Comparison of multiple regression models
- Recursive 7-day forecasting engine
- Automated retraining strategy
- Modular system architecture

---

## System Architecture

The project follows a modular architecture where each component has a specific responsibility.

```
Financial Transactions
          |
          v
User Eligibility Checking
          |
          v
Feature Engineering
          |
          v
Preprocessing Pipeline
          |
          v
Model Training & Optimization
          |
          v
Forecasting Engine
          |
          v
7-Day Net Balance Prediction
          |
          v
Retraining Manager
```

---

## Project Structure

```
Financial-Forecasting/
│
├── Eligibility/
│   └── eligibility.py
│
├── Feature_Engineering/
│   ├── EDA.py
│   ├── pipelines.py
│   └── x_y_creation.py
│
├── Training/
│   └── model_opt.py
│
├── Prediction/
│   └── predict.py
│
├── Retraining/
│   └── retrain.py
│
├── Documentation/
│   └── Technical_Documentation.pdf
│
└── README.md
```

---

## Feature Engineering

The model uses both calendar-based and historical features.

### Calendar Features

| Feature     | Description                           |
| ----------- | ------------------------------------- |
| DayOfWeek   | Captures weekly patterns              |
| IsWeekend   | Identifies weekend behavior           |
| DayOfMonth  | Captures monthly recurring events     |
| Month       | Captures seasonal patterns            |
| WeekOfMonth | Captures within-month weekly patterns |

### Lag Features

| Feature              | Description                         |
| -------------------- | ----------------------------------- |
| NetBalance_lag1      | Previous day's net balance          |
| NetBalance_lag7      | Net balance from seven days earlier |
| NetBalance_rollmean7 | 7-day rolling average               |
| NetBalance_rollstd7  | 7-day rolling standard deviation    |

The preprocessing pipeline ensures that feature generation and scaling are performed without future information leakage.

---

## Models Evaluated

The following regression models were evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees

Models were evaluated using:

- R²
- MAE
- RMSE
- Cross-validation standard deviation
- Learning curve analysis

---

## Validation Strategy

Because financial data is time-dependent, standard random cross-validation is not suitable.

The project uses:

```
TimeSeriesSplit
n_splits = 5
```

This approach preserves chronological order and prevents future information from being used during training.

Hyperparameter optimization was performed using:

```
RandomizedSearchCV
n_iter = 25
scoring = neg_RMSE
```

---

## Final Model Selection

Random Forest was selected as the final forecasting model.

Reasons:

- Highest test-set R² among evaluated models
- Lowest learning curve gap
- Strong generalization ability
- Low cross-validation variability
- Lower risk of overfitting compared with more complex models

Although Gradient Boosting achieved slightly better MAE and CV RMSE mean, Random Forest provided a better balance between accuracy and stability.

---

## Forecasting Strategy

The forecasting engine uses a recursive multi-step approach.

Example:

```
Day t historical data
        |
        v
Predict Day t+1
        |
        v
Use prediction as input
        |
        v
Predict Day t+2
        |
        ...
        |
        v
Predict Day t+7
```

The final output contains:

- Seven predicted daily net balances
- Sum of predicted values

A known limitation of recursive forecasting is error propagation, where prediction errors may accumulate over longer forecast horizons.

---

## Retraining Strategy

The model is retrained only when sufficient new information becomes available.

Retraining conditions:

| Condition        | Threshold                               |
| ---------------- | --------------------------------------- |
| New Transactions | At least 10 new transactions            |
| Time Interval    | At least 7 days since previous training |

This prevents unnecessary retraining while allowing the model to adapt to new financial behavior.

---

## Limitations

### Limited Feature Space

The model relies mainly on transaction history and calendar information.

External variables such as:

- Inflation rate
- Holiday events
- Salary cycles

are not included.

### Synthetic Dataset

The model was trained and evaluated using synthetic financial data.

Although statistical analysis suggests that the generated data captures realistic patterns, real-world performance requires validation using actual user transaction data.

### Financial Behavior Uncertainty

Human financial behavior contains unpredictable events such as unexpected expenses and irregular purchases.

Weak temporal correlation in the data limits the achievable forecasting accuracy.

### Recursive Forecasting Error

Prediction errors may accumulate across multiple forecasting steps, reducing accuracy for later forecast horizons.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Machine Learning Regression Models

---

## Documentation

A complete technical documentation file is available:

```
Documentation/Technical_Documentation.pdf
```

The documentation includes:

- Dataset analysis
- Exploratory data analysis
- Feature engineering details
- Model evaluation
- Forecasting methodology
- Retraining strategy
- System limitations

---

## Future Improvements

Potential improvements include:

- Training on real user transaction data
- Adding external economic features
- Exploring advanced time-series models
- Incorporating user-specific behavior modeling
- Improving uncertainty estimation for predictions
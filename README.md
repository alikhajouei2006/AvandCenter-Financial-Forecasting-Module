# AvandCenter Financial Forecasting Module

A machine learning module for the AvandCenter project. It forecasts a user's daily net balance (income minus expense) for the next seven days from their transaction history.

The module was trained and tested on synthetic transaction data. The best model explains about 17% of the variance in daily net balance on the test set (R² = 0.174), so treat the numbers as a test of the evaluation setup, not as evidence of real-world accuracy. The results and limitations sections give the details.

## Contents

- [Overview](#overview)
- [Data](#data)
- [Installation](#installation)
- [Project structure](#project-structure)
- [Pipeline](#pipeline)
- [Features](#features)
- [Models and validation](#models-and-validation)
- [Results](#results)
- [Final model](#final-model)
- [Forecasting](#forecasting)
- [Retraining](#retraining)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Documentation](#documentation)

## Overview

Daily spending is irregular, and a new user has little history to learn from. This module predicts

```
Net Balance = Income - Expense
```

for each of the next seven days. It does this by predicting one day at a time and feeding each prediction back in as input to the next step (recursive multi-step forecasting). Several regression models were compared using time-series cross-validation.

## Data

The data is synthetic. It was generated to look like everyday purchases, recurring payments such as rent, and random one-off events. No real user data is used or included.

The data file is in `data/`. Checks on its distribution, autocorrelation and variability suggest it is a reasonable approximation of real behavior, but it cannot fully reproduce it.

## Installation

```bash
git clone https://github.com/alikhajouei2006/AvandCenter-Financial-Forecasting-Module
cd AvandCenter-Financial-Forecasting-Module
pip install -r requirements.txt
```

## Project structure

```
AvandCenter-Financial-Forecasting-Module/
├── Eligibility/
│   └── eligibility.py
├── Feature_Engineering/
│   ├── EDA.py
│   ├── pipelines.py
│   └── x_y_creation.py
├── Training/
│   └── model_opt.py
├── Prediction/
│   └── predict.py
├── Retraining/
│   └── retrain.py
├── Documentation/
│   └── Technical_Documentation.pdf
├── figures/                  # Plots from the technical documentation
├── data/                     # Synthetic transaction data
├── requirements.txt
├── LICENSE
└── README.md
```

## Pipeline

Each stage is a separate module:

```
Financial transactions
        |
Eligibility check            Eligibility/eligibility.py
        |
Feature engineering          Feature_Engineering/EDA.py, x_y_creation.py
        |
Preprocessing pipeline       Feature_Engineering/pipelines.py
        |
Training and tuning          Training/model_opt.py
        |
Forecasting (7 days)         Prediction/predict.py
        |
Retraining check             Retraining/retrain.py
```

A user must have at least 14 days of activity and 20 recorded transactions before a forecast is made.

## Features

There are nine features: five from the calendar and four from past net balance.

| Feature | Description |
| ------- | ----------- |
| DayOfWeek | Day of the week (0-6) |
| IsWeekend | 1 on Saturday and Sunday |
| DayOfMonth | Day of the month (1-31), for salary and rent dates |
| Month | Month (1-12) |
| WeekOfMonth | Week within the month (1-5) |
| NetBalance_lag1 | Net balance of the previous day |
| NetBalance_lag7 | Net balance seven days earlier |
| NetBalance_rollmean7 | 7-day rolling mean |
| NetBalance_rollstd7 | 7-day rolling standard deviation |

All lag and rolling features are shifted by one day so that no future value reaches the model. The preprocessing pipeline builds the date features, then the lag features, then scales everything with `StandardScaler`, so the scaler is fitted on training data only.

## Models and validation

Linear Regression, Ridge and Lasso were used as baselines. Decision Tree, Random Forest, Gradient Boosting and Extra Trees were then tuned and compared.

Because the data is a time series, random cross-validation would let the model see the future. The project uses `TimeSeriesSplit` with 5 splits. Hyperparameters were tuned with `RandomizedSearchCV` (`n_iter = 25`, scoring `neg_RMSE`). Models were compared on R², MAE, RMSE, cross-validation standard deviation and learning curves.

## Results

All numbers come from the synthetic data.

Baseline models:

| Model | R² train | R² test | MAE test | RMSE test |
| ----- | -------- | ------- | -------- | --------- |
| Linear Regression | 0.1375 | 0.0339 | 7521.98 | 13143.27 |
| Ridge | 0.1375 | 0.0350 | 7511.33 | 13136.27 |
| Lasso | 0.1375 | 0.0340 | 7521.49 | 13142.98 |

Tuned tree-based models:

| Metric | Decision Tree | Random Forest | Gradient Boosting | Extra Trees |
| ------ | ------------- | ------------- | ----------------- | ----------- |
| R² test (%) | 10.40 | 17.36 | 14.05 | 16.08 |
| MAE test | 6269.9 | 5884.3 | 5454.0 | 5713.6 |
| RMSE test | 12658.0 | 12155.9 | 12397.2 | 12250.2 |
| CV RMSE std | 2674.31 | 1564.51 | 1564.51 | 1627.66 |
| Learning-curve gap | 5423.82 | 1053.65 | 3287.02 | 2075.05 |
| CV RMSE mean | 13159.5 | 12743.4 | 12715.8 | 12579.8 |

Random Forest is best on R², RMSE, CV standard deviation (tied with Gradient Boosting) and learning-curve gap. Gradient Boosting has the lowest MAE, and Extra Trees the lowest mean CV RMSE.

The low R² is mostly a property of the data. Autocorrelation is weak (only lag 1 is slightly significant), and the series is dominated by a few large spikes of up to ±60,000 from salary and rent, which keep RMSE high.

<p align="center">
  <img src="figures/test-r2-comparison.png" width="32%" alt="Test-set R² of the tree-based models">
  <img src="figures/test-mae-rmse-comparison.png" width="32%" alt="Test-set MAE and RMSE of the tree-based models">
  <img src="figures/learning-curve-gap.png" width="32%" alt="Learning-curve gap of the tree-based models">
</p>

The other figures (time series, histogram, autocorrelation, learning curves, CV variability) are in [`figures/`](figures/) and in the technical documentation.

## Final model

Random Forest was chosen. The selection rule was the lowest sum of learning-curve gap and cross-validation standard deviation, which weighs overfitting and stability together. Random Forest also had the highest test R² and the lowest test RMSE, and it trains and predicts quickly, which helps with periodic retraining.

Gradient Boosting was slightly better on MAE and Extra Trees on mean CV RMSE, but the differences were small compared with the gap in overfitting.

## Forecasting

The model predicts day t+1, appends that prediction to the history, rebuilds the features, and predicts t+2. This repeats until day t+7. The output is the seven predicted daily values and their sum.

Errors build up over the steps, so later days are less accurate than earlier ones.

## Retraining

The model is retrained only when both conditions hold:

| Condition | Threshold |
| --------- | --------- |
| New transactions | At least 10 since the last training |
| Time since last training | At least 7 days |

This avoids retraining on a handful of new records.

## Limitations

- **Synthetic data.** Real-world performance is unknown and needs checking on actual user transactions.
- **Weak signal.** Daily spending is close to random, and the features are only calendar information and past net balance. Inflation, holidays and salary cycles are not included.
- **Recursive forecasting.** Errors accumulate over the seven days.
- **New users.** A user who has just met the minimum activity has little history, so early forecasts are less reliable.

## Future work

- Test on real transaction data.
- Add external features such as holidays and salary dates.
- Try time-series models built for this setting.
- Model each user's own behavior.
- Add prediction intervals.

## Documentation

`Documentation/Technical_Documentation.pdf` covers the data analysis, feature engineering, model evaluation, forecasting method, retraining rules and limitations in more detail.

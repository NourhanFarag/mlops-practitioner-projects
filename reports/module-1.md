# Module 1 Report

## Baseline Metrics

- Validation MAE: 4.22 minutes
- Validation RMSE: 6.51 minutes

## Baseline Model

The initial baseline is implemented in:

`notebooks/00-baseline.ipynb`

The notebook represents the "before" version of the project before the
workflow is refactored into reusable Python modules.

The baseline model uses:

- `PU_DO` as a categorical feature
- `trip_distance` as a numerical feature
- `duration` in minutes as the target
- `DictVectorizer` for categorical encoding
- `LinearRegression` as the regression model

## Baseline Data Preparation

The dataset was cleaned by keeping:

- trip durations between 1 and 60 minutes
- trip distances greater than 0 and no more than 50 miles

After cleaning, 37,533 trip records remained.

## Baseline Observations

The linear regression baseline outperformed a median dummy predictor:

- Dummy MAE: 7.30 minutes
- Dummy RMSE: 10.56 minutes
- Linear Regression MAE: 4.22 minutes
- Linear Regression RMSE: 6.51 minutes

Training error was lower than validation error:

- Training MAE: 3.03 minutes
- Validation MAE: 4.22 minutes
- Training RMSE: 4.65 minutes
- Validation RMSE: 6.51 minutes

The model also produced a small number of predictions above the
60-minute target-cleaning boundary. These limitations are kept as part
of the baseline and can be revisited in later iterations.

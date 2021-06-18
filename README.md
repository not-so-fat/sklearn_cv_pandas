# sklearn-cv-pandas
RandomizedSearchCV/GridSearchCV with pandas.DataFrame interface

## Why do I want this?

- I usually prepare features as pandas.DataFrame
- Scikit learn input should be `array-like`. https://scikit-learn.org/stable/glossary.html#term-array-like.
- Although it includes pandas.DataFrame, there are some issues;
    - It does not support `Int64` data type
    - Output model does not remember which columns should be used

## Solution

- Provide GridSearchCV / RandomizedSearchCV with pandas.DataFrame interface
    - Internally preprocess DataFrame to be applicable for sklearn
- Output of `fit` command is now original `Model` object, which
    - stores column name information
    - provides pandas.DataFrame interface for prediction


## Installation
```
pip install sklearn_cv_pandas
```

## Usage

### Configure CV object

Instantiate CV in the same manner as original ones.
```
from scipy import stats
from sklearn import linear_model
from sklearn_cv_pandas import RandomizedSearchCV

estimator = linear_model.Lasso()
param_dist = dict(alpha=stats.loguniform(1e-5, 10))
cv = RandomizedSearchCV(estimator, param_dist, scoring="mean_absolute_error")
```

### `fit` with pandas.DataFrame

Our CV object has new methods `fit_holdout_pandas` and `fit_cv_pandas`.
Original ones requires `x` and `y` as `numpy.array`.
Instead of numpy array, you can specify one `pandas.DataFrame` 
and column names for `x` (`feature_columns`), and column name of `y` (`target_column`).
```
model = cv.fit_cv_pandas(
    df, target_column="y", feature_columns=["x{}".format(i) for i in range(100)], n_fold=5
)
```

### `predict` with pandas.DataFrame

You can run prediction with pandas.DataFrame interface as well.
Output of `fit_holdout_pandas` and `fit_cv_pandas` stores `feature_columns` and `target_column`.
You can just input `pandas.DataFrame` for prediction into the method `predict`.

```
model.predict(df)
```

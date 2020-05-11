# sklearn-cv-pandas
RandomizedSearchCV/GridSearchCV with pandas.DataFrame interface

## Installation
```
pip install sklearn_cv_pandas
```

## Usage

To tune hyper parameters, instantiate CV as the same as original ones, and use methods `fit_sv_pandas` or `fit_cv_pandas`
```
from scipy import stats
from sklearn import linear_model
from sklearn_cv_pandas import RandomizedSearchCV

estimator = linear_model.Lasso()
param_dist = dict(alpha=stats.loguniform(1e-5, 10))
cv = RandomizedSearchCV(estimator, param_dist, scoring="mean_absolute_error")
model = cv.fit_cv_pandas(
    df, target_column="y", feature_columns=["x{}".format(i) for i in range(100)], n_fold=5
)
```

To make prediction, use method `predict` of the output of `fit_sv_pandas` or `fit_cv_pandas`

```
model.predict(df)
```
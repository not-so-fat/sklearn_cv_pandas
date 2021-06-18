import pytest

from scipy import stats
from sklearn import (
    linear_model,
    tree,
    pipeline,
    impute,
    preprocessing
)

from sklearn_cv_pandas import (
    RandomizedSearchCV,
    GridSearchCV
)
from tests import utils


def execute_scenario(model_type, is_cl, with_prep, cv_type, holdout_type):
    cv = _get_cv(model_type, is_cl, with_prep, cv_type)
    df_training = utils.get_input_df(100, with_prep)
    df_validation = utils.get_input_df(100, with_prep)
    df_test = utils.get_input_df(10, with_prep)
    target_column = "target_cl" if is_cl else "target_rg"
    feature_columns = ["column{}".format(i) for i in range(6)]
    model = cv.fit_cv_pandas(df_training, target_column, feature_columns, n_fold=3) \
        if holdout_type == "cv" \
        else cv.fit_holdout_pandas(df_training, target_column, feature_columns, ratio_training=0.8) \
        if holdout_type == "holdout_ratio" \
        else cv.fit_holdout_pandas(df_training, target_column, feature_columns, df_validation)
    _assert_prediction(model, df_test, is_cl)


def test_random_linear_holdout_ratio_cl():
    execute_scenario("linear", True, False, "random", "holdout_ratio")


def test_random_tree_holdout_2dfs_rg():
    execute_scenario("tree", False, False, "random", "holdout_2dfs")


def test_random_tree_with_prep_cv_cl():
    execute_scenario("tree", True, True, "random", "cv")


def test_grid_linear_holdout_ratio_cl():
    execute_scenario("linear", True, False, "grid", "holdout_ratio")


def test_grid_tree_with_prep_holdout_2dfs_rg():
    execute_scenario("tree", False, True, "grid", "holdout_2dfs")


def test_grid_tree_cv_cl():
    execute_scenario("tree", True, False, "grid", "cv")


def _get_cv(model_type, is_cl, with_prep, cv_type):
    estimator = _get_estimator(model_type, is_cl, with_prep)
    metric = "roc_auc" if is_cl else "neg_root_mean_squared_error"
    if cv_type == "random":
        params = _get_params_random(model_type, is_cl, with_prep)
        return RandomizedSearchCV(estimator, params, scoring=metric)
    else:
        params = _get_params_grid(model_type, is_cl, with_prep)
        return GridSearchCV(estimator, params, scoring=metric)


def _get_estimator(model_type, is_cl, with_preprocessing):
    if model_type == "linear":
        ml_estimator = linear_model.LogisticRegression(solver="liblinear") if is_cl else linear_model.Lasso()
    else:
        ml_estimator = tree.DecisionTreeClassifier() if is_cl else tree.DecisionTreeRegressor()
    return _add_preprocessing(ml_estimator) if with_preprocessing else ml_estimator


def _add_preprocessing(estimator):
    return pipeline.Pipeline(
        steps=[
            ("mvi", impute.SimpleImputer()),
            ("std", preprocessing.StandardScaler()),
            ("ml", estimator)
        ]
    )


def _get_params_random(model_type, is_cl, with_preprocessing):
    if model_type == "linear":
        ml_params = dict(
            penalty=["l1", "l2"],
            C=stats.loguniform(1e-5, 10)
        ) if is_cl else dict(alpha=stats.loguniform(1e-5, 10))
    else:
        ml_params = dict(max_depth=list(range(5, 16)))
    return _convert_ml_params(ml_params) if with_preprocessing else ml_params


def _get_params_grid(model_type, is_cl, with_preprocessing):
    if model_type == "linear":
        ml_params = dict(
            penalty=["l1", "l2"],
            C=[1e-5, 1e-3]
        ) if is_cl else dict(alpha=[1e-5, 1e-3, 1e-1, 10])
    else:
        ml_params = dict(max_depth=[5, 8, 11, 14])
    return _convert_ml_params(ml_params) if with_preprocessing else ml_params


def _convert_ml_params(ml_params):
    return {"{}__{}".format("ml", k): v for k, v in ml_params.items()}


def _assert_prediction(model, df_test, is_cl):
    pred_df = model.predict(df_test)
    expected_columns = ["score", "id1", "id2", "target_cl", "target_rg"]
    if is_cl:
        expected_columns.insert(1, "predicted_class")
    assert list(pred_df.columns) == expected_columns
    assert len(pred_df) == 10

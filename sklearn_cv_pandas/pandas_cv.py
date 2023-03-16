import logging
from functools import reduce
from operator import mul

import numpy
from sklearn import model_selection

from . import (
    model,
    preprocessing
)


logger = logging.getLogger(__name__)


class RandomizedSearchCV(model_selection.RandomizedSearchCV):
    """
    sklearn.model_selection.RandomizedSearchCV with pandas DataFrame interface
    """
    def __init__(self, estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, pre_dispatch='2*n_jobs',
                 cv=None, refit=True, verbose=10, random_state=None, error_score=numpy.nan, return_train_score=True):
        """
        The same manner as [sklearn.model_selection.RandomizedSearchCV](
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
        """
        super(RandomizedSearchCV, self).__init__(
            estimator, param_distributions, n_iter=n_iter, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
            cv=cv, refit=refit, verbose=verbose, random_state=random_state, error_score=error_score,
            return_train_score=return_train_score
        )

    def fit_holdout_pandas(self, df_training, target_column, feature_columns,
                      df_validation=None, ratio_training=None, **kwargs):
        """
        `fit` for pandas DataFrame to perform one set of holdout validation
        Args:
            df_training (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            df_validation (pandas.DataFrame): if specified, used as validation data set
            ratio_training (float): if specified, `df_training` is split for training / validation
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        logger.warning("start learning with {} hyper parameters".format(self.n_iter))
        return _holdout_logic(
            self, df_training, target_column, feature_columns, df_validation, ratio_training, **kwargs
        )

    def fit_cv_pandas(self, df, target_column, feature_columns, cv, **kwargs):
        """
        `fit` for pandas DataFrame to perform cross validation
        Args:
            df (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            cv (int or cross-validation generator): The number of fold for CV
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        logger.warning("start learning with {} hyper parameters".format(self.n_iter))
        return _cv_logic(
            self, df, target_column, feature_columns, cv, **kwargs
        )


class GridSearchCV(model_selection.GridSearchCV):
    def __init__(self, estimator, param_grid, scoring=None,
                 n_jobs=None, pre_dispatch='2*n_jobs', cv=None, refit=True,
                 verbose=10, error_score=numpy.nan, return_train_score=True, random_state=None):
        super(GridSearchCV, self).__init__(
            estimator, param_grid, scoring=scoring, n_jobs=n_jobs, pre_dispatch=pre_dispatch,
            cv=cv, refit=refit, verbose=verbose, error_score=error_score, return_train_score=return_train_score
        )
        self.random_state = random_state

    def fit_holdout_pandas(self, df_training, target_column, feature_columns,
                      df_validation=None, ratio_training=None, **kwargs):
        """
        `fit` for pandas DataFrame to perform one set of holdout validation
        Args:
            df_training (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            df_validation (pandas.DataFrame): if specified, used as validation data set
            ratio_training (float): if specified, `df_training` is split for training / validation
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        logger.warning("start learning with {} parameters".format(_get_num_parameters(self.param_grid)))
        return _holdout_logic(
            self, df_training, target_column, feature_columns, df_validation, ratio_training, **kwargs
        )

    def fit_cv_pandas(self, df, target_column, feature_columns, cv, **kwargs):
        """
        `fit` for pandas DataFrame to perform cross validation
        Args:
            df (pandas.DataFrame): training data set
            target_column (str): column name of prediction target
            feature_columns (list of str): column names of features
            cv (int or cross-validation generator): The number of fold for CV
            **kwargs: Other keyword arguments for original `fit`

        Returns:
            conjurer.ml.Model
        """
        logger.warning("start learning with {} parameters".format(_get_num_parameters(self.param_grid)))
        return _cv_logic(
            self, df, target_column, feature_columns, cv, **kwargs
        )


def _split_for_holdout(df_training, target_column, feature_columns, df_validation, ratio_training, random_state):
    if df_validation is not None:
        x = numpy.concatenate(
            (
                preprocessing.get_x(df_training, feature_columns), 
                preprocessing.get_x(df_validation, feature_columns)
            ),
            axis=0
        )
        y = numpy.concatenate(
            (
                preprocessing.get_y(df_training, target_column), 
                preprocessing.get_y(df_validation, target_column)
            ),
            axis=0
        )
        num_training = len(df_training)
        num_validation = len(df_validation)
    else:
        shuffled_df = df_training.sample(len(df_training), random_state=random_state)
        x = preprocessing.get_x(shuffled_df, feature_columns)
        y = preprocessing.get_y(shuffled_df, target_column)
        num_training = int(ratio_training * len(df_training))
        num_validation = len(df_training) - num_training
    return x, y, num_training, num_validation


def _get_num_parameters(param_grid):
    product = lambda list_values: reduce(mul, list_values, 1)
    return len(param_grid) if isinstance(param_grid, list) \
        else product([len(param_grid[elem]) for elem in param_grid.keys()])


def _holdout_logic(cv_obj, df_training, target_column, feature_columns, df_validation, ratio_training, **kwargs):
    x, y, num_training, num_validation = _split_for_holdout(
        df_training, target_column, feature_columns, df_validation, ratio_training, cv_obj.random_state)
    cv_obj.cv = model_selection.PredefinedSplit(
        numpy.array([-1] * num_training + [0] * num_validation))
    cv_obj.fit(x, y, **kwargs)
    return model.Model(cv_obj, feature_columns=feature_columns, target_column=target_column)


def _cv_logic(cv_obj, df, target_column, feature_columns, cv, **kwargs):
    x = preprocessing.get_x(df, feature_columns)
    y = preprocessing.get_y(df, target_column)
    if isinstance(cv, model_selection.KFold) or isinstance(cv, model_selection.StratifiedKFold) \
            or isinstance(cv, model_selection.GroupKFold):
        cv_obj.cv = cv
    elif isinstance(cv, int):
        if set(y) == {0, 1}:
            cv_obj.cv = model_selection.StratifiedKFold(n_splits=cv, shuffle=True, random_state=cv_obj.random_state)
        else:
            cv_obj.cv = model_selection.KFold(n_splits=cv, shuffle=True, random_state=cv_obj.random_state)
    else:
        raise Exception("parameter `cv`={cv} is not supported, it should be sklearn.**KFold or integer")
    cv_obj.fit(x, y, **kwargs)
    return model.Model(cv_obj, feature_columns=feature_columns, target_column=target_column)

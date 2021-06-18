import copy
from pandas.api import types


def get_x(df, feature_columns):
    for c in feature_columns:
        _check_dtype(df[c])
    converted_df = copy.deepcopy(df)
    for c in feature_columns:
        converted_df[c] = converted_df[c].astype('float64')
    return converted_df[feature_columns].values


def get_y(df, target_column):
    _check_dtype(df[target_column])
    return df[target_column].astype('float64').values


def _check_dtype(series):
    assert types.is_numeric_dtype(series), "{} (dtype: {}) must be one of numeric type".format(seires.name, series.dtype)

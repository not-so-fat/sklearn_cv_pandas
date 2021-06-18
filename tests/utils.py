import numpy
from numpy import random
import pandas


def get_input_df(sample_size, with_missing):
    df = pandas.DataFrame({
        **{
            "id1": range(sample_size),
            "id2": range(10000, 10000+sample_size),
            "target_cl": random.choice([1, 0], sample_size),
            "target_rg": random.normal(0, 1, sample_size),
        },
        **{
            "column{}".format(i): random.normal(0, 1, sample_size)
            for i in range(3)
        },
        **{
            "column{}".format(i): random.choice([1, 0], sample_size)
            for i in range(3, 6)
        }
    })
    if with_missing:
        df["column0"].iloc[5] = numpy.nan
        df["column3"].iloc[5] = numpy.nan
    for i in range(3, 6):
        df["column{}".format(i)] = df["column{}".format(i)].astype("Int64")
    return df

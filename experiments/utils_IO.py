import numpy as np
import os
import pandas as pd


def np_save(folder, filename, obj):
    if not(os.path.exists(folder)):
        os.mkdir(folder)
    np.save(folder + filename, obj)


def read_dataset(dataset_name, ALL_PATHS):
    encoding = "utf-8" if dataset_name == "zomato" else "ISO-8859-1"
    return pd.read_csv(
        ALL_PATHS[dataset_name]["data"],
        sep=",",
        encoding=encoding,
        dtype=str,
        keep_default_na=False,
        skipinitialspace=True,
    )

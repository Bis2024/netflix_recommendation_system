"""
preprocessing.py

Contains helper functions for loading datasets
and preparing features for recommendation.
"""

import pandas as pd


def load_dataset(path):
    """
    Load Netflix dataset.

    Parameters
    ----------
    path : str

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_csv(path)


def create_text_feature(df):
    """
    Combine important text columns.

    Returns
    -------
    DataFrame
    """

    df = df.copy()

    df["text"] = (
        df["listed_in"].fillna("") + " " +
        df["description"].fillna("")
    )

    return df
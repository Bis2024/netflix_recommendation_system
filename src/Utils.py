
"""
utils.py
"""

import joblib


def load_pickle(path):
    """
    Load pickle model.
    """

    return joblib.load(path)


def save_pickle(model,
                path):
    """
    Save pickle model.
    """

    joblib.dump(model,
                path)
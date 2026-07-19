"""Reusable functions to load the Titanic dataset."""

import os

import seaborn as sns

RAW_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "titanic.csv"
)


def load_titanic_data(save_raw: bool = True):
    """Load the Titanic dataset and optionally cache a raw copy to disk.

    Args:
        save_raw: If True, saves a copy of the loaded data to data/raw/titanic.csv.

    Returns:
        The Titanic dataset as a pandas DataFrame.
    """
    df = sns.load_dataset("titanic")

    if save_raw:
        df.to_csv(RAW_DATA_PATH, index=False)

    return df

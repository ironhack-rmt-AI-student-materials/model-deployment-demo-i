"""Reusable functions to preprocess the Titanic dataset."""



FEATURES = ["pclass", "sex", "age", "fare", "embarked"]
TARGET = "survived"

SEX_MAPPING = {"male": 0, "female": 1}
EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}




def compute_imputation_values(df):
    """
    Computes imputation statistics from the given dataframe.
    - IMPORTANT: must be called on the training split only (never on the full
      dataset or the test split), otherwise the returned statistics leak
      information from the test set into training.

    Returns a dict with:
    - 'age': the median age (robust to outliers).
    - 'embarked': the most frequent embarkation port.
    """

    return {
        "age": df["age"].median(),
        "embarked": df["embarked"].mode()[0],
    }


def fill_missing_values(df, impute_values):
    """
    Handles missing values in the Titanic dataset by applying imputation.
    - Fills missing 'age' values with the given median age.
    - Fills missing 'embarked' values with the given most frequent category.

    Args:
        df: The dataframe to impute.
        impute_values: A dict (as returned by `compute_imputation_values`)
            with 'age' and 'embarked' fill values. Must be computed from the
            training split and reused as-is for validation/test/inference
            data to avoid data leakage.
    """

    # Create a copy to keep the original dataframe unchanged
    df = df.copy()

    # Replace missing ages with the given median age
    df["age"] = df["age"].fillna(impute_values["age"])

    # Replace missing embarkation ports with the given most common value
    df["embarked"] = df["embarked"].fillna(impute_values["embarked"])

    # Return the dataframe after imputing the missing values
    return df




def encode_categorical(df):
    """
    Converts categorical features into numerical codes:
    - Encodes 'sex' as binary values (male=0, female=1).
    - Encodes 'embarked' as integer labels (S=0, C=1, Q=2).
    """

    # Create a copy to keep the original dataframe unchanged
    df = df.copy()

    # Replace 'sex' categories with their corresponding numeric codes
    df["sex"] = df["sex"].map(SEX_MAPPING)

    # Replace 'embarked' categories with their corresponding numeric codes
    df["embarked"] = df["embarked"].map(EMBARKED_MAPPING)

    # Return the dataframe with the categorical features encoded
    return df

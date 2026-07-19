"""Reusable functions to preprocess the Titanic dataset."""



FEATURES = ["pclass", "sex", "age", "fare", "embarked"]
TARGET = "survived"

SEX_MAPPING = {"male": 0, "female": 1}
EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}




def fill_missing_values(df):
    """
    Handles missing values in the Titanic dataset by applying imputation.
    - Fills missing 'age' values with the median age (robust to outliers).
    - Fills missing 'embarked' values with the most frequent category.
    """

    # Create a copy to keep the original dataframe unchanged
    df = df.copy()

    # Replace missing ages with the median age 
    df["age"] = df["age"].fillna(df["age"].median())

    # Replace missing embarkation ports with the most common value
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

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

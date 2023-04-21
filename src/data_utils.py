import os
from typing import Tuple

import gdown
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def get_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Download from GDrive all the needed datasets for the project.

    Returns:
        app_train : pd.DataFrame
            Training dataset

        app_test : pd.DataFrame
            Test dataset

        columns_description : pd.DataFrame
            Extra dataframe with detailed description about dataset features
    """
    # Download HomeCredit_columns_description.csv
    if not os.path.exists(config.DATASET_DESCRIPTION):
        gdown.download(
            config.DATASET_DESCRIPTION_URL, config.DATASET_DESCRIPTION, quiet=False
        )

    # Download application_test_aai.csv
    if not os.path.exists(config.DATASET_TEST):
        gdown.download(config.DATASET_TEST_URL, config.DATASET_TEST, quiet=False)

    # Download application_train_aai.csv
    if not os.path.exists(config.DATASET_TRAIN):
        gdown.download(config.DATASET_TRAIN_URL, config.DATASET_TRAIN, quiet=False)

    app_train = pd.read_csv(config.DATASET_TRAIN)
    app_test = pd.read_csv(config.DATASET_TEST)
    columns_description = pd.read_csv(config.DATASET_DESCRIPTION_URL)

    return app_train, app_test, columns_description


def get_feature_target(
    app_train: pd.DataFrame, app_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Separates our train and test datasets columns between Features
    (the input to the model) and Targets (what the model has to predict with the
    given features).

    Arguments:
        app_train : pd.DataFrame
            Training datasets
        app_test : pd.DataFrame
            Test datasets

    Returns:
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
    """
    X_train, y_train, X_test, y_test = (
        app_train.drop("TARGET", axis=1),
        app_train["TARGET"],
        app_test.drop("TARGET", axis=1),
        app_test["TARGET"],
    )
    # TODO
    # Assign to X_train all the columns from app_train except "TARGET"
    # Assign to y_train the "TARGET" column
    # Assign to X_test all the columns from app_test except "TARGET"
    # Assign to y_test the "TARGET" column

    return X_train, y_train, X_test, y_test


def get_train_val_sets(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Split training dataset in two new sets used for train and validation.

    Arguments:
        X_train : pd.DataFrame
            Original training features
        y_train: pd.Series
            Original training labels/target

    Returns:
        X_train : pd.DataFrame
            Training features
        X_val : pd.DataFrame
            Validation features
        y_train : pd.Series
            Training target
        y_val : pd.Series
            Validation target
    """
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=15, shuffle=True)

    # TODO
    # Use the `sklearn.model_selection.train_test_split` function with
    # `X_train`, `y_train` datasets.
    # Assign only 20% of the dataset for testing (see `test_size` parameter in
    # `train_test_split`).
    # Assign a seed so we get reproducible output across multiple function
    # calls (see `random_state` parameter in `train_test_split`).
    # Shuffle the data (see `shuffle` parameter in `train_test_split`).

    return X_train, X_val, y_train, y_val

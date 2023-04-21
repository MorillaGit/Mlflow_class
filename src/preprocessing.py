from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


def preprocess_data(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pre processes data for modeling. Receives train, val and test dataframes
    and returns numpy ndarrays of cleaned up dataframes with feature engineering
    already performed.

    Arguments:
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame

    Returns:
        train : np.ndarrary
        val : np.ndarrary
        test : np.ndarrary
    """
    # Print shape of input data
    # print("Input train data shape: ", train_df.shape)
    # print("Input val data shape: ", val_df.shape)
    # print("Input test data shape: ", test_df.shape, "\n")

    # Make a copy of the dataframes
    # Make a copy of the dataframes
    working_train_df = train_df.copy()
    working_val_df = val_df.copy()
    working_test_df = test_df.copy()

    # Replace 365243 in "DAYS_EMPLOYED" column with NaN
    cols_to_replace = ["DAYS_EMPLOYED"]
    for df in [working_train_df, working_val_df, working_test_df]:
        df[cols_to_replace] = df[cols_to_replace].replace({365243: np.nan})

    # Split columns into ordinal and categorical variables
    categorical_cols = [
        col
        for col in working_train_df.select_dtypes(include=["object"])
        if working_train_df[col].nunique() > 2
    ]
    ordinal_cols = [
        col
        for col in working_train_df.select_dtypes(include=["object"])
        if working_train_df[col].nunique() <= 2
    ]

    # One-hot encode categorical variables
    ohe = OneHotEncoder(drop=None)
    ohe_df = ohe.fit_transform(working_train_df[categorical_cols]).toarray()
    ohe_df_2 = ohe.transform(working_test_df[categorical_cols]).toarray()
    ohe_df_3 = ohe.transform(working_val_df[categorical_cols]).toarray()

    # Ordinal encode ordinal variables
    oe = OrdinalEncoder()
    ordinal_df = oe.fit_transform(working_train_df[ordinal_cols])
    ordinal_df_2 = oe.transform(working_test_df[ordinal_cols])
    ordinal_df_3 = oe.transform(working_val_df[ordinal_cols])

    # Drop original categorical and ordinal columns from dataframes
    working_train_df = working_train_df.drop(columns=categorical_cols + ordinal_cols)
    working_test_df = working_test_df.drop(columns=categorical_cols + ordinal_cols)
    working_val_df = working_val_df.drop(columns=categorical_cols + ordinal_cols)

    # Convert dataframes to numpy arrays and concatenate with encoded categorical and ordinal data
    working_train_df = np.concatenate(
        (working_train_df.to_numpy(), ordinal_df, ohe_df), axis=1
    )
    working_test_df = np.concatenate(
        (working_test_df.to_numpy(), ordinal_df_2, ohe_df_2), axis=1
    )
    working_val_df = np.concatenate(
        (working_val_df.to_numpy(), ordinal_df_3, ohe_df_3), axis=1
    )

    # Impute missing values with SimpleImputer
    imputer = SimpleImputer()
    working_train_df = imputer.fit_transform(working_train_df)
    working_test_df = imputer.transform(working_test_df)
    working_val_df = imputer.transform(working_val_df)

    # Scale the data using MinMaxScaler
    scaler = MinMaxScaler()
    working_train_df = scaler.fit_transform(working_train_df)
    working_test_df = scaler.transform(working_test_df)
    working_val_df = scaler.transform(working_val_df)

    # Print the shapes of the resulting arrays
    # print("Output train data shape: ", working_train_df.shape, "\n")
    # print("Output val data shape: ", working_val_df.shape, "\n")
    # print("Output test data shape: ", working_test_df.shape, "\n")

    return working_train_df, working_val_df, working_test_df

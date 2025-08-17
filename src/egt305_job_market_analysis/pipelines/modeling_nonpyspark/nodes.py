import numpy as np
import pandas as pd
import json
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.dummy import DummyRegressor


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

# Fallback dummy model
dummy_model = DummyRegressor(strategy="mean")
dummy_model.fit([[0]], [0])
dummy_metrics = {"skipped": True}


def to_py(val):
    """Convert NumPy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.generic, np.bool_)):
        return val.item()
    return val


# ---------------------------------------------------------------------
# Data Split + Encoding
# ---------------------------------------------------------------------

def split_and_encode_model_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Splits and encodes model input data into train/test sets for regression.
    - Ordinal encodes education with predefined order
    - One-hot encodes nominal features
    - Numeric columns are passed through
    - job_id is preserved separately for joining predictions
    - salary_k is the regression target
    """

    df = data.copy()

    # Define feature groups
    ordinal_features = ["education"]
    nominal_features = ["company_id", "job_role", "major", "industry"]
    numeric_features = ["years_experience", "distance_from_cbd"]

    # Ensure categorical dtypes
    for col in nominal_features:
        if col in df.columns and not pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype("category")
    if "education" in df.columns:
        df["education"] = df["education"].astype("category")

    # Define target and identifiers
    y = df["salary_k"].astype(float)
    job_ids = df["job_id"]

    # Drop target + identifier from features
    X = df.drop(columns=["salary_k", "job_id"])

    # Train/test split (no stratify for regression)
    X_train, X_test, y_train, y_test, job_train, job_test = train_test_split(
        X, y, job_ids,
        test_size=test_size,
        random_state=random_state
    )

    # Preprocessor: ordinal + one-hot + passthrough
    preprocessor = ColumnTransformer(transformers=[
        ("ord", OrdinalEncoder(categories=[["NONE", "HIGH_SCHOOL", "BACHELORS", "MASTERS", "DOCTORAL"]],
                               handle_unknown="use_encoded_value", unknown_value=-1),
         ordinal_features),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features)
    ], remainder="passthrough")  # numeric passthrough

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # Reconstruct feature names
    ordinal_names = ordinal_features
    onehot_names = preprocessor.named_transformers_["nom"].get_feature_names_out(nominal_features)
    final_feature_names = list(ordinal_names) + list(onehot_names) + numeric_features

    # Sanity check
    assert X_train_enc.shape[1] == len(final_feature_names), "Mismatch between encoded features and names"

    X_train_df = pd.DataFrame(X_train_enc, columns=final_feature_names, index=X_train.index).astype(np.float32)
    X_test_df = pd.DataFrame(X_test_enc, columns=final_feature_names, index=X_test.index).astype(np.float32)

    return {
        "X_train": X_train_df.reset_index(drop=True),
        "X_test": X_test_df.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "job_train": job_train.reset_index(drop=True),
        "job_test": job_test.reset_index(drop=True),
        "model_input_feature_names": pd.Series(final_feature_names)
    }


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def train_linear_model(
    X_train, X_test, y_train, y_test, job_test,
    model_input_feature_names, skip: bool
) -> tuple:
    """Train a Linear Regression model for salary prediction."""
    if skip:
        return dummy_model, dummy_metrics, pd.DataFrame()

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Feature importance = absolute coefficient weights
    coefs = model.coef_
    feature_importance = {str(k): to_py(abs(v)) for k, v in zip(model_input_feature_names, coefs)}

    # Predictions DataFrame
    predictions_df = X_test.copy()
    predictions_df["job_id"] = job_test.values
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_pred

    # Metrics
    metrics = {
        "r2_score": to_py(r2_score(y_test, y_pred)),
        "rmse": to_py(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": to_py(mean_absolute_error(y_test, y_pred)),
        "feature_importance": feature_importance
    }

    return model, metrics, predictions_df


def train_randomforest_model(
    X_train, X_test, y_train, y_test, job_test,
    model_input_feature_names,
    n_estimators: int, max_depth: Optional[int],
    skip: bool
) -> tuple:
    """Train a Random Forest Regressor for salary prediction."""
    if skip:
        return dummy_model, dummy_metrics, pd.DataFrame()

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Feature importance (Gini importance for regression)
    importances = model.feature_importances_
    feature_importance = {str(k): to_py(v) for k, v in zip(model_input_feature_names, importances)}

    # Predictions DataFrame
    predictions_df = X_test.copy()
    predictions_df["job_id"] = job_test.values
    predictions_df["y_true"] = y_test.values
    predictions_df["y_pred"] = y_pred

    # Metrics
    metrics = {
        "r2_score": to_py(r2_score(y_test, y_pred)),
        "rmse": to_py(mean_squared_error(y_test, y_pred, squared=False)),
        "mae": to_py(mean_absolute_error(y_test, y_pred)),
        "feature_importance": feature_importance
    }

    return model, metrics, predictions_df

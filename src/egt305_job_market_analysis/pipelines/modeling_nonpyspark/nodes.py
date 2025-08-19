# ======================================================
# Global Imports
# ======================================================
import logging
import warnings
from typing import Optional, Tuple, Generator

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold


# ======================================================
# Setup
# ======================================================
# Make CUDA multiprocessing safe for environments where this script may be imported.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Silence specific deprecation noise from older amp usage (if any upstream libs trigger it).
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp.*is deprecated.*")

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Torch] Using device: {device}")

# ======================================================
# Split (numeric features + integer-encoded categoricals for NN)
# ======================================================

def split_model_data_three_way(
    data: pd.DataFrame,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Split into train/valid/test sets with integer-encoded categorical columns.

    Returns:
      X_train, X_valid, X_test,
      y_train, y_valid, y_test,
      job_train, job_valid, job_test
    """
    df = data.copy()
    target_col = "salary_k"
    id_col = "job_id"

    # Encode categoricals
    for col in cat_cols:
        if not pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes.astype("int32")

    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    job_ids = df[id_col]

    # First split off test
    X_train_valid, X_test, y_train_valid, y_test, job_train_valid, job_test = train_test_split(
        X, y, job_ids, test_size=test_size, random_state=random_state
    )

    # Then split train into train/valid
    X_train, X_valid, y_train, y_valid, job_train, job_valid = train_test_split(
        X_train_valid, y_train_valid, job_train_valid,
        test_size=valid_size, random_state=random_state
    )

    return (
        X_train.reset_index(drop=True),
        X_valid.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        job_train.reset_index(drop=True),
        job_valid.reset_index(drop=True),
        job_test.reset_index(drop=True),
    )


# ======================================================
# Model
# ======================================================

class EmbeddingNNRegressor(nn.Module):
    """
    Simple MLP regressor that concatenates numeric features with learned embeddings
    for each categorical input column.
    """
    def __init__(self, n_num_features: int, cat_cardinalities: list[int]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, min(32, (card // 2) + 1)) for card in cat_cardinalities
        ])
        emb_total = sum(e.embedding_dim for e in self.embeddings)
        self.net = nn.Sequential(
            nn.Linear(n_num_features + emb_total, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor):
        emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_cat = (
            torch.cat(emb_list, dim=1)
            if emb_list
            else torch.zeros(x_num.size(0), 0, device=x_num.device)
        )
        x_all = torch.cat([x_num, emb_cat], dim=1)
        return self.net(x_all)


# ======================================================
# Training
# ======================================================

def train_nn_torch(
    data: pd.DataFrame,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 5,
):
    """
    Train a neural network with embeddings for categorical columns and MinMax scaling
    of the target. Uses a 3-way split (train/valid/test).

    Returns:
      (
        model_state_dict,     # state dict of trained model (best by val loss)
        metrics,              # {"r2", "mae", "rmse"} computed on test set
        predictions_df,       # DataFrame: ["job_id", "y_true", "y_pred"] (test set only, original scale)
        history,              # {"train_loss": [...], "val_loss": [...]}
        metadata,             # {"num_cols", "cat_cols", "cat_cardinalities"}
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        job_train, job_valid, job_test
      )
    """
    # 1) Three-way split
    (
        X_train, X_valid, X_test,
        y_train, y_valid, y_test,
        job_train, job_valid, job_test,
    ) = split_model_data_three_way(
        data=data,
        cat_cols=cat_cols,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )

    # 2) Identify numeric vs categorical
    cat_cols = list(cat_cols)
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    cat_cardinalities = [int(X_train[c].max() + 1) for c in cat_cols]

    # 3) Scale target
    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_valid_scaled = y_scaler.transform(y_valid.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    # 4) To tensors
    def to_tensors(X, y):
        return (
            torch.tensor(X[num_cols].values, dtype=torch.float32),
            torch.tensor(X[cat_cols].values, dtype=torch.long),
            torch.tensor(y, dtype=torch.float32),
        )
    X_train_num, X_train_cat, y_train_t = to_tensors(X_train, y_train_scaled)
    X_valid_num, X_valid_cat, y_valid_t = to_tensors(X_valid, y_valid_scaled)
    X_test_num, X_test_cat, y_test_t = to_tensors(X_test, y_test_scaled)

    train_loader = DataLoader(
        TensorDataset(X_train_num, X_train_cat, y_train_t),
        batch_size=batch_size, shuffle=True, pin_memory=True
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid_num, X_valid_cat, y_valid_t),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )

    # 5) Model / loss / optimizer
    model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val, best_state, no_improve = float("inf"), None, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        # Train
        model.train()
        running = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb_num.size(0)
        train_loss = running / len(train_loader.dataset)

        # Validate
        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb_num, xb_cat, yb in valid_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                pred = model(xb_num, xb_cat)
                loss = criterion(pred, yb)
                running += loss.item() * xb_num.size(0)
        val_loss = running / len(valid_loader.dataset)
        scheduler.step(val_loss)

        # Record losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val, best_state, no_improve = val_loss, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                tqdm.write(f"Early stopping at epoch {epoch+1}")
                if best_state is not None:
                    model.load_state_dict(best_state)
                break

        tqdm.write(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # 6) Final test evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_num.to(device), X_test_cat.to(device)).cpu().numpy().flatten()
    y_true_scaled = y_test_t.cpu().numpy().flatten()

    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = y_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()

    predictions = pd.DataFrame({"job_id": job_test, "y_true": y_true, "y_pred": y_pred})

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_cardinalities": cat_cardinalities,
    }

    # Return model artifacts AND all split sets (train/valid/test)
    return (
        model.state_dict(),   # trained model weights
        metrics,              # test set metrics
        predictions,          # DataFrame with job_id, y_true, y_pred (test set)
        history,              # training/validation loss curves
        metadata              # info about features & embeddings
    )   

# ======================================================
# Split (OHE for sklearn models, with train/valid/test)
# ======================================================
def split_ohe_three_way(
    data: pd.DataFrame,
    target_col: str = "salary_k",
    id_col: str = "job_id",
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           pd.Series, pd.Series, pd.Series,
           pd.Series, pd.Series, pd.Series,
           list[str]]:
    """
    Split into train/valid/test and apply One-Hot Encoding on categorical columns.

    Returns:
      X_train_enc, X_valid_enc, X_test_enc,
      y_train, y_valid, y_test,
      job_train, job_valid, job_test,
      feature_names
    """
    df = data.copy()

    # Features / target / IDs
    X = df.drop(columns=[target_col, id_col])
    y = df[target_col]
    job_ids = df[id_col]

    # First split train+valid vs test
    X_train_valid, X_test, y_train_valid, y_test, job_train_valid, job_test = train_test_split(
        X, y, job_ids, test_size=test_size, random_state=random_state
    )

    # Then split train vs valid
    X_train, X_valid, y_train, y_valid, job_train, job_valid = train_test_split(
        X_train_valid, y_train_valid, job_train_valid,
        test_size=valid_size, random_state=random_state
    )

    # Normalize cat_cols input
    cat_list = [c for c in list(cat_cols) if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_list]

    # Fit encoder on combined sets (train+valid+test) to ensure consistent columns
    if len(cat_list) > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(pd.concat([X_train[cat_list], X_valid[cat_list], X_test[cat_list]], axis=0))
        X_train_cat = enc.transform(X_train[cat_list])
        X_valid_cat = enc.transform(X_valid[cat_list])
        X_test_cat = enc.transform(X_test[cat_list])
        ohe_names = enc.get_feature_names_out(cat_list).tolist()
    else:
        X_train_cat = np.empty((X_train.shape[0], 0), dtype=float)
        X_valid_cat = np.empty((X_valid.shape[0], 0), dtype=float)
        X_test_cat = np.empty((X_test.shape[0], 0), dtype=float)
        ohe_names = []

    # Numeric parts
    X_train_num = X_train[num_cols].to_numpy()
    X_valid_num = X_valid[num_cols].to_numpy()
    X_test_num = X_test[num_cols].to_numpy()

    # Final encoded design matrices
    X_train_enc = np.hstack([X_train_num, X_train_cat])
    X_valid_enc = np.hstack([X_valid_num, X_valid_cat])
    X_test_enc = np.hstack([X_test_num, X_test_cat])

    feature_names = num_cols + ohe_names

    return (
        X_train_enc, X_valid_enc, X_test_enc,
        y_train.reset_index(drop=True),
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        job_train.reset_index(drop=True),
        job_valid.reset_index(drop=True),
        job_test.reset_index(drop=True),
        feature_names
    )


# ======================================================
# Standardized training: Linear Regression (with OHE)
# ======================================================
def train_linear_regression_std(
    data: pd.DataFrame,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    LinearRegression, dict, dict, dict, dict
]:
    """
    Train a LinearRegression model using OHE features with train/valid/test split.

    Returns:
      (
        fitted_model,
        metrics,        # {"r2", "mae", "rmse"} on test set
        predictions,    # {"valid": DataFrame, "test": DataFrame}
        history,        # empty dict to standardize signature
        metadata        # feature metadata
      )
    """
    (
        X_train_enc, X_valid_enc, X_test_enc,
        y_train, y_valid, y_test,
        job_train, job_valid, job_test,
        feature_names
    ) = split_ohe_three_way(
        data=data,
        cat_cols=cat_cols,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )

    model = LinearRegression()
    model.fit(X_train_enc, y_train)

    # Validation predictions
    y_valid_pred = model.predict(X_valid_enc)
    valid_df = pd.DataFrame({
        "job_id": job_valid,
        "y_true": y_valid,
        "y_pred": y_valid_pred
    })
    valid_df["split"] = "valid"

    # Test predictions
    y_test_pred = model.predict(X_test_enc)
    test_df = pd.DataFrame({
        "job_id": job_test,
        "y_true": y_test,
        "y_pred": y_test_pred
    })
    test_df["split"] = "test"

    # Combine into one DataFrame
    predictions = pd.concat([valid_df, test_df], ignore_index=True)

    metrics = {
        "r2": float(r2_score(y_test, y_test_pred)),
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }

    history = {}
    metadata = {
        "cat_cols": list(cat_cols),
        "feature_names": feature_names,
        "model_type": "LinearRegression",
        "random_state": random_state,
        "test_size": test_size,
        "valid_size": valid_size,
    }

    return model, metrics, predictions, history, metadata


# ======================================================
# Standardized training: Random Forest (with OHE)
# ======================================================
def train_random_forest_std(
    data: pd.DataFrame,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> tuple[
    RandomForestRegressor, dict, dict, dict, dict
]:
    """
    Train a RandomForestRegressor using OHE features with train/valid/test split.

    Returns:
      (
        fitted_model,
        metrics,        # {"r2", "mae", "rmse"} on test set
        predictions,    # {"valid": DataFrame, "test": DataFrame}
        history,        # empty dict to standardize signature
        metadata        # feature metadata
      )
    """
    (
        X_train_enc, X_valid_enc, X_test_enc,
        y_train, y_valid, y_test,
        job_train, job_valid, job_test,
        feature_names
    ) = split_ohe_three_way(
        data=data,
        cat_cols=cat_cols,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train_enc, y_train)

    # Validation predictions
    y_valid_pred = model.predict(X_valid_enc)
    valid_df = pd.DataFrame({
        "job_id": job_valid,
        "y_true": y_valid,
        "y_pred": y_valid_pred
    })
    valid_df["split"] = "valid"

    # Test predictions
    y_test_pred = model.predict(X_test_enc)
    test_df = pd.DataFrame({
        "job_id": job_test,
        "y_true": y_test,
        "y_pred": y_test_pred
    })
    test_df["split"] = "test"

    # Combine into one DataFrame
    predictions = pd.concat([valid_df, test_df], ignore_index=True)
    metrics = {
        "r2": float(r2_score(y_test, y_test_pred)),
        "mae": float(mean_absolute_error(y_test, y_test_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
    }
    
    history = {}
    metadata = {
        "cat_cols": list(cat_cols),
        "feature_names": feature_names,
        "model_type": "RandomForestRegressor",
        "n_estimators": n_estimators,
        "random_state": random_state,
        "test_size": test_size,
        "valid_size": valid_size,
    }

    return model, metrics, predictions, history, metadata

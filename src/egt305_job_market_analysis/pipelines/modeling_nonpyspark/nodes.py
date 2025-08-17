import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
# Make multiprocessing safe for CUDA
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # It's already set, safe to ignore
    pass
import warnings
# Suppress PyTorch AMP deprecation warnings
warnings.filterwarnings(
    "ignore",
    message=".*torch\\.cuda\\.amp.*is deprecated.*"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Torch] Using device: {device}")

def split_and_encode_model_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train/test, encode categorical features,
    and return clean train/test sets for modeling.
    """

    df = data.copy()
    target_col = "salary_k"

    ordinal_features = ["education"]
    ordinal_categories = [["NONE","HIGH_SCHOOL","BACHELORS","MASTERS","DOCTORAL"]]
    nominal_features = ["job_role", "major", "industry", "company_id"]
    numeric_features = ["years_experience", "distance_from_cbd"]

    # Split features/target
    X = df.drop(columns=[target_col, "job_id"])
    y = df[target_col]
    job_ids = df["job_id"]

    X_train, X_test, y_train, y_test, job_train, job_test = train_test_split(
        X, y, job_ids, test_size=test_size, random_state=random_state
    )

    # Encode categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(categories=ordinal_categories, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_features),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        ],
        remainder="passthrough"
    )

    X_train_enc = preprocessor.fit_transform(X_train).astype("float32")
    X_test_enc = preprocessor.transform(X_test).astype("float32")

    # Convert back to DataFrame for Kedro datasets
    ordinal_names = ordinal_features
    onehot_names = preprocessor.named_transformers_["nom"].get_feature_names_out(nominal_features)
    final_feature_names = list(ordinal_names) + list(onehot_names) + numeric_features

    X_train_df = pd.DataFrame(X_train_enc, columns=final_feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_enc, columns=final_feature_names, index=X_test.index)

    return (
        X_train_df.reset_index(drop=True),
        X_test_df.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        job_train.reset_index(drop=True),
        job_test.reset_index(drop=True)
    )

# Pick device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Torch] Using device: {device}")

# ======================================================
# 1. Torch Models
# ======================================================

class LinearRegressionTorch(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


class NNRegressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


# ======================================================
# 2. Training Functions
# ======================================================

def train_linear_torch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    job_test: pd.Series
):
    """Train a Torch Linear Regression model."""

    # Convert to tensors and move to device
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    model = LinearRegressionTorch(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t)

    # Convert predictions back to CPU for metrics
    y_pred = y_pred_t.cpu().numpy().flatten()
    y_true = y_test_t.cpu().numpy().flatten()

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }

    predictions = pd.DataFrame({
        "job_id": job_test,
        "y_true": y_true,
        "y_pred": y_pred
    })

    # Return state_dict so it can be pickled
    return model.state_dict(), metrics, predictions


def train_nn_torch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    job_test: pd.Series
):
    """Train a Torch NN Regressor with GPU, multiprocessing, and AMP."""

    # --- keep dataset on CPU ---
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_t  = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(
        dataset,
        batch_size=2048,         # keep GPU busy
        shuffle=True,
        num_workers=8,          # tune for your CPU cores
        pin_memory=True,
        persistent_workers=True
    )

    model = NNRegressor(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scaler = torch.cuda.amp.GradScaler()  # AMP

    for epoch in range(100):  # tune epochs
        model.train()
        for xb, yb in loader:
            # transfer to GPU asynchronously
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # --- Evaluation ---
    model.eval()
    with torch.no_grad():
        y_pred_t = model(X_test_t.to(device))
    y_pred = y_pred_t.cpu().numpy().flatten()
    y_true = y_test_t.numpy().flatten()

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }

    predictions = pd.DataFrame({
        "job_id": job_test,
        "y_true": y_true,
        "y_pred": y_pred
    })

    return model.state_dict(), metrics, predictions

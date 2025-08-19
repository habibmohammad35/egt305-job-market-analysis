# ======================================================
# Imports
# ======================================================
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tqdm.auto import tqdm

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# Use global device if already defined; otherwise pick CUDA if available.
device = globals().get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# ======================================================
# Model
# ======================================================
class EmbeddingNNRegressor(nn.Module):
    """
    Simple MLP regressor that concatenates numeric features with learned embeddings
    for each categorical input column.
    """
    def __init__(self, n_num_features: int, cat_cardinalities: List[int]):
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
        if len(self.embeddings) > 0:
            emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            emb_cat = torch.cat(emb_list, dim=1)
        else:
            emb_cat = torch.zeros(x_num.size(0), 0, device=x_num.device, dtype=x_num.dtype)
        x = torch.cat([x_num, emb_cat], dim=1)
        return self.net(x)


# ======================================================
# Spark preprocessing + three-way split + target scaling (all in Spark)
# ======================================================
def spark_preprocess_and_split_three_way(
    sdf: SparkDataFrame,
    *,
    cat_cols: Tuple[str, str] = ("edu_major", "industry_role"),
    target_col: str = "salary_k",
    id_col: str = "job_id",
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,   # train_pdf, valid_pdf, test_pdf
    List[str], List[str], List[int],            # num_cols, cat_cols_list, cat_cardinalities
    float, float                                 # y_min, y_max computed on TRAIN
]:
    """
    Perform all data operations in Spark:
      - drop null targets, StringIndex categorical cols, cast numerics, median-impute numerics,
      - split into train/valid/test using Spark randomSplit (test first, then valid from remainder),
      - compute y_min/y_max on TRAIN only and create y_scaled on all splits.
    Finally, collect train/valid/test to pandas for Torch.

    Returns:
      train_pdf, valid_pdf, test_pdf,
      num_cols, cat_cols_list, cat_cardinalities,
      y_min, y_max  (computed from TRAIN only)
    """
    # Guard & drop rows with null target
    sdf = sdf.filter(F.col(target_col).isNotNull())

    # Keep only categorical columns that exist
    cat_present = [c for c in cat_cols if c in sdf.columns]

    # Index categoricals with a 'keep' bucket for invalid/unseen
    stages = []
    for c in cat_present:
        stages.append(StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep"))

    if stages:
        pipe = Pipeline(stages=stages)
        si_model = pipe.fit(sdf)
        sdf = si_model.transform(sdf)

        # Replace original cat columns with integer indices
        cat_cardinalities: List[int] = []
        for i, c in enumerate(cat_present):
            sdf = (
                sdf.drop(c)
                   .withColumnRenamed(f"{c}_idx", c)
                   .withColumn(c, F.col(c).cast(T.IntegerType()))
            )
            # +1 for 'keep' bucket
            card = len(si_model.stages[i].labels) + 1
            cat_cardinalities.append(int(max(1, card)))
    else:
        cat_cardinalities = []

    # Numeric columns: everything except id, target, and the (now-indexed) categoricals
    current_cols = sdf.columns
    cat_cols_list = cat_present
    num_cols = [c for c in current_cols if c not in (id_col, target_col) and c not in cat_cols_list]

    # Cast numerics to double
    for c in num_cols:
        sdf = sdf.withColumn(c, F.col(c).cast(T.DoubleType()))

    # Median-impute numerics (approx percentile_approx)
    for c in num_cols:
        med = sdf.selectExpr(f"percentile_approx({c}, 0.5)").first()[0]
        fill_val = float(med) if med is not None else 0.0
        sdf = sdf.fillna({c: fill_val})

    # Ensure id_col and target_col types are reasonable
    sdf = sdf.withColumn(id_col, F.col(id_col).cast(T.StringType()))
    sdf = sdf.withColumn(target_col, F.col(target_col).cast(T.DoubleType()))

    # --- Spark splits ---
    # 1) Split off TEST
    w_train_valid = max(1e-9, 1.0 - float(test_size))
    w_test = float(test_size)
    train_valid_sdf, test_sdf = sdf.randomSplit([w_train_valid, w_test], seed=random_state)

    # Fallback: if test is empty, re-sample from full set deterministically
    if test_sdf.count() == 0 and sdf.count() > 0:
        # sample fraction for test
        frac = min(0.99, max(0.01, w_test))
        sampled_test = sdf.sample(withReplacement=False, fraction=frac, seed=random_state)
        # remainder via anti-join on id_col
        remainder = sdf.join(sampled_test.select(id_col).withColumnRenamed(id_col, "_id_"),
                             sdf[id_col] == F.col("_id_"), "left_anti").drop("_id_")
        test_sdf = sampled_test
        train_valid_sdf = remainder

    # 2) Split TRAIN vs VALID from the train_valid portion
    w_train = max(1e-9, 1.0 - float(valid_size))
    w_valid = float(valid_size)
    train_sdf, valid_sdf = train_valid_sdf.randomSplit([w_train, w_valid], seed=random_state)

    # Fallbacks for empty splits
    if train_sdf.count() == 0 and train_valid_sdf.count() > 0:
        train_sdf = train_valid_sdf.sample(False, 0.8, seed=random_state)
        valid_sdf = train_valid_sdf.join(train_sdf.select(id_col).withColumnRenamed(id_col, "_id_"),
                                         train_valid_sdf[id_col] == F.col("_id_"), "left_anti").drop("_id_")
    if valid_sdf.count() == 0 and train_valid_sdf.count() > 1:
        valid_sdf = train_valid_sdf.sample(False, max(0.01, w_valid), seed=random_state)
        train_sdf = train_valid_sdf.join(valid_sdf.select(id_col).withColumnRenamed(id_col, "_id_"),
                                         train_valid_sdf[id_col] == F.col("_id_"), "left_anti").drop("_id_")

    # --- Target scaling (fit on TRAIN only) ---
    y_stats = train_sdf.agg(F.min(target_col).alias("ymin"), F.max(target_col).alias("ymax")).first()
    y_min = float(y_stats["ymin"]) if y_stats and y_stats["ymin"] is not None else 0.0
    y_max = float(y_stats["ymax"]) if y_stats and y_stats["ymax"] is not None else y_min
    denom = y_max - y_min if (y_max - y_min) != 0 else 1e-12

    def add_scaled(df):
        return df.withColumn("y_scaled", (F.col(target_col) - F.lit(y_min)) / F.lit(denom))

    train_sdf = add_scaled(train_sdf)
    valid_sdf = add_scaled(valid_sdf)
    test_sdf  = add_scaled(test_sdf)

    # Columns to collect (order: id, target, cats, nums, y_scaled last)
    select_cols = [id_col, target_col] + cat_cols_list + num_cols + ["y_scaled"]

    train_pdf = train_sdf.select(*select_cols).toPandas().reset_index(drop=True)
    valid_pdf = valid_sdf.select(*select_cols).toPandas().reset_index(drop=True)
    test_pdf  = test_sdf.select(*select_cols).toPandas().reset_index(drop=True)

    # Basic guards
    if train_pdf.empty or valid_pdf.empty or test_pdf.empty:
        raise ValueError("One of the splits (train/valid/test) is empty. Check split ratios and data size.")

    # Ensure dtypes are consistent post-collection
    for c in num_cols:
        train_pdf[c] = pd.to_numeric(train_pdf[c], errors="coerce")
        valid_pdf[c] = pd.to_numeric(valid_pdf[c], errors="coerce")
        test_pdf[c]  = pd.to_numeric(test_pdf[c], errors="coerce")

    for c in cat_cols_list:
        train_pdf[c] = pd.to_numeric(train_pdf[c], errors="coerce").fillna(0).astype(int)
        valid_pdf[c] = pd.to_numeric(valid_pdf[c], errors="coerce").fillna(0).astype(int)
        test_pdf[c]  = pd.to_numeric(test_pdf[c], errors="coerce").fillna(0).astype(int)

    # Final NaN/Inf cleaning for numerics (rare after Spark impute, but safe)
    train_pdf[num_cols] = train_pdf[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    valid_pdf[num_cols] = valid_pdf[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_pdf[num_cols]  = test_pdf[num_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return train_pdf, valid_pdf, test_pdf, num_cols, cat_cols_list, cat_cardinalities, y_min, y_max

# ======================================================
# Torch training (Spark preproc + split + scale → Torch)
# ======================================================
def train_nn_torch_spark(
    data: SparkDataFrame,
    cat_cols: Tuple[str, str] = ("edu_major", "industry_role"),
    target_col: str = "salary_k",
    id_col: str = "job_id",
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 5,
):
    """
    Full Spark data pipeline (encode/impute/split/scale) + local Torch training with early stopping on VALID.
    Returns:
        (
            model_state_dict,   # weights at best val
            metrics,            # {"r2", "mae", "rmse"} on TEST
            predictions_df,     # [id_col, "y_true", "y_pred"] for TEST (original scale)
            history,            # {"train_loss": [...], "val_loss": [...]}
            metadata            # {"num_cols","cat_cols","cat_cardinalities",...}
        )
    """
    # 1) Spark preprocessing + three-way split + target scaling (train-based)
    (train_pdf, valid_pdf, test_pdf,
     num_cols, cat_cols_list, cat_cardinalities,
     y_min, y_max) = spark_preprocess_and_split_three_way(
        data,
        cat_cols=cat_cols,
        target_col=target_col,
        id_col=id_col,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )
    denom = (y_max - y_min) if (y_max - y_min) != 0 else 1e-12

    # 2) Build Torch tensors (features + scaled target)
    def to_tensors(pdf: pd.DataFrame):
        # numeric matrix
        if num_cols:
            Xn = np.nan_to_num(pdf[num_cols].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            Xn_t = torch.tensor(Xn, dtype=torch.float32)
        else:
            Xn_t = torch.empty((len(pdf), 0), dtype=torch.float32)

        # categorical matrix
        if cat_cols_list:
            Xc = np.nan_to_num(pdf[cat_cols_list].to_numpy(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.int64)
            Xc_t = torch.tensor(Xc, dtype=torch.long)
        else:
            Xc_t = torch.empty((len(pdf), 0), dtype=torch.long)

        y_t = torch.tensor(pdf["y_scaled"].to_numpy().reshape(-1, 1), dtype=torch.float32)
        return Xn_t, Xc_t, y_t

    X_train_num, X_train_cat, y_train_t = to_tensors(train_pdf)
    X_valid_num, X_valid_cat, y_valid_t = to_tensors(valid_pdf)
    X_test_num,  X_test_cat,  y_test_t  = to_tensors(test_pdf)

    # 3) Clamp cat indices to valid ranges
    for i, card in enumerate(cat_cardinalities):
        if X_train_cat.shape[1] > i:
            X_train_cat[:, i].clamp_(0, card - 1)
        if X_valid_cat.shape[1] > i:
            X_valid_cat[:, i].clamp_(0, card - 1)
        if X_test_cat.shape[1] > i:
            X_test_cat[:, i].clamp_(0, card - 1)

    # 4) DataLoaders
    use_pin = device.type == "cuda"
    num_workers = max(0, min(4, (os.cpu_count() or 2) - 1))
    train_loader = DataLoader(
        TensorDataset(X_train_num, X_train_cat, y_train_t),
        batch_size=batch_size, shuffle=True,
        pin_memory=use_pin, num_workers=num_workers,
        persistent_workers=(num_workers > 0), drop_last=False,
    )
    valid_loader = DataLoader(
        TensorDataset(X_valid_num, X_valid_cat, y_valid_t),
        batch_size=batch_size, shuffle=False,
        pin_memory=use_pin, num_workers=num_workers,
        persistent_workers=(num_workers > 0), drop_last=False,
    )

    # 5) Model / loss / optimizer / scheduler
    model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    no_improve = 0

    def finite_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask = torch.isfinite(pred) & torch.isfinite(y)
        if mask.sum() == 0:
            return torch.zeros((), device=pred.device)
        return ((pred[mask] - y[mask]) ** 2).mean()

    # 6) Training loop with early stopping on VALID
    for epoch in tqdm(range(epochs), desc="Training (Spark)", unit="epoch"):
        model.train()
        total = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = finite_mse(pred, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total += loss.item() * xb_num.size(0)
        train_loss = total / max(1, len(train_loader.dataset))

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb_num, xb_cat, yb in valid_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                pred = model(xb_num, xb_cat)
                loss = finite_mse(pred, yb)
                total += loss.item() * xb_num.size(0)
        val_loss = total / max(1, len(valid_loader.dataset))

        if np.isfinite(val_loss):
            scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if best_state is None or (np.isfinite(val_loss) and val_loss < best_val):
            best_val = float(val_loss)
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_state)
                break

        print(f"Epoch {epoch+1}/{epochs} | Train {train_loss:.6f} | Val {val_loss:.6f}")

    # 7) Final TEST evaluation (inverse-transform to original scale)
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_num.to(device), X_test_cat.to(device)).cpu().numpy().flatten()
    y_true_scaled = y_test_t.cpu().numpy().flatten()

    y_pred = y_pred_scaled * denom + y_min
    y_true = y_true_scaled * denom + y_min

    predictions = pd.DataFrame({id_col: test_pdf[id_col].reset_index(drop=True),
                                "y_true": y_true, "y_pred": y_pred})

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }

    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols_list,
        "cat_cardinalities": cat_cardinalities,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": epochs,
        "patience": patience,
        "test_size": test_size,
        "valid_size": valid_size,
        "random_state": random_state,
        "y_min": y_min,
        "y_max": y_max,
    }

    return model.state_dict(), metrics, predictions, history, metadata

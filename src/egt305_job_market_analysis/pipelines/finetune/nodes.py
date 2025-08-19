# ======================================================
# Imports
# ======================================================
from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm

# Spark imports (only used in the spark-specific pipeline)
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F, types as T
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

# Global device
device = globals().get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

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


# =========================
# Shared training helper
# =========================
def _torch_fit_for_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float = 0.0,
):
    """Fit a Torch model for a few epochs, with early stopping on validation loss."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 2)
    )

    history = {"train_loss": [], "val_loss": []}
    best_state, best_val, no_improve = None, float("inf"), 0

    for ep in range(epochs):
        # training
        model.train()
        tot = 0.0
        for xb_num, xb_cat, yb in train_loader:
            xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_num, xb_cat)
            loss = criterion(pred, yb)
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tot += loss.item() * xb_num.size(0)
        tr = tot / max(1, len(train_loader.dataset))

        # validation
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb_num, xb_cat, yb in valid_loader:
                xb_num, xb_cat, yb = xb_num.to(device), xb_cat.to(device), yb.to(device)
                pred = model(xb_num, xb_cat)
                loss = criterion(pred, yb)
                tot += loss.item() * xb_num.size(0)
        va = tot / max(1, len(valid_loader.dataset))

        scheduler.step(va)
        history["train_loss"].append(float(tr))
        history["val_loss"].append(float(va))

        if va < best_val:
            best_val, best_state, no_improve = va, model.state_dict(), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                model.load_state_dict(best_state)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model.state_dict(), float(best_val), history


# ======================================
# A) NON-SPARK finetune grid (3 epochs)
# ======================================
def finetune_nn_torch_nonspark_grid(
    data: pd.DataFrame,
    base_state_dict: dict,
    base_metadata: dict,
    param_grid: list[dict],
    *,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    epochs_per_run: int = 3,     # exactly 3 as requested
    patience: int = 2,
    batch_size_default: int = 256,
):
    """
    Re-split data (train/valid/test), then grid-search fine-tune the loaded model
    for a short run (3 epochs per config). Select best by validation loss.
    """
    # 1) split (same as nonspark trainer)
    (X_train, X_valid, X_test,
     y_train, y_valid, y_test,
     job_train, job_valid, job_test) = split_model_data_three_way(
        data=data,
        cat_cols=cat_cols,
        test_size=test_size,
        valid_size=valid_size,
        random_state=random_state,
    )

    # feature meta
    cat_cols_list = list(cat_cols)
    num_cols = [c for c in X_train.columns if c not in cat_cols_list]
    cat_cardinalities = [int(X_train[c].max() + 1) for c in cat_cols_list]

    # scale target with train min/max
    y_min, y_max = float(y_train.min()), float(y_train.max())
    denom = max(1e-12, (y_max - y_min))
    y_train_scaled = (y_train - y_min) / denom
    y_valid_scaled = (y_valid - y_min) / denom
    y_test_scaled  = (y_test  - y_min) / denom

    # to tensors
    def tens(X, y):
        Xn = torch.tensor(X[num_cols].values, dtype=torch.float32) if len(num_cols) else torch.empty((len(X), 0), dtype=torch.float32)
        Xc = torch.tensor(X[cat_cols_list].values, dtype=torch.long) if len(cat_cols_list) else torch.empty((len(X), 0), dtype=torch.long)
        yt = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
        return Xn, Xc, yt

    Xtr_n, Xtr_c, ytr = tens(X_train, y_train_scaled)
    Xva_n, Xva_c, yva = tens(X_valid, y_valid_scaled)
    Xte_n, Xte_c, yte = tens(X_test,  y_test_scaled)

    # loaders factory
    def loaders(batch_size):
        train_loader = DataLoader(TensorDataset(Xtr_n, Xtr_c, ytr), batch_size=batch_size, shuffle=True, pin_memory=(device.type=="cuda"))
        valid_loader = DataLoader(TensorDataset(Xva_n, Xva_c, yva), batch_size=batch_size, shuffle=False, pin_memory=(device.type=="cuda"))
        return train_loader, valid_loader

    # 2) grid search: reload base weights and fine-tune 3 epochs
    best = {"val": float("inf"), "state": None, "params": None, "history": None}
    for params in param_grid:
        lr = params.get("lr", 1e-3)
        bs = params.get("batch_size", batch_size_default)
        wd = params.get("weight_decay", 0.0)

        model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
        model.load_state_dict(base_state_dict)   # warm start

        tr_loader, va_loader = loaders(bs)
        state, vloss, hist = _torch_fit_for_epochs(
            model, tr_loader, va_loader, epochs=epochs_per_run, patience=patience, lr=lr, weight_decay=wd
        )
        if vloss < best["val"]:
            best.update({"val": vloss, "state": state,
                         "params": {"lr":lr, "batch_size":bs, "weight_decay":wd},
                         "history": hist})

    # 3) test eval for best
    best_model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
    best_model.load_state_dict(best["state"])
    best_model.eval()
    with torch.no_grad():
        y_pred_scaled = best_model(Xte_n.to(device), Xte_c.to(device)).cpu().numpy().flatten()
    y_true_scaled = yte.cpu().numpy().flatten()

    y_pred = y_pred_scaled * denom + y_min
    y_true = y_true_scaled * denom + y_min
    predictions = pd.DataFrame({"job_id": job_test, "y_true": y_true, "y_pred": y_pred})

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols_list,
        "cat_cardinalities": cat_cardinalities,
        "grid_best_params": best["params"],
        "epochs_per_run": epochs_per_run,
        "patience": patience,
        "test_size": test_size,
        "valid_size": valid_size,
        "random_state": random_state,
    }
    history = best["history"]
    return best["state"], metrics, predictions, history, metadata


# ==================================
# B) SPARK finetune grid (3 epochs)
# ==================================
def finetune_nn_torch_spark_grid(
    data: SparkDataFrame,
    base_state_dict: dict,
    base_metadata: dict,
    param_grid: list[dict],
    *,
    cat_cols: tuple[str, str] = ("edu_major", "industry_role"),
    target_col: str = "salary_k",
    id_col: str = "job_id",
    test_size: float = 0.2,
    valid_size: float = 0.2,
    random_state: int = 42,
    epochs_per_run: int = 3,     # exactly 3
    patience: int = 2,
    batch_size_default: int = 256,
):
    # 1) Spark preprocess + 3-way split + scale
    (train_pdf, valid_pdf, test_pdf,
     num_cols, cat_cols_list, cat_cardinalities,
     y_min, y_max) = spark_preprocess_and_split_three_way(
        data,
        cat_cols=cat_cols, target_col=target_col, id_col=id_col,
        test_size=test_size, valid_size=valid_size, random_state=random_state
    )
    denom = max(1e-12, (y_max - y_min))

    def to_tensors(pdf: pd.DataFrame):
        Xn = torch.tensor(pdf[num_cols].to_numpy(), dtype=torch.float32) if num_cols else torch.empty((len(pdf), 0), dtype=torch.float32)
        Xc = torch.tensor(pdf[cat_cols_list].to_numpy(), dtype=torch.long) if cat_cols_list else torch.empty((len(pdf), 0), dtype=torch.long)
        y  = torch.tensor(pdf["y_scaled"].to_numpy().reshape(-1, 1), dtype=torch.float32)
        return Xn, Xc, y

    Xtr_n, Xtr_c, ytr = to_tensors(train_pdf)
    Xva_n, Xva_c, yva = to_tensors(valid_pdf)
    Xte_n, Xte_c, yte = to_tensors(test_pdf)

    # clamp cats
    for i, card in enumerate(cat_cardinalities):
        if Xtr_c.shape[1] > i: Xtr_c[:, i].clamp_(0, card - 1)
        if Xva_c.shape[1] > i: Xva_c[:, i].clamp_(0, card - 1)
        if Xte_c.shape[1] > i: Xte_c[:, i].clamp_(0, card - 1)

    def loaders(batch_size):
        train_loader = DataLoader(TensorDataset(Xtr_n, Xtr_c, ytr), batch_size=batch_size, shuffle=True, pin_memory=(device.type=="cuda"))
        valid_loader = DataLoader(TensorDataset(Xva_n, Xva_c, yva), batch_size=batch_size, shuffle=False, pin_memory=(device.type=="cuda"))
        return train_loader, valid_loader

    # 2) grid search on 3-epoch finetunes
    best = {"val": float("inf"), "state": None, "params": None, "history": None}
    for params in param_grid:
        lr = params.get("lr", 1e-3)
        bs = params.get("batch_size", batch_size_default)
        wd = params.get("weight_decay", 0.0)

        model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
        model.load_state_dict(base_state_dict)

        tr_loader, va_loader = loaders(bs)
        state, vloss, hist = _torch_fit_for_epochs(
            model, tr_loader, va_loader,
            epochs=epochs_per_run, patience=patience, lr=lr, weight_decay=wd
        )
        if vloss < best["val"]:
            best.update({"val": vloss, "state": state,
                         "params": {"lr":lr, "batch_size":bs, "weight_decay":wd},
                         "history": hist})

    # 3) test eval for best
    best_model = EmbeddingNNRegressor(len(num_cols), cat_cardinalities).to(device)
    best_model.load_state_dict(best["state"])
    best_model.eval()
    with torch.no_grad():
        y_pred_scaled = best_model(Xte_n.to(device), Xte_c.to(device)).cpu().numpy().flatten()
    y_true_scaled = yte.cpu().numpy().flatten()

    y_pred = y_pred_scaled * denom + y_min
    y_true = y_true_scaled * denom + y_min
    predictions = pd.DataFrame({id_col: test_pdf[id_col].reset_index(drop=True), "y_true": y_true, "y_pred": y_pred})

    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    metadata = {
        "num_cols": num_cols,
        "cat_cols": cat_cols_list,
        "cat_cardinalities": cat_cardinalities,
        "grid_best_params": best["params"],
        "epochs_per_run": epochs_per_run,
        "patience": patience,
        "test_size": test_size,
        "valid_size": valid_size,
        "random_state": random_state,
        "y_min": y_min,
        "y_max": y_max,
    }
    history = best["history"]
    return best["state"], metrics, predictions, history, metadata

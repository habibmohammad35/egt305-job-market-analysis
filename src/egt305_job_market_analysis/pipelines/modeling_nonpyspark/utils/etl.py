import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List
import logging
from functools import wraps

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Used to give a before & after a function
def report_changes(step_name: str):
    """
    Decorator factory that wraps a DataFrame-cleaning function,
    snapshots its input/output, and prints a summary of what changed.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
            # 1) Snapshot before
            before = df.copy()
            cols_before   = set(before.columns)
            shape_before  = before.shape
            nulls_before  = before.isnull().sum().to_dict()

            # 2) Run the actual cleaning function
            result = func(before, *args, **kwargs)

            # 3) Snapshot after
            after        = result.copy()
            cols_after   = set(after.columns)
            shape_after  = after.shape
            nulls_after  = after.isnull().sum().to_dict()

            # 4) Compute diffs
            added_cols   = cols_after - cols_before
            dropped_cols = cols_before - cols_after

            # 5) Print report header
            print(f"\n=== {step_name} ===")
            print(f"Shape: {shape_before} → {shape_after}")

            # 6) Report added/dropped columns
            if added_cols:
                print(f"  + Columns added: {sorted(added_cols)}")
            if dropped_cols:
                print(f"  – Columns dropped: {sorted(dropped_cols)}")

            # 7) Compute null count deltas over the union of all columns
            all_cols   = cols_before | cols_after
            null_deltas = {
                col: nulls_after.get(col, 0) - nulls_before.get(col, 0)
                for col in sorted(all_cols)
                if nulls_after.get(col, 0) != nulls_before.get(col, 0)
            }

            # 8) Report null changes
            if null_deltas:
                print("  * Null counts delta:")
                for col, delta in null_deltas.items():
                    sign = "+" if delta > 0 else ""
                    print(f"      {col}: {sign}{delta}")

            # 9) Print sample row before/after
            print("\n  First row before:")
            print(before.iloc[0].to_frame().T)
            print("\n  First row after:")
            print(after.iloc[0].to_frame().T)

            print("-" * 30)
            return after

        return wrapper
    return decorator
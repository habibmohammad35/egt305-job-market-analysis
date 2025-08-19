import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle # to prevent warnings
import seaborn as sns
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Union
from sklearn.metrics import confusion_matrix # type: ignore
import os

def set_plot_style() -> None:
    """
    Set the plot style for consistent visualizations.
    
    Args:
        None
    Returns:
        None
    """
    plot_style_dict = {
        'font.family': ['Arial', 'sans-serif'],
        'font.sans-serif': ['Arial', 'sans-serif'],
        'axes.facecolor': '#f2f0e8',
        'axes.edgecolor': 'black',
        'axes.labelcolor': '#011547',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'text.color': '#011547',
        'xtick.color': '#011547',
        'ytick.color': '#011547',
        'figure.figsize': (10, 6),
    }
    sns.set_theme(palette="husl", rc=plot_style_dict)
    plt.rcParams.update(plot_style_dict)
    logging.info("Custom plot style set.")

def plot_target_distribution(df, target):
    """
    Plot a vertical bar chart of target value counts with labels on top and padding.
    """
    counts = df[target].value_counts(dropna=False)
    ax = counts.plot(kind="bar", figsize=(6, 4))

    # Add labels on top of bars
    for i, v in enumerate(counts.values):
        ax.text(i, v + (v * 0.01), str(v), ha='center', va='bottom', fontsize=10)

    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    ax.set_xticklabels(counts.index.astype(str), rotation=0)

    # Add padding to top
    ax.set_ylim(top=counts.max() * 1.1)

    plt.tight_layout()
    plt.show()

def plot_numeric_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of all numeric columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        
    Returns:
        None
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)
    
    # Check if there are no numeric columns
    if n == 0:
        print("No numeric columns found in the DataFrame. Skipping plot.")
        return

    # Create subplots: one column, multiple rows
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharey=True)

    # Ensure axes is iterable
    if n == 1:
        axes = [axes]

    # Plot with shared y-axis but individual x-axis for readability in different scales
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], orient='h')
        # Add vertical lines for 1st and 99th percentiles
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        axes[i].axvline(p1, color='red', linestyle='--', label='1st percentile')
        axes[i].axvline(p99, color='green', linestyle='--', label='99th percentile')
        
        axes[i].set_title(f"{col}", loc='left', fontsize=12, fontweight='bold', color='#011547')
        axes[i].set_xlabel("")  
        axes[i].set_ylabel("")  
        axes[i].legend(loc='lower right', ncol=2, fontsize=10, frameon=False)

    # Shared xlabel and title
    fig.suptitle("Boxplot of Numeric Columns", fontsize=14, fontweight='bold', color='#011547')
    fig.supxlabel("Value", fontsize=12, fontweight='bold', color='#011547')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_evaluation(metrics, predictions, history, title_prefix="Model"):
    """
    Visualize model evaluation results.
    
    Args:
        metrics (dict): Dictionary with 'r2', 'mae', 'rmse'.
        predictions (pd.DataFrame): Must have 'y_true' and 'y_pred'.
        history (dict): Must have 'train_loss' and 'val_loss'.
        title_prefix (str): Title prefix for plots.
    """
    # -----------------------
    # 1. Print Metrics
    # -----------------------
    print(f"\n{title_prefix} Performance Metrics")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k.upper():<6}: {v:.4f}")

    # -----------------------
    # 2. Loss Curves
    # -----------------------
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"{title_prefix} Loss Curves")
    plt.legend()

    # -----------------------
    # 3. Predicted vs True
    # -----------------------
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=predictions["y_true"], y=predictions["y_pred"], alpha=0.6)
    max_val = max(predictions["y_true"].max(), predictions["y_pred"].max())
    min_val = min(predictions["y_true"].min(), predictions["y_pred"].min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
    plt.xlabel("True Salary (k)")
    plt.ylabel("Predicted Salary (k)")
    plt.title(f"{title_prefix} Predictions vs True")

    # -----------------------
    # 4. Residuals Plot
    # -----------------------
    plt.subplot(1, 3, 3)
    residuals = predictions["y_true"] - predictions["y_pred"]
    sns.scatterplot(x=predictions["y_true"], y=residuals, alpha=0.6)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("True Salary (k)")
    plt.ylabel("Residual (True - Pred)")
    plt.title(f"{title_prefix} Residuals")

    plt.tight_layout()
    plt.show()

def plot_categorical_distributions(
    df: pd.DataFrame,
    columns: List[str],
    hue: Optional[str] = None,
    max_cols: int = 2,
    figsize: tuple = (14, 6),
):
    """
    Plot count distributions of multiple categorical or discrete columns using subplots.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        columns (List[str]): List of categorical column names to plot.
        hue (Optional[str]): Column to compare categories against (e.g., target variable).
        max_cols (int): Max columns per row in subplot grid.
        figsize (tuple): Size of each subplot figure.

    Returns:
        None (plots displayed)
    """
    num_cols = len(columns)
    ncols = min(num_cols, max_cols)
    nrows = (num_cols + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0], figsize[1] * nrows))
    axes = axes.flatten() if num_cols > 1 else [axes]

    for idx, column in enumerate(columns):
        ax = axes[idx]
        # Ensure column is string-type for consistency
        df[column] = df[column].astype(str)

        # Sorted for consistency
        ordered_vals = sorted(df[column].dropna().unique())

        sns.countplot(data=df, x=column, order=ordered_vals, hue=hue, palette="husl", ax=ax)

        # Annotate bar labels
        for p in ax.patches:
            if isinstance(p, Rectangle):
                height = int(p.get_height())
                ax.annotate(
                    f"{height}",
                    (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        ax.set_title(f"Distribution of '{column}'", fontsize=12)
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)

        if hue is not None:
            ax.legend(title=hue, loc='best')
        else:
            ax.get_legend().remove()

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)): # type: ignore
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

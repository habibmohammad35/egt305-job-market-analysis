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

import matplotlib.pyplot as plt


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

# One function to display metrics and plot all relevent graphs and save to docs folder
def plot_classification_evaluation(
    metrics: dict,
    predictions_df: pd.DataFrame,
    model_name: str = "Model",
    top_n_features: int = 20,
    output_dir: str = "data/08_reporting"
) -> None:
    """
    Display all metrics, confusion matrix, and diagnostic plots (ROC, PRC, Feature Importance),
    and save the plots to disk.

    Args:
        metrics (dict): Model evaluation metrics.
        predictions_df (pd.DataFrame): Must contain 'y_true' and 'y_pred'.
        model_name (str): Name to display on and use for saved plots.
        top_n_features (int): Number of top features to include in the bar chart.
        output_dir (str): Directory to save plots (default is data/08_reporting).
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nMetrics for: {model_name}\n" + "-"*40)
    print(f"Accuracy            : {metrics.get('accuracy', 0):.4f}")
    print(f"ROC AUC             : {metrics.get('roc_auc', 0):.4f}")
    print(f"PRC AUC             : {metrics.get('prc_auc', 0):.4f}")

    print("\nClass-wise Report:")
    class_keys = [
        k for k in metrics['classification_report'].keys()
        if k not in ['accuracy', 'macro avg', 'weighted avg']
    ]
    for label in sorted(class_keys):
        cls = metrics['classification_report'].get(label, {})
        print(f"  Class {label}:")
        print(f"    Precision: {cls.get('precision', 0):.3f} | Recall: {cls.get('recall', 0):.3f} | F1: {cls.get('f1-score', 0):.3f} | Support: {cls.get('support', 0)}")

    print("\nMacro Avg:")
    macro = metrics["classification_report"].get("macro avg", {})
    print(f"    Precision: {macro.get('precision', 0):.3f} | Recall: {macro.get('recall', 0):.3f} | F1: {macro.get('f1-score', 0):.3f}")

    print("\nWeighted Avg:")
    weighted = metrics["classification_report"].get("weighted avg", {})
    print(f"    Precision: {weighted.get('precision', 0):.3f} | Recall: {weighted.get('recall', 0):.3f} | F1: {weighted.get('f1-score', 0):.3f}")

    # Confusion Matrix
    cm = confusion_matrix(predictions_df["y_true"], predictions_df["y_pred"])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_confusion_matrix.png")
    plt.show()

    # PRC, ROC, Feature Importance
    prc = metrics.get("prc_curve", {})
    roc = metrics.get("roc_curve", {})
    feature_importance = pd.Series(metrics.get("feature_importance", {})).sort_values(ascending=False).head(top_n_features)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PRC
    axes[0].plot(prc.get("recall", []), prc.get("precision", []), label=model_name)
    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].grid(True)
    axes[0].legend()

    # ROC
    axes[1].plot(roc.get("fpr", []), roc.get("tpr", []), label=model_name)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].grid(True)
    axes[1].legend()

    # Feature Importance
    feature_importance.plot(kind="barh", ax=axes[2])
    axes[2].invert_yaxis()
    axes[2].set_title(f"Top {top_n_features} Features\n({model_name})")
    axes[2].set_xlabel("Importance (|coefficient| or score)")
    axes[2].grid(True, axis='x')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_summary_plots.png")
    plt.show()

def plot_regression_evaluation(
    metrics: dict,
    predictions_df: pd.DataFrame,
    model_name: str = "Model",
    output_dir: str = "data/08_reporting"
) -> None:
    """
    Display regression metrics and diagnostic plots, save plots to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nMetrics for: {model_name}\n" + "-"*40)
    print(f"R²       : {metrics.get('r2', 0):.4f}")
    print(f"MAE      : {metrics.get('mae', 0):.4f}")
    print(f"MSE      : {metrics.get('mse', 0):.4f}")
    print(f"RMSE     : {metrics.get('rmse', 0):.4f}")

    # Scatter plot of predicted vs actual
    plt.figure(figsize=(6,6))
    plt.scatter(predictions_df["y_true"], predictions_df["y_pred"], alpha=0.3)
    plt.plot([predictions_df["y_true"].min(), predictions_df["y_true"].max()],
             [predictions_df["y_true"].min(), predictions_df["y_true"].max()],
             "r--")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"{model_name} - Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_scatter.png")
    plt.show()

    # Residual plot
    residuals = predictions_df["y_true"] - predictions_df["y_pred"]
    plt.figure(figsize=(6,4))
    plt.hist(residuals, bins=40, edgecolor="k")
    plt.title(f"{model_name} - Residuals")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_residuals.png")
    plt.show()

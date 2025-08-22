# All functions in this file are for VIZ purposes in EDA.ipynb.

#Imports
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from matplotlib.patches import Rectangle
from typing import List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix
import os

# Functions
def set_plot_style() -> None:
    """
    Set the plot style for consistent visualizations.
    
    Args:
        None
    Returns:
        None
    """
    plot_style_dict = {
        'font.family': ['DejaVu Sans', 'sans-serif'],
        'font.sans-serif': ['DejaVu Sans', 'sans-serif'],
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

def plot_categoricals(df, cols):
    """
    Plots count plots for specified categorical columns with counts annotated.
    
    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        cols (list): List of categorical column names to plot.
    """
    for col in cols:
        if col in df.columns:
            plt.figure(figsize=(10, 5))
            ax = sns.countplot(
                data=df,
                x=col,
                order=df[col].value_counts().index,
                palette='viridis'
            )

            # Add counts above each bar with extra padding
            for p in ax.patches:
                ax.annotate(
                    f'{int(p.get_height()):,}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    xytext=(0, 8),  # increased offset for padding
                    textcoords='offset points'
                )

            # Add extra space above tallest bar
            ax.set_ylim(0, ax.get_ylim()[1] * 1.10)

            plt.title(f"Count plot for {col}", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Column '{col}' not found in dataframe.")

def plot_numeric_whisker(df, col):
    """
    Plots a box (whisker) plot for a specified numeric column
    with dotted lines and labels at whiskers and quartiles.
    """
    if col not in df.columns:
        print(f"Column '{col}' not found in DataFrame.")
        return
    
    if not pd.api.types.is_numeric_dtype(df[col]):
        print(f"Column '{col}' is not numeric.")
        return

    set_plot_style()

    # Compute quartiles & whiskers
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    median = df[col].median()
    iqr = q3 - q1
    lower_whisker = max(df[col].min(), q1 - 1.5 * iqr)
    upper_whisker = min(df[col].max(), q3 + 1.5 * iqr)

    # Plot
    plt.figure(figsize=(12, 5))
    ax = sns.boxplot(
        x=df[col],
        color="#69b3a2",
        width=0.4,
        fliersize=4,
        linewidth=1.5
    )
    
    # Add dotted reference lines + labels
    for val, label in [
        (lower_whisker, "Lower whisker"),
        (q1, "Q1"),
        (median, "Median"),
        (q3, "Q3"),
        (upper_whisker, "Upper whisker")
    ]:
        ax.axvline(val, color='black', linestyle='--', alpha=0.6, linewidth=1)
        ax.text(val, 0.05, f"{val:.2f}", transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, color='black', fontweight='bold')

    plt.title(f"Whisker Plot for {col}", fontsize=14, fontweight='bold', color='#011547', pad=15)
    plt.xlabel(col, fontsize=12, fontweight='bold', color='#011547')
    plt.tight_layout()
    plt.show()

def plot_numeric_distribution(df):
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

def plot_bar(df, x_col, y_col, title=None):
    """
    Simple bar plot for categorical vs numeric data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        x_col (str): Column name for categorical variable (x-axis)
        y_col (str): Column name for numeric variable (y-axis)
        title (str, optional): Title of the plot
    """
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x=x_col, y=y_col, palette="husl")
    
    # Add labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom',
                    fontsize=9, color='black', xytext=(0, 3),
                    textcoords='offset points')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.01)  # Add some space above the tallest bar

    
    # Labels and title
    plt.title(title or f"{y_col} by {x_col}", fontsize=14, fontweight='bold', color='#011547')
    plt.xlabel(x_col, fontsize=12, fontweight='bold', color='#011547')
    plt.ylabel(y_col, fontsize=12, fontweight='bold', color='#011547')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_clustered_bars(df, x_col, y_cols, title=None):
    """
    Plot a clustered bar chart for multiple numeric columns grouped by a categorical column.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Categorical column for x-axis.
        y_cols (list): List of numeric column names to plot side by side.
        title (str, optional): Title of the plot.
    """
    # Reshape the dataframe into long format for seaborn
    df_long = df.melt(id_vars=[x_col], value_vars=y_cols, var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_long, x=x_col, y='Value', hue='Metric', palette="husl")
    
    # Add value labels on each bar
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.0f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9, color='black',
                    xytext=(0, 3), textcoords='offset points')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.02)  # Add some space above the tallest bar
    
    plt.title(title or f"{', '.join(y_cols)} by {x_col}", fontsize=14, fontweight='bold', color='#011547')
    plt.xlabel(x_col, fontsize=12, fontweight='bold', color='#011547')
    plt.ylabel("Value (k)", fontsize=12, fontweight='bold', color='#011547')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Metric", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.show()

# Reused previous function for classification evaluation
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

import pandas as pd
import matplotlib.pyplot as plt

def show_model_results(results: dict, feature_names=None, title="Model Results"):
    """
    Nicely display regression results with metrics and optional plots.
    
    Args:
        results (dict): Output dictionary from Kedro node (LR or RF).
        feature_names (list, optional): Names of features (for coefficients/importances).
        title (str): Title for the plots.
    """
    # ======================
    # 1. Print metrics as table
    # ======================
    metrics = {
        "Validation": {
            "RMSE": results.get("val_rmse"),
            "MAE": results.get("val_mae"),
            "R²": results.get("val_r2"),
        },
        "Test": {
            "RMSE": results.get("test_rmse"),
            "MAE": results.get("test_mae"),
            "R²": results.get("test_r2"),
        }
    }
    df_metrics = pd.DataFrame(metrics).T
    display(df_metrics.round(4))

    # ======================
    # 2. Plot metrics
    # ======================
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))  # wider for more spacing

    # Left: RMSE & MAE
    df_err = df_metrics[["RMSE", "MAE"]]
    bars_err = df_err.plot(kind="bar", ax=ax[0])
    ax[0].set_title(f"{title} - Errors (Lower = Better)")
    ax[0].set_ylabel("Error")
    ax[0].grid(True, axis="y")

    # Add values on bars
    for container in bars_err.containers:
        bars_err.bar_label(container, fmt="%.2f", label_type="edge")

    # Right: R²
    bars_r2 = df_metrics[["R²"]].plot(kind="bar", ax=ax[1])
    ax[1].set_title(f"{title} - R² Score (Higher = Better)")
    ax[1].set_ylabel("R²")
    ax[1].set_ylim(0, 1)
    ax[1].grid(True, axis="y")

    for container in bars_r2.containers:
        bars_r2.bar_label(container, fmt="%.2f", label_type="edge")

    plt.subplots_adjust(wspace=0.4)  # more horizontal space between subplots
    plt.show()

    # ======================
    # 3. Plot coefficients or feature importances
    # ======================
    if "coefficients" in results:  # Linear Regression
        coefs = results["coefficients"]
        if feature_names and len(feature_names) == len(coefs):
            coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs})
        else:
            coef_df = pd.DataFrame({"Feature": range(len(coefs)), "Coefficient": coefs})
        ax_coef = coef_df.plot(x="Feature", y="Coefficient", kind="bar", figsize=(10, 5), legend=False)
        ax_coef.set_title(f"{title} - Coefficients")
        ax_coef.axhline(0, color="black", linewidth=0.8)
        ax_coef.set_ylabel("Value")
        ax_coef.grid(True, axis="y")

        # Add values on bars
        for container in ax_coef.containers:
            ax_coef.bar_label(container, fmt="%.2f", label_type="edge")

        plt.show()

    elif "feature_importances" in results:  # Random Forest
        importances = results["feature_importances"]
        if feature_names and len(feature_names) == len(importances):
            imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        else:
            imp_df = pd.DataFrame({"Feature": range(len(importances)), "Importance": importances})
        ax_imp = imp_df.plot(x="Feature", y="Importance", kind="bar", figsize=(10, 5), legend=False)
        ax_imp.set_title(f"{title} - Feature Importances")
        ax_imp.set_ylabel("Importance")
        ax_imp.grid(True, axis="y")

        for container in ax_imp.containers:
            ax_imp.bar_label(container, fmt="%.2f", label_type="edge")

        plt.show()

    # ======================
    # 4. Print intercept if available
    # ======================
    if "intercept" in results:
        print(f"Intercept: {results['intercept']:.4f}")

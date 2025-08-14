# All functions in this file are for VIZ purposes in EDA.ipynb.

#Imports
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from matplotlib.patches import Rectangle
from typing import List, Optional
import numpy as np

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
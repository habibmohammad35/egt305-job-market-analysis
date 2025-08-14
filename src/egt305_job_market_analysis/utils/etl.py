# All functions in this file are for ETL operations using Panda in EDA.ipynb.

# Imports
import pandas as pd

def clean_column_names(df, rename_map=None):
    """
    Standardize column names by lowercasing and applying optional renaming.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    rename_map : dict, optional
        Dictionary mapping existing column names to new shorter names.
        Keys must match the current column names exactly.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned column names.
    """
    df = df.copy()
    
    # Lowercase all column names
    df.columns = df.columns.str.lower()
    
    # Apply rename map if provided
    if rename_map:
        # Ensure rename map keys are lowercase for matching
        rename_map = {k.lower(): v.lower() for k, v in rename_map.items()}
        df.rename(columns=rename_map, inplace=True)
    
    return df

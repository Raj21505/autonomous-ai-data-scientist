"""
Categorical encoding/decoding utility for binary categorical columns.

Handles columns like: yes/no, available/not, present/absent, true/false, etc.
Encodes to 0/1 for model training, decodes predictions back to original category names.
"""

import pandas as pd
import numpy as np


BINARY_CATEGORIES = {
    "yes": ("yes", "no"),
    "no": ("yes", "no"),
    "available": ("available", "not available"),
    "not available": ("available", "not available"),
    "present": ("present", "absent"),
    "absent": ("present", "absent"),
    "true": ("true", "false"),
    "false": ("true", "false"),
    "1": ("1", "0"),
    "0": ("1", "0"),
    "y": ("yes", "no"),
    "n": ("yes", "no"),
}


def classify_categorical_columns(df):
    """
    Classify categorical columns into types:
    - binary_categorical: exactly 2 unique values
    - clustered_categorical: 3-10 unique values
    - fully_categorical: >10 unique values (to be removed)
    
    Returns dict with classifications and mappings.
    """
    classifications = {
        "binary_categorical": {},
        "clustered_categorical": {},
        "fully_categorical": []
    }
    
    for col in df.columns:
        if df[col].dtype in ('object', 'str', 'string', 'category') or str(df[col].dtype).startswith('str'):
            unique_vals = df[col].dropna().unique()
            num_unique = len(unique_vals)
            
            if num_unique == 2:
                # Binary: detect mapping
                val_list = list(unique_vals)
                matched_mapping = None
                for val in val_list:
                    val_lower = str(val).lower().strip()
                    if val_lower in BINARY_CATEGORIES:
                        matched_mapping = BINARY_CATEGORIES[val_lower]
                        break
                
                if matched_mapping:
                    pos_cat, neg_cat = matched_mapping
                    val_strings = [str(v).lower().strip() for v in val_list]
                    if pos_cat.lower() in val_strings:
                        actual_pos = val_list[val_strings.index(pos_cat.lower())]
                        actual_neg = val_list[1 - val_strings.index(pos_cat.lower())]
                    else:
                        actual_pos, actual_neg = val_list[0], val_list[1]
                    classifications["binary_categorical"][col] = (str(actual_pos), str(actual_neg))
                else:
                    # Still binary but no standard mapping, treat as clustered
                    classifications["clustered_categorical"][col] = list(unique_vals)
                    
            elif 3 <= num_unique <= 10:
                classifications["clustered_categorical"][col] = list(unique_vals)
            elif num_unique > 5:
                classifications["fully_categorical"].append(col)
    
    return classifications


def detect_binary_categorical_columns(df):
    """
    Detect binary categorical columns (2 unique non-null values).
    Returns dict: {column_name: (category_1, category_2)}
    """
    categorical_mapping = {}
    
    for col in df.columns:
        # Check if column is string/object/category type
        if df[col].dtype in ('object', 'str', 'string', 'category') or str(df[col].dtype).startswith('str'):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 2:
                # Check if these are known binary categories
                val_list = list(unique_vals)
                
                # Try to match known patterns
                matched_mapping = None
                for val in val_list:
                    val_lower = str(val).lower().strip()
                    if val_lower in BINARY_CATEGORIES:
                        matched_mapping = BINARY_CATEGORIES[val_lower]
                        break
                
                if matched_mapping:
                    # Use the actual values but with consistent mapping
                    pos_cat, neg_cat = matched_mapping
                    # Find which actual value corresponds to positive
                    val_strings = [str(v).lower().strip() for v in val_list]
                    if pos_cat.lower() in val_strings:
                        # Use original casing from data
                        actual_pos = val_list[val_strings.index(pos_cat.lower())]
                        actual_neg = val_list[1 - val_strings.index(pos_cat.lower())]
                    else:
                        actual_pos, actual_neg = val_list[0], val_list[1]
                    categorical_mapping[col] = (str(actual_pos), str(actual_neg))
    
    return categorical_mapping


def encode_categorical_columns(df, categorical_mapping=None):
    """
    Encode binary categorical columns to 0/1.
    If categorical_mapping is None, auto-detect it.
    
    Returns: (encoded_df, categorical_mapping)
    """
    df_encoded = df.copy()
    
    if categorical_mapping is None:
        categorical_mapping = detect_binary_categorical_columns(df_encoded)
    
    for col, (positive_cat, negative_cat) in categorical_mapping.items():
        if col in df_encoded.columns:
            # Convert column values to lowercase for comparison
            df_encoded[col] = df_encoded[col].fillna("").astype(str).str.lower().str.strip()
            
            # Encode: positive_cat -> 1, negative_cat -> 0
            df_encoded[col] = df_encoded[col].apply(
                lambda x: 1 if x == positive_cat.lower() else (0 if x == negative_cat.lower() else np.nan)
            )
            
            # Convert to numeric, coercing errors to NaN
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    
    return df_encoded, categorical_mapping


def decode_predictions(predictions, categorical_mapping):
    """
    Decode predictions back to categorical format.
    
    Args:
        predictions: dict with column names and predicted values
        categorical_mapping: dict from encode_categorical_columns
    
    Returns: dict with decoded categorical values
    """
    decoded = predictions.copy()
    
    for col, (positive_cat, negative_cat) in categorical_mapping.items():
        if col in decoded:
            val = decoded[col]
            if pd.isna(val):
                decoded[col] = None
            elif val >= 0.5:  # Threshold for binary classification
                decoded[col] = positive_cat
            else:
                decoded[col] = negative_cat
    
    return decoded


def get_categorical_info(df):
    """
    Get info about detected categorical columns for API response.
    Returns dict with column classifications.
    """
    classifications = classify_categorical_columns(df)
    
    info = {
        "binary_categorical": [],
        "clustered_categorical": [],
        "fully_categorical": classifications["fully_categorical"]
    }
    
    for col, (positive_cat, negative_cat) in classifications["binary_categorical"].items():
        info["binary_categorical"].append({
            "column": col,
            "positive_class": positive_cat,
            "negative_class": negative_cat,
            "encoding": {"1": positive_cat, "0": negative_cat}
        })
    
    for col, values in classifications["clustered_categorical"].items():
        info["clustered_categorical"].append({
            "column": col,
            "values": values
        })
    
    return info

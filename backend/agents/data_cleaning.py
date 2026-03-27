import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def detect_column_type(series):
    """Intelligently detect column type: numeric, categorical, datetime, or boolean."""
    # Remove NaN values for analysis
    non_null = series.dropna()
    
    if len(non_null) == 0:
        return 'unknown'
    
    # Check for boolean
    unique_vals = set(non_null.unique())
    if unique_vals.issubset({0, 1, True, False, 'True', 'False', 'true', 'false', 'yes', 'no', 'Yes', 'No', 'YES', 'NO'}):
        if len(unique_vals) <= 2:
            return 'boolean'
    
    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    
    # Try to infer datetime format
    if series.dtype == 'object':
        sample = non_null.iloc[0]
        try:
            pd.to_datetime(sample)
            # Check if all values look like dates
            if non_null.astype(str).apply(lambda x: _is_datetime_format(x)).sum() / len(non_null) > 0.8:
                return 'datetime'
        except:
            pass
    
    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        return 'numeric'
    
    try:
        pd.to_numeric(non_null)
        return 'numeric'
    except:
        pass
    
    # Default to categorical
    return 'categorical'


def _is_datetime_format(val):
    """Check if a string looks like a datetime."""
    str_val = str(val).strip()
    datetime_patterns = [
        r'^\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'^\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'^\d{1,2}-\d{1,2}-\d{4}',  # M-D-YYYY
    ]
    import re
    for pattern in datetime_patterns:
        if re.match(pattern, str_val):
            return True
    return False


def has_high_variation_outliers(col):
    """Check if column has many outliers (high variation)."""
    try:
        # Calculate IQR
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return False
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check percentage of outliers
        outlier_count = ((col < lower_bound) | (col > upper_bound)).sum()
        outlier_ratio = outlier_count / len(col)
        
        # If >10% outliers, consider it high variation
        return outlier_ratio > 0.1
    except:
        return False


def handle_missing_values(df, col_type, col_name, col, total_rows):
    """Impute missing values based on column type.

    Note: column dropping rules are handled separately in clean_dataset to preserve
    explicit priority order.
    """
    miss_count = int(col.isnull().sum())

    if miss_count == 0:
        return col, {"method": "none", "missing": 0}
    
    # Handle based on type
    if col_type == 'numeric':
        # Use median if column has many outliers/high variation, otherwise mean
        try:
            has_outliers = has_high_variation_outliers(col.dropna())
            
            if has_outliers:
                fill_value = col.median()
                method = "median (high variation detected)"
            else:
                fill_value = col.mean()
                method = "mean"
            
            if pd.isna(fill_value):
                fill_value = col.median()
                method = "median"
            
            col = col.fillna(fill_value)
        except:
            col = col.fillna(col.median())
            method = "median"
    
    elif col_type == 'datetime':
        # Forward fill for datetime
        col = col.fillna(method='ffill')
        if col.isnull().sum() > 0:
            col = col.fillna(method='bfill')
        method = "ffill/bfill"
    
    elif col_type == 'categorical':
        # Use mode or "Unknown"
        try:
            mode_val = col.mode(dropna=True)
            if len(mode_val) > 0:
                fill_value = mode_val.iloc[0]
            else:
                fill_value = "Unknown"
        except:
            fill_value = "Unknown"
        col = col.fillna(fill_value)
        method = f"mode: {fill_value}"
    
    else:  # boolean
        # Forward fill or most common value
        try:
            mode_val = col.mode(dropna=True)
            if len(mode_val) > 0:
                fill_value = mode_val.iloc[0]
            else:
                fill_value = False
        except:
            fill_value = False
        col = col.fillna(fill_value)
        method = f"mode: {fill_value}"
    
    return col, {"method": method, "missing": miss_count}


def get_high_missing_columns(df, threshold=0.4):
    """Return columns with missing ratio >= threshold."""
    cols = []
    for col in df.columns:
        miss_ratio = float(df[col].isnull().sum()) / max(1, len(df))
        if miss_ratio >= threshold:
            cols.append(col)
    return cols


def get_rows_missing_ratio(df):
    """Return ratio of rows containing at least one missing value."""
    rows_with_missing = int(df.isnull().any(axis=1).sum())
    ratio = rows_with_missing / max(1, len(df))
    return rows_with_missing, ratio


def convert_to_datetime(col):
    """Attempt to convert column to datetime."""
    try:
        return pd.to_datetime(col, errors='coerce')
    except:
        return col


def handle_outliers_iqr(col, col_type):
    """Detect and handle outliers using IQR method."""
    if col_type != 'numeric':
        return col, {"outliers": 0, "action": "none"}
    
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier_mask = (col < lower_bound) | (col > upper_bound)
    outlier_count = outlier_mask.sum()
    
    if outlier_count > 0:
        # Winsorize: cap outliers to bounds
        col = col.clip(lower=lower_bound, upper=upper_bound)
        return col, {"outliers": int(outlier_count), "action": "winsorized", "bounds": (float(lower_bound), float(upper_bound))}
    
    return col, {"outliers": 0, "action": "none"}


def handle_invalid_values(col, col_type, col_name):
    """Remove or fix invalid domain-specific values."""
    invalid_count = 0
    
    # Define domain rules based on column name patterns
    if col_type == 'numeric':
        # Age validation
        if 'age' in col_name.lower():
            before = col.shape[0]
            col = col[(col >= 0) & (col <= 120)]
            invalid_count = before - col.shape[0]
        
        # Salary/Income validation
        elif any(x in col_name.lower() for x in ['salary', 'income', 'revenue']):
            before = col.shape[0]
            col = col[col >= 0]
            invalid_count = before - col.shape[0]
        
        # Percentage validation
        elif any(x in col_name.lower() for x in ['percent', 'ratio', 'rate']):
            before = col.shape[0]
            col = col[(col >= 0) & (col <= 100)]
            invalid_count = before - col.shape[0]
    
    return col, {"invalid_removed": invalid_count}


def standardize_format(col, col_type):
    """Standardize data formats."""
    if col_type == 'categorical':
        # Lowercase and strip whitespace
        col = col.astype(str).str.lower().str.strip()
    
    elif col_type == 'datetime':
        # Standardize to YYYY-MM-DD
        try:
            col = pd.to_datetime(col).dt.strftime('%Y-%m-%d')
        except:
            pass
    
    return col


def check_consistency(df):
    """Check for logical consistency between columns."""
    issues = []
    
    # Check date consistency (start_date <= end_date)
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ['start', 'end']) and 'date' in c.lower()]
    if len(date_cols) >= 2:
        for i in range(len(date_cols) - 1):
            try:
                start = pd.to_datetime(df[date_cols[i]], errors='coerce')
                end = pd.to_datetime(df[date_cols[i+1]], errors='coerce')
                inconsistent = (start > end).sum()
                if inconsistent > 0:
                    issues.append(f"Date inconsistency in {date_cols[i]} vs {date_cols[i+1]}: {inconsistent} rows")
            except:
                pass
    
    # Check quantity >= 0
    qty_cols = [c for c in df.columns if 'quantity' in c.lower() or 'count' in c.lower()]
    for col in qty_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                issues.append(f"Negative values found in {col}: {negative_count} rows")
    
    return issues


def handle_rare_categories(col, col_type, threshold=0.01):
    """Replace rare categories (frequency < threshold) with 'Other'."""
    if col_type != 'categorical':
        return col, {"rare_replaced": 0}
    
    value_counts = col.value_counts(normalize=True)
    rare_cats = value_counts[value_counts < threshold].index
    
    rare_count = (col.isin(rare_cats)).sum()
    if rare_count > 0:
        col = col.replace(rare_cats, "Other")
    
    return col, {"rare_replaced": int(rare_count), "threshold": threshold}


def drop_irrelevant_columns(df):
    """Drop constant columns, irrelevant ID columns, and low-impact columns like names."""
    cols_to_drop = []
    
    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    cols_to_drop.extend(constant_cols)
    
    # Drop irrelevant ID columns (high cardinality with mostly unique values)
    for col in df.columns:
        if any(x in col.lower() for x in ['id', 'uuid', 'guid', 'pk']):
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.95:  # 95%+ unique values
                cols_to_drop.append(col)
    
    # Drop low-impact columns like names, descriptions, etc.
    low_impact_keywords = ['name', 'description', 'notes', 'comments', 'remarks', 'email', 'phone', 'address']
    for col in df.columns:
        col_lower = col.lower()
        for keyword in low_impact_keywords:
            if keyword in col_lower:
                cols_to_drop.append(col)
                break
    
    cols_to_drop = list(set(cols_to_drop))
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df, cols_to_drop


def handle_unique_value_columns(df, col_type_dict):
    """Remove rows with missing values in unique-value columns (ID, roll no, etc).
    
    These columns should have no missing values as they are identifiers."""
    rows_before = len(df)
    cols_with_missing_in_unique = []
    
    for col in df.columns:
        if any(x in col.lower() for x in ['id', 'roll', 'code', 'number', 'serial', 'reference']):
            # Check if this column has mostly unique values
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.8:  # Mostly unique
                # Check for missing values
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    cols_with_missing_in_unique.append({
                        'column': col,
                        'missing_count': int(missing_count),
                        'rows_to_remove': int(missing_count)
                    })
                    # Remove rows with missing values in this column
                    df = df[df[col].notna()]
    
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    return df, rows_removed, cols_with_missing_in_unique


def handle_rows_with_low_missing_ratio(df, threshold=0.05):
    """Remove rows that have missing values if those rows are less than threshold% of total.
    
    If fewer than threshold% of rows have any missing values, remove those rows entirely."""
    rows_before = len(df)
    rows_with_any_missing = df[df.isnull().any(axis=1)]
    
    rows_with_missing_count = len(rows_with_any_missing)
    rows_with_missing_ratio = rows_with_missing_count / max(1, len(df))
    
    removed_count = 0
    if rows_with_missing_ratio < threshold:
        # Remove rows with missing values
        df = df.dropna().reset_index(drop=True)
        removed_count = rows_before - len(df)
    
    return df, removed_count, rows_with_missing_ratio


def normalize_categorical_values(series, col):
    """Standardize categorical values (e.g., Male, M, male -> male)."""
    if not pd.api.types.is_object_dtype(series):
        return series
    
    # Mapping for common variations
    mappings = {
        'male': ['m', 'M', 'Male', 'MALE', 'male'],
        'female': ['f', 'F', 'Female', 'FEMALE', 'female'],
        'yes': ['y', 'Y', 'Yes', 'YES', 'yes', '1', 'true', 'True', 'TRUE'],
        'no': ['n', 'N', 'No', 'NO', 'no', '0', 'false', 'False', 'FALSE'],
    }
    
    # Create reverse mapping
    replace_dict = {}
    for canonical, variants in mappings.items():
        for variant in variants:
            replace_dict[variant] = canonical
    
    # Apply mapping to actual values in column
    return series.replace(replace_dict)


def clean_dataset(df: pd.DataFrame, scale_numeric=False):
    """
    Professionally clean dataset with intelligent rules:
    
    1. Detect column types (numeric, categorical, datetime, boolean)
    2. Remove duplicate rows
     3. Handle unique-value columns (ID, roll no) - remove rows with missing values
     4. Priority condition check:
         - if any column has >40% missing and rows-with-missing <5%, remove rows first
     5. Recheck >40% rule and drop those columns
     6. Fill remaining missing values by type:
       - Categorical: use mode
       - Numeric with high variation/outliers: use median
       - Numeric with low variation: use mean
    7. Convert data types
    8. Standardize formats (lowercase, whitespace, dates)
    9. Detect and handle outliers using IQR (winsorization)
    10. Remove/fix invalid domain-specific values
    11. Handle rare categories
    12. Drop constant and irrelevant columns (IDs, names, descriptions, etc.)
    13. Check logical consistency
    
    Returns: (cleaned_df, report)
    """
    report = {
        "initial_shape": df.shape,
        "column_types": {},
        "duplicates_removed": 0,
        "unique_col_rows_removed": 0,
        "unique_col_missing_details": [],
        "low_missing_rows_removed": 0,
        "low_missing_ratio": 0,
        "priority_rows_first_applied": False,
        "columns_removed": [],
        "missing_handled": {},
        "outliers_handled": {},
        "invalid_values_handled": {},
        "rare_categories_handled": {},
        "consistency_issues": [],
        "final_shape": None,
        "summary": []
    }
    
    df = df.reset_index(drop=True)
    total_rows = len(df)
    
    # Step 1: Detect column types
    for col in df.columns:
        report["column_types"][col] = detect_column_type(df[col])
    
    # Step 2: Remove exact duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = dup_count
    
    # Step 3: Handle missing values in unique-value columns (ID, roll no, etc.)
    # If these columns have missing values, remove those rows
    df, unique_rows_removed, unique_missing_details = handle_unique_value_columns(df, report["column_types"])
    report["unique_col_rows_removed"] = unique_rows_removed
    report["unique_col_missing_details"] = unique_missing_details
    
    # Step 4: Priority check
    # If both conditions are true, remove rows with missing first:
    # 1) at least one column has >=40% missing
    # 2) rows with missing are <5%
    high_missing_cols_before = get_high_missing_columns(df, threshold=0.4)
    _, row_missing_ratio_before = get_rows_missing_ratio(df)
    if high_missing_cols_before and row_missing_ratio_before < 0.05:
        rows_before = len(df)
        df = df.dropna().reset_index(drop=True)
        report["low_missing_rows_removed"] = rows_before - len(df)
        report["low_missing_ratio"] = float(row_missing_ratio_before)
        report["priority_rows_first_applied"] = True

    # Step 5: Recheck and drop columns with >=40% missing
    high_missing_cols_after = get_high_missing_columns(df, threshold=0.4)
    if high_missing_cols_after:
        for col in high_missing_cols_after:
            miss_count = int(df[col].isnull().sum())
            miss_ratio = miss_count / max(1, len(df))
            report["columns_removed"].append({
                col: {
                    "reason": "missing_values",
                    "details": {
                        "method": "dropped",
                        "missing": miss_count,
                        "reason": ">=40% missing",
                        "ratio": float(miss_ratio),
                    },
                }
            })
        df = df.drop(columns=high_missing_cols_after)

    # Step 6: Fill remaining missing values using type-aware imputation
    for col in df.columns:
        col_type = report["column_types"].get(col, detect_column_type(df[col]))
        col_data, missing_info = handle_missing_values(df, col_type, col, df[col], total_rows)
        df[col] = col_data
        report["missing_handled"][col] = missing_info
    
    # Step 7: Convert data types
    for col in df.columns:
        col_type = report["column_types"][col]
        if col_type == 'datetime':
            df[col] = convert_to_datetime(df[col])
        elif col_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
    
    # Step 8: Standardize formats
    for col in df.columns:
        col_type = report["column_types"][col]
        df[col] = standardize_format(df[col], col_type)
        df[col] = normalize_categorical_values(df[col], col)
    
    # Step 9: Handle outliers
    for col in df.columns:
        col_type = report["column_types"][col]
        if col_type == 'numeric':
            df[col], outlier_info = handle_outliers_iqr(df[col], col_type)
            if outlier_info["outliers"] > 0:
                report["outliers_handled"][col] = outlier_info
    
    # Step 10: Handle invalid values
    for col in df.columns:
        col_type = report["column_types"][col]
        if col_type == 'numeric':
            df[col], invalid_info = handle_invalid_values(df[col], col_type, col)
            if invalid_info["invalid_removed"] > 0:
                report["invalid_values_handled"][col] = invalid_info
    
    # Step 11: Handle rare categories
    for col in df.columns:
        col_type = report["column_types"][col]
        if col_type == 'categorical':
            df[col], rare_info = handle_rare_categories(df[col], col_type)
            if rare_info["rare_replaced"] > 0:
                report["rare_categories_handled"][col] = rare_info
    
    # Step 12: Drop irrelevant columns
    df, irrelevant_cols = drop_irrelevant_columns(df)
    if irrelevant_cols:
        report["columns_removed"].extend([{col: {"reason": "irrelevant/constant"}} for col in irrelevant_cols])
    
    # Step 13: Check consistency
    consistency_issues = check_consistency(df)
    report["consistency_issues"] = consistency_issues
    
    # Step 14: Apply scaling if needed
    if scale_numeric:
        from sklearn.preprocessing import StandardScaler
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            report["scaling_applied"] = True
    
    # Remove any remaining rows with missing values
    rows_before = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    report["final_rows_removed_missing"] = rows_removed
    
    report["final_shape"] = df.shape
    
    # Generate summary
    report["summary"] = [
        f"Initial shape: {report['initial_shape']} → Final shape: {report['final_shape']}",
        f"Duplicates removed: {report['duplicates_removed']}",
        f"Rows removed (missing in ID columns): {report['unique_col_rows_removed']}",
        f"Priority rows-first rule applied: {report['priority_rows_first_applied']}",
        f"Rows removed (low missing ratio <5% and >40% col missing present): {report['low_missing_rows_removed']}",
        f"Columns removed (>40% missing / irrelevant): {len(report['columns_removed'])}",
        f"Missing values filled intelligently in {len(report['missing_handled'])} columns",
        f"Outliers detected and handled in {len(report['outliers_handled'])} columns",
        f"Invalid values removed: {len(report['invalid_values_handled'])} columns",
        f"Rare categories handled: {len(report['rare_categories_handled'])} columns",
    ]
    
    return df, report

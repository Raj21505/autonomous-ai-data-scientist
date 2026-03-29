#backend/agents/data_understanding.py

def analyze_dataset(df, target=None):
    """Analyze the dataframe and determine a target column.

    If `target` is provided and is a valid column it will be used and
    marked as confirmed. Otherwise an inferred target is chosen from
    columns with low cardinality (<= 10 unique values). The function
    returns an `analysis` dict that includes both `inferred_target`
    and `target_confirmed` so callers (or a frontend) can ask the user
    to confirm or override the target.
    """

    analysis = {}

    analysis["rows"] = df.shape[0]
    analysis["columns"] = df.shape[1]
    analysis["column_names"] = list(df.columns)

    numeric = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

    analysis["numeric_features"] = numeric
    analysis["categorical_features"] = categorical

    inferred = None
    for col in df.columns:
        if df[col].nunique() <= 10:
            inferred = col
            break

    # If user supplied a target and it exists, use it and mark confirmed
    if target and target in df.columns:
        chosen = target
        confirmed = True
    else:
        chosen = inferred
        confirmed = False if inferred else False

    analysis["inferred_target"] = inferred
    analysis["target_column"] = chosen
    analysis["target_confirmed"] = confirmed

    if chosen:
        unique = df[chosen].nunique()
        if unique == 2:
            analysis["problem_type"] = "Binary Classification"
        elif unique > 2:
            analysis["problem_type"] = "Multi-class Classification"
        else:
            analysis["problem_type"] = "Regression"
    else:
        analysis["problem_type"] = "Unknown"

    return analysis

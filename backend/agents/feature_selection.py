import pandas as pd


def target_correlation(df, target):
    correlations = {}

    for col in df.columns:
        if col != target and pd.api.types.is_numeric_dtype(df[col]):
            corr = df[col].corr(df[target])
            correlations[col] = abs(corr) if corr is not None else 0

    correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
    return correlations


def remove_low_importance(correlations, threshold=0.05):
    removed = [col for col, val in correlations.items() if val < threshold]
    return removed

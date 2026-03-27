"""
Dynamic interactive dashboard schema generator.
Builds dataset-specific KPI/cards/charts configuration for frontend Plotly rendering.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


def _top_feature_importance(trainer: Any, top_k: int = 10) -> List[Dict[str, Any]]:
    if trainer is None:
        return []
    try:
        items = trainer.get_feature_importance().items()
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)[:top_k]
        return [{"feature": str(k), "importance": _safe_float(v)} for k, v in sorted_items]
    except Exception:
        return []


def _safe_series_float(values: pd.Series, limit: int = 800) -> List[float]:
    return [_safe_float(v) for v in values.dropna().astype(float).head(limit).tolist()]


def _dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def _is_identifier_like(column_name: str) -> bool:
    lowered = str(column_name).lower()
    return any(token in lowered for token in ["id", "uuid", "code"])


def _build_distribution_chart(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        return {
            "id": "distribution",
            "title": f"Distribution: {col}",
            "type": "histogram",
            "span": 2,
            "x": _safe_series_float(df[col]),
            "x_title": col,
            "y_title": "Count",
        }

    first_col = df.columns[0]
    counts = df[first_col].astype(str).value_counts(dropna=False).head(10)
    return {
        "id": "distribution",
        "title": f"Distribution: {first_col}",
        "type": "bar",
        "x": [str(v) for v in counts.index.tolist()],
        "y": [int(v) for v in counts.values.tolist()],
        "x_title": first_col,
        "y_title": "Count",
    }


def _find_category_share_column(df: pd.DataFrame) -> str | None:
    keywords = [
        "country",
        "state",
        "city",
        "region",
        "category",
        "segment",
        "department",
        "group",
        "type",
    ]

    categorical_cols = [
        c
        for c in df.columns
        if str(df[c].dtype) in {"object", "category", "bool"} and not _is_identifier_like(c)
    ]

    for col in categorical_cols:
        lowered = str(col).lower()
        if any(key in lowered for key in keywords):
            distinct = int(df[col].nunique(dropna=False))
            if 2 <= distinct <= 20:
                return col

    for col in categorical_cols:
        distinct = int(df[col].nunique(dropna=False))
        if 2 <= distinct <= 20:
            return col

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if _is_identifier_like(col):
            continue
        distinct = int(df[col].nunique(dropna=True))
        if 2 <= distinct <= 10:
            return col

    return None


def _build_category_share_or_fallback_chart(df: pd.DataFrame) -> Dict[str, Any]:
    category_col = _find_category_share_column(df)
    if category_col:
        counts = df[category_col].astype(str).value_counts(dropna=False).head(8)
        return {
            "id": "category_share",
            "title": f"Category Share: {category_col}",
            "type": "pie",
            "labels": [str(x) for x in counts.index.tolist()],
            "values": [int(x) for x in counts.values.tolist()],
        }

    numeric_cols = [c for c in df.select_dtypes(include=["number"]).columns.tolist() if not _is_identifier_like(c)]
    if numeric_cols:
        col = numeric_cols[0]
        return {
            "id": "fallback_numeric_spread",
            "title": f"Spread: {col}",
            "type": "box",
            "y": _safe_series_float(df[col]),
            "x_title": "",
            "y_title": col,
        }

    col = df.columns[0]
    counts = df[col].astype(str).value_counts(dropna=False).head(10)
    return {
        "id": "fallback_top_values",
        "title": f"Top Values: {col}",
        "type": "bar",
        "x": [str(v) for v in counts.index.tolist()],
        "y": [int(v) for v in counts.values.tolist()],
        "x_title": col,
        "y_title": "Count",
    }


def _build_correlation_chart(df: pd.DataFrame, target: str) -> Dict[str, Any] | None:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return None

    corr = numeric_df.corr(numeric_only=True)
    # Use a compact heatmap to avoid another bar chart and improve visual variety.
    cols = corr.columns.tolist()[:8]
    z = corr.loc[cols, cols].fillna(0.0).values.tolist()

    return {
        "id": "corr_heatmap",
        "title": "Correlation Heatmap",
        "type": "heatmap",
        "span": 2,
        "height": 300,
        "x": [str(c) for c in cols],
        "y": [str(c) for c in cols],
        "z": z,
    }


def _build_correlation_or_fallback_chart(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    corr_chart = _build_correlation_chart(df, target)
    if corr_chart:
        return corr_chart

    missing_pct = round((_safe_float(df.isna().sum().sum()) / max(df.shape[0] * df.shape[1], 1)) * 100, 2)
    duplicate_pct = round((_safe_float(df.duplicated().sum()) / max(df.shape[0], 1)) * 100, 2)
    return {
        "id": "quality_heatmap",
        "title": "Heatmap: Quality Overview",
        "type": "heatmap",
        "x": ["Missing %", "Duplicate %"],
        "y": ["Dataset"],
        "z": [[missing_pct, duplicate_pct]],
    }


def _build_feature_importance_chart(df: pd.DataFrame, top_features: List[Dict[str, Any]]) -> Dict[str, Any]:
    if top_features:
        return {
            "id": "feature_importance",
            "title": "Top Feature Importance",
            "type": "bar",
            "x": [str(x["feature"]) for x in top_features],
            "y": [_safe_float(x["importance"]) for x in top_features],
            "orientation": "h",
            "x_title": "Feature",
            "y_title": "Importance",
        }

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] >= 1:
        variance = numeric_df.var(numeric_only=True).sort_values(ascending=False).head(10)
        return {
            "id": "feature_importance_fallback",
            "title": "Top Feature Importance (Variance Proxy)",
            "type": "bar",
            "x": [str(c) for c in variance.index.tolist()],
            "y": [_safe_float(v) for v in variance.values.tolist()],
            "orientation": "h",
            "x_title": "Feature",
            "y_title": "Variance",
        }

    counts = df.nunique(dropna=False).sort_values(ascending=False).head(10)
    return {
        "id": "feature_importance_fallback",
        "title": "Top Feature Importance (Cardinality Proxy)",
        "type": "bar",
        "x": [str(c) for c in counts.index.tolist()],
        "y": [_safe_float(v) for v in counts.values.tolist()],
        "orientation": "h",
        "x_title": "Feature",
        "y_title": "Distinct Values",
    }


def _build_model_comparison_chart(results: Dict[str, Any], task_type: str) -> Dict[str, Any]:
    comparison = results.get("comparison", []) or []
    metric = "f1_score"

    if comparison:
        return {
            "id": "model_comparison",
            "title": f"Model Comparison ({metric})",
            "type": "line",
            "span": 3,
            "height": 300,
            "x": [str(m.get("model", "")) for m in comparison],
            "y": [_safe_float(m.get(metric, 0.0)) for m in comparison],
            "x_title": "Model",
            "y_title": metric,
        }

    best_model = str(results.get("best_model", "Best Model"))
    score = _safe_float((results.get("metrics", {}) or {}).get(metric, 0.0))
    return {
        "id": "model_comparison_fallback",
        "title": f"Model Comparison ({metric})",
        "type": "bar",
        "x": [best_model],
        "y": [score],
        "x_title": "Model",
        "y_title": metric,
    }


def _build_scatter_chart(df: pd.DataFrame, target: str) -> Dict[str, Any] | None:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        return None

    x_col = numeric_cols[0]
    y_col = target if target in numeric_cols and target != x_col else numeric_cols[1]

    sample = df[[x_col, y_col]].dropna().head(800)
    if sample.empty:
        return None

    return {
        "id": "scatter_relation",
        "title": f"Relationship: {x_col} vs {y_col}",
        "type": "scatter",
        "x": [_safe_float(v) for v in sample[x_col].tolist()],
        "y": [_safe_float(v) for v in sample[y_col].tolist()],
        "x_title": x_col,
        "y_title": y_col,
    }


def _build_relationship_or_fallback_chart(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    relation_chart = _build_scatter_chart(df, target)
    if relation_chart:
        return relation_chart

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        col = numeric_cols[0]
        values = _safe_series_float(df[col])
        return {
            "id": "relationship_fallback_line",
            "title": f"Relationship: Index vs {col}",
            "type": "line",
            "x": list(range(len(values))),
            "y": values,
            "x_title": "Index",
            "y_title": col,
        }

    col = df.columns[0]
    counts = df[col].astype(str).value_counts(dropna=False).head(10)
    return {
        "id": "relationship_fallback_bar",
        "title": f"Relationship: {col} Frequency",
        "type": "bar",
        "x": [str(v) for v in counts.index.tolist()],
        "y": [int(v) for v in counts.values.tolist()],
        "x_title": col,
        "y_title": "Count",
    }


def _build_missing_data_chart(df: pd.DataFrame) -> Dict[str, Any]:
    missing = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    present = max(total_cells - missing, 0)
    return {
        "id": "missing_data",
        "title": "Data Completeness",
        "type": "pie",
        "labels": ["Present", "Missing"],
        "values": [present, missing],
    }


def _build_numeric_summary(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return {"available": False, "message": "No numeric columns found"}

    # Use the numeric column with the most non-null values for a stable summary card.
    best_col = max(numeric_cols, key=lambda c: int(df[c].notna().sum()))
    series = df[best_col].dropna().astype(float)
    if series.empty:
        return {
            "available": False,
            "message": "Numeric columns exist but have no usable values",
        }

    return {
        "available": True,
        "column": str(best_col),
        "numeric_columns": int(len(numeric_cols)),
        "count": int(series.shape[0]),
        "mean": round(_safe_float(series.mean()), 6),
        "median": round(_safe_float(series.median()), 6),
        "std": round(_safe_float(series.std()), 6),
        "min": round(_safe_float(series.min()), 6),
        "max": round(_safe_float(series.max()), 6),
        "q1": round(_safe_float(series.quantile(0.25)), 6),
        "q3": round(_safe_float(series.quantile(0.75)), 6),
    }


def generate_dashboard_schema(
    df: pd.DataFrame,
    target: str,
    results: Dict[str, Any],
    trainer: Any,
) -> Dict[str, Any]:
    """Generate dashboard schema for dataset-specific interactive frontend rendering."""
    summary = _dataset_summary(df)
    task_type = str(results.get("preparation", {}).get("task_type", "classification"))
    best_model = str(results.get("best_model", "Unknown"))
    best_metrics = results.get("metrics", {}) or {}

    top_features = _top_feature_importance(trainer)

    kpis = [
        {"label": "Rows", "value": summary["rows"]},
        {"label": "Columns", "value": summary["columns"]},
        {"label": "Missing", "value": summary["missing_values"]},
        {"label": "Best Model", "value": best_model},
    ]

    if task_type == "classification":
        kpis.append({"label": "Accuracy", "value": round(_safe_float(best_metrics.get("accuracy")) * 100, 2), "suffix": "%"})
        kpis.append({"label": "F1", "value": round(_safe_float(best_metrics.get("f1_score")), 4)})
    else:
        kpis.append({"label": "R2", "value": round(_safe_float(best_metrics.get("r2_score")), 4)})
        kpis.append({"label": "RMSE", "value": round(_safe_float(best_metrics.get("rmse")), 4)})

    charts: List[Dict[str, Any]] = [
        _build_feature_importance_chart(df, top_features),
        _build_model_comparison_chart(results, task_type),
        _build_relationship_or_fallback_chart(df, target),
        _build_distribution_chart(df),
        _build_correlation_or_fallback_chart(df, target),
        _build_missing_data_chart(df),
        _build_category_share_or_fallback_chart(df),
    ]

    return {
        "title": "AI Insights Dashboard",
        "target": target,
        "task_type": task_type,
        "dataset_summary": summary,
        "numeric_summary": _build_numeric_summary(df),
        "kpis": kpis,
        "charts": charts,
    }

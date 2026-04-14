from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from uuid import uuid4
import pandas as pd
import numpy as np
import json
import base64
from pathlib import Path

from backend.state import datasets, trained_models
from backend.utils.file_loader import load_csv
from backend.utils.categorical_encoder import (
    detect_binary_categorical_columns,
    encode_categorical_columns,
    decode_predictions,
    get_categorical_info,
)
from backend.agents.eda_agent import run_eda
from backend.agents.data_understanding import analyze_dataset
from backend.agents.data_cleaning import clean_dataset
from backend.agents.feature_selection import target_correlation, remove_low_importance
from backend.agents.model_training import ModelTrainer, train_and_evaluate_models
from backend.agents.dashboard_schema_generator import generate_dashboard_schema
from backend.utils.llm_client import generate_dataset_summary, generate_dashboard_summary

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = BASE_DIR / "frontend"


def _to_json_serializable(obj):
    """Convert various types to JSON-serializable format, handling NaN and inf"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, float):
        # Handle NaN and inf
        if np.isnan(obj):
            return 0.0
        elif np.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, (pd.Int64Dtype().type,)):
        return int(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        f = float(obj)
        if np.isnan(f):
            return 0.0
        elif np.isinf(f):
            return 0.0
        return f
    return obj


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    df = load_csv(file)
    uid = str(uuid4())
    
    # Detect categorical classifications
    from backend.utils.categorical_encoder import classify_categorical_columns
    classifications = classify_categorical_columns(df)
    
    datasets[uid] = {
        "original": df, 
        "cleaned": None,
        "categorical_classifications": classifications,
        "categorical_mapping": classifications["binary_categorical"],  # For backward compatibility
    }

    analysis = analyze_dataset(df, target=None)

    # add some quick stats
    analysis["missing_counts"] = {
        k: int(v) for k, v in df.isnull().sum().to_dict().items()
    }
    analysis["duplicates"] = int(df.duplicated().sum())
    analysis["sample"] = df.head(10).to_dict(orient="records")
    
    # Add categorical information to analysis
    from backend.utils.categorical_encoder import get_categorical_info
    analysis["categorical_info"] = get_categorical_info(df)

    ai_summary = generate_dataset_summary(analysis, df.head(3).to_dict(orient="records"))
    if ai_summary:
        analysis["ai_summary"] = ai_summary
        analysis["ai_summary_source"] = "llm"
    
    # Convert analysis dict to JSON-serializable format (handle NaN/inf)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            elif obj == float('inf'):
                return float('inf')
            elif obj == float('-inf'):
                return float('-inf')
        return _to_json_serializable(obj)
    
    analysis = make_serializable(analysis)

    return {"id": uid, "analysis": analysis}


@app.post("/clean")
async def clean(id: str = Form(...), target: str = Form(...)):
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")

    df = datasets[id]["original"].copy()
    before_rows = df.shape[0]
    before_cols = df.shape[1]

    # Classify categorical columns
    from backend.utils.categorical_encoder import classify_categorical_columns
    classifications = classify_categorical_columns(df)
    
    # Remove fully categorical columns
    removed_columns = classifications["fully_categorical"]
    if removed_columns:
        df = df.drop(columns=removed_columns)
    
    # Store classifications and removed columns
    datasets[id]["categorical_classifications"] = classifications
    datasets[id]["removed_columns"] = removed_columns

    # Encode binary categorical columns to 0/1
    categorical_mapping = classifications["binary_categorical"]
    df_encoded, _ = encode_categorical_columns(df, categorical_mapping)

    # baseline cleaning (duplicates, missing, constant cols)
    df_after, report = clean_dataset(df_encoded)

    # feature selection relative to target (only if target is numeric after encoding)
    correlations = {}
    removed_by_corr = []
    if target in df_after.columns:
        # Check if target column is numeric (it should be after encoding if it's in categorical_mapping)
        if pd.api.types.is_numeric_dtype(df_after[target]):
            correlations = target_correlation(df_after, target)
            removed_by_corr = remove_low_importance(correlations)
            if removed_by_corr:
                df_after = df_after.drop(columns=[c for c in removed_by_corr if c in df_after.columns])

    after_rows = df_after.shape[0]
    after_cols = df_after.shape[1]

    # generate heatmap for numeric features (including target if numeric)
    numeric_features = df_after.select_dtypes(include=["number"]).columns.tolist()
    heatmap_b64 = None
    if len(numeric_features) >= 2:
        heatmap = run_eda(df_after, {"numeric_features": numeric_features})
        heatmap_b64 = heatmap.get("correlation_plot_base64")

    # Save cleaned
    datasets[id]["cleaned"] = df_after

    # Summarize removed columns for reporting (normalized to a flat list of column names)
    report_removed_col_names = []
    for item in report.get("columns_removed", []) or []:
        if isinstance(item, dict):
            report_removed_col_names.extend([str(col_name) for col_name in item.keys()])
        elif isinstance(item, str):
            report_removed_col_names.append(item)

    cols_removed_list = []
    cols_removed_list.extend([str(c) for c in removed_columns])
    cols_removed_list.extend(report_removed_col_names)
    cols_removed_list.extend([str(c) for c in removed_by_corr])

    # Keep order and remove duplicates
    cols_removed_list = list(dict.fromkeys(cols_removed_list))
    duplicates_removed = int(report.get("duplicates_removed", 0))
    missing_handled = report.get("missing_handled", {}) or {}

    # Build human-readable explanations for cleaning decisions
    unique_col_missing_details = report.get("unique_col_missing_details", []) or []
    unique_col_rows_removed = int(report.get("unique_col_rows_removed", 0))
    low_missing_rows_removed = int(report.get("low_missing_rows_removed", 0))
    low_missing_ratio = float(report.get("low_missing_ratio", 0.0))
    priority_rows_first_applied = bool(report.get("priority_rows_first_applied", False))

    columns_removed_explanations = []

    for col_name in removed_columns:
        columns_removed_explanations.append({
            "column": col_name,
            "reason": "fully_categorical",
            "details": (
                f"Column '{col_name}' was removed because it is a fully categorical "
                "feature and not suitable for numeric modeling in the current pipeline."
            )
        })

    for item in report.get("columns_removed", []) or []:
        if isinstance(item, dict):
            for col_name, details in item.items():
                reason = details.get("reason", "removed") if isinstance(details, dict) else "removed"
                detail_text = ""
                if isinstance(details, dict):
                    nested = details.get("details")
                    if isinstance(nested, dict):
                        miss = nested.get("missing")
                        why = nested.get("reason")
                        if miss is not None and why:
                            detail_text = f" (missing={miss}, rule={why})"
                columns_removed_explanations.append({
                    "column": col_name,
                    "reason": reason,
                    "details": f"Column '{col_name}' was removed because {reason}{detail_text}."
                })

    for col_name in removed_by_corr:
        columns_removed_explanations.append({
            "column": col_name,
            "reason": "low_target_correlation",
            "details": (
                f"Column '{col_name}' was removed during feature selection because "
                "its correlation with the target was below the threshold."
            )
        })

    # Keep order but avoid duplicate entries from overlapping cleaning stages.
    deduped_columns_removed = []
    seen_removed_pairs = set()
    for entry in columns_removed_explanations:
        key = (entry.get("column"), entry.get("reason"))
        if key in seen_removed_pairs:
            continue
        seen_removed_pairs.add(key)
        deduped_columns_removed.append(entry)
    columns_removed_explanations = deduped_columns_removed

    rows_removed_explanations = []
    if duplicates_removed > 0:
        rows_removed_explanations.append(
            f"Removed {duplicates_removed} duplicate row(s) to avoid repeated records."
        )
    if unique_col_rows_removed > 0:
        rows_removed_explanations.append(
            f"Removed {unique_col_rows_removed} row(s) where unique identifier columns (e.g., id/roll) had missing values."
        )
    if low_missing_rows_removed > 0:
        if priority_rows_first_applied:
            rows_removed_explanations.append(
                f"Removed {low_missing_rows_removed} row(s) with missing values first because: "
                f"(a) at least one column had >=40% missing values and "
                f"(b) only {low_missing_ratio:.2%} rows had missing data (<5%)."
            )
        else:
            rows_removed_explanations.append(
                f"Removed {low_missing_rows_removed} row(s) with missing values based on row-level missing-data rule."
            )
    final_rows_removed_missing = int(report.get("final_rows_removed_missing", 0))
    if final_rows_removed_missing > 0:
        rows_removed_explanations.append(
            f"Removed {final_rows_removed_missing} row(s) that still had missing values after cleaning."
        )

    missing_fill_explanations = []
    for col_name, info in missing_handled.items():
        if not isinstance(info, dict):
            continue
        method = info.get("method", "none")
        miss = int(info.get("missing", 0))
        if miss <= 0 or method in {"none", "dropped"}:
            continue
        if "mode" in str(method).lower():
            why = "categorical/boolean values are best filled using the most frequent value"
        elif "median" in str(method).lower():
            why = "the column showed high variation/outlier influence, so median is more robust"
        elif "mean" in str(method).lower():
            why = "the column had relatively stable numeric distribution, so mean is appropriate"
        elif "ffill" in str(method).lower() or "bfill" in str(method).lower():
            why = "datetime values were propagated from nearby rows"
        else:
            why = "rule-based imputation was applied"
        missing_fill_explanations.append({
            "column": col_name,
            "method": method,
            "missing_filled": miss,
            "details": f"Filled {miss} missing value(s) in '{col_name}' using {method} because {why}."
        })

    # Count imputed cells and breakdown by method
    imputed_cells = 0
    imputed_by_method = {}
    for col, info in missing_handled.items():
        method = info.get("method")
        missing_count = int(info.get("missing", 0))
        if method and method != "dropped":
            imputed_cells += missing_count
            imputed_by_method[method] = imputed_by_method.get(method, 0) + missing_count

    result = {
        "rows_before": int(before_rows),
        "rows_after": int(after_rows),
        "rows_removed": int(before_rows - after_rows),
        "cols_before": int(before_cols),
        "cols_after": int(after_cols),
        "cols_removed": cols_removed_list,
        "cols_removed_count": len(cols_removed_list),
        "duplicates_removed": duplicates_removed,
        "total_missing_cells": int(report.get("total_missing", 0)),
        "rows_with_missing": int(report.get("rows_with_missing", 0)),
        "row_missing_ratio": float(report.get("row_missing_ratio", 0.0)),
        "imputed_cells": int(imputed_cells),
        "imputed_by_method": imputed_by_method,
        "missing_handled": missing_handled,
        "cleaning_explanations": {
            "columns_removed": columns_removed_explanations,
            "rows_removed": rows_removed_explanations,
            "missing_filled": missing_fill_explanations,
            "unique_id_missing": unique_col_missing_details,
            "summary": [
                f"Rows: {before_rows} -> {after_rows} (removed: {before_rows - after_rows})",
                f"Columns: {before_cols} -> {after_cols} (removed: {len(cols_removed_list)})",
                f"Removed columns list: {', '.join(cols_removed_list) if cols_removed_list else 'None'}",
                *(report.get("summary", []) or []),
            ],
        },
        "sample": df_after.head(15).to_dict(orient="records"),
        "heatmap": heatmap_b64,
    }

    return result


@app.get("/data/{id}")
async def get_data(id: str, full: int = 0):
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")

    df = datasets[id].get("cleaned")
    if df is None:
        df = datasets[id].get("original")
    if df is None:
        raise HTTPException(status_code=404, detail="No data for this id")

    if full:
        csv = df.to_csv(index=False)
        filename = f"cleaned_{id}.csv"
        return Response(content=csv, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={filename}"})

    return {"sample": df.head(50).to_dict(orient="records")}


@app.post("/eda/{id}")
async def generate_eda(id: str, target: str = Form(default="")):
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")

    # Use cleaned data if available, otherwise original
    df = datasets[id].get("cleaned")
    if df is None:
        df = datasets[id].get("original")
    if df is None:
        raise HTTPException(status_code=404, detail="No data for this id")

    # Prepare analysis
    analysis = analyze_dataset(df, target=target if target else None)
    
    # Generate comprehensive EDA
    eda_results = run_eda(df, analysis)
    
    # Remove base64 image data before attempting JSON serialization to avoid oversizing
    eda_results_safe = {}
    for key, value in eda_results.items():
        if isinstance(value, dict):
            safe_dict = {}
            for k, v in value.items():
                # Store only plot names, not actual base64 (too large)
                if not isinstance(v, str) or not v.startswith('iVBOR'):
                    safe_dict[k] = v
            eda_results_safe[key] = safe_dict
        elif not isinstance(value, str) or not value.startswith('iVBOR'):
            eda_results_safe[key] = value
    
    # Store full results with images in memory for retrieval
    datasets[id]["eda_results"] = eda_results
    
    return {
        "dataset_info": {
            "rows": analysis["rows"],
            "columns": analysis["columns"],
            "numeric_features": analysis["numeric_features"],
            "categorical_features": analysis["categorical_features"],
            "target_column": analysis["target_column"],
            "problem_type": analysis["problem_type"]
        },
        "eda_summary": eda_results_safe
    }


@app.get("/eda-image/{id}/{category}/{sub_key}")
async def get_eda_image(id: str, category: str, sub_key: str):
    """Retrieve specific EDA visualization image"""
    if id not in datasets or "eda_results" not in datasets[id]:
        raise HTTPException(status_code=404, detail="EDA data not found")
    
    eda_results = datasets[id]["eda_results"]
    
    try:
        # Navigate through nested structure
        if category not in eda_results:
            raise HTTPException(status_code=404, detail="Category not found")
        
        category_data = eda_results[category]
        image_b64 = None
        
        # Direct access
        if sub_key in category_data:
            image_b64 = category_data[sub_key]
        else:
            # Handle nested dictionaries (e.g., histograms::feature_name)
            if '::' in sub_key:
                parts = sub_key.split('::', 1)
                if parts[0] in category_data and isinstance(category_data[parts[0]], dict):
                    image_b64 = category_data[parts[0]].get(parts[1])
            else:
                # Try searching in nested dicts
                for key, value in category_data.items():
                    if isinstance(value, dict) and sub_key in value:
                        image_b64 = value[sub_key]
                        break
        
        if image_b64 and isinstance(image_b64, str) and image_b64.startswith('iVBOR'):
            return Response(content=base64.b64decode(image_b64), media_type="image/png")
        
        raise HTTPException(status_code=404, detail="Image not found")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving image: {str(e)}")


@app.get("/categorical-info/{id}")
async def get_categorical_info_endpoint(id: str):
    """Get categorical column mappings for a dataset."""
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")
    
    classifications = datasets[id].get("categorical_classifications", {})
    removed_columns = datasets[id].get("removed_columns", [])
    
    # Return the stored classifications, with fully_categorical as removed
    return {
        "binary_categorical": classifications.get("binary_categorical", {}),
        "clustered_categorical": classifications.get("clustered_categorical", {}),
        "fully_categorical": removed_columns
    }


# ============= ML MODEL TRAINING ENDPOINTS =============

@app.post("/train-models")
async def train_models_endpoint(id: str = Form(...), target: str = Form(...)):
    """
    Train ML models on cleaned data
    Automatically detects classification vs regression and trains appropriate models
    
    Args:
        id: Dataset ID
        target: Target column name
        
    Returns:
        Training results with model performance metrics
    """
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")
    
    df_cleaned = datasets[id].get("cleaned")
    if df_cleaned is None:
        raise HTTPException(status_code=400, detail="Dataset not cleaned. Run cleaning first.")
    
    if target not in df_cleaned.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found")
    
    try:
        # Train models (auto-detects classification vs regression)
        trainer, results = train_and_evaluate_models(df_cleaned, target)
        
        task_type = results["preparation"].get("task_type", "unknown")
        task_label = "Classification" if task_type == "classification" else "Regression"
        
        # Check if any models trained successfully
        successful_models = [r for r in results["training"] if r.get("status") == "trained"]
        
        if not successful_models:
            failed_models = [r for r in results["training"] if r.get("status") == "failed"]
            failure_lines = [f"{m.get('model')}: {m.get('error', 'unknown error')}" for m in failed_models[:8]]
            failure_hint = "; ".join(failure_lines) if failure_lines else "No model-specific error details available"
            raise HTTPException(
                status_code=400,
                detail=f"All models failed during training. {failure_hint}"
            )
        
        # Store trainer and results
        trained_models[id] = {
            "target": target,
            "trainer": trainer,
            "results": results,
        }

        # Auto-generate interactive dashboard schema after training.
        dashboard_schema = None
        dashboard_error = None
        try:
            dashboard_schema = generate_dashboard_schema(
                df=df_cleaned,
                target=target,
                results=results,
                trainer=trainer,
            )
            trained_models[id]["dashboard_schema"] = dashboard_schema
        except Exception as e:
            dashboard_error = str(e)
            trained_models[id]["dashboard_schema_error"] = dashboard_error
        
        return {
            "status": "success",
            "task_type": task_type,
            "task_label": task_label,
            "message": f"✓ Trained {len(successful_models)}/{len(results['training'])} {task_label} models successfully",
            "preparation": results["preparation"],
            "training_results": results["training"],
            "best_model": results["best_model"],
            "best_model_metrics": results["metrics"],
            "comparison": results["comparison"],
            "dashboard_schema": dashboard_schema,
            "dashboard_schema_error": dashboard_error,
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        raise HTTPException(status_code=400, detail=f"Model training failed: {error_msg[:200]}")


@app.get("/model-results/{id}")
async def get_model_results(id: str):
    """Get trained model results and comparison"""
    if id not in trained_models:
        raise HTTPException(status_code=404, detail="No trained models for this dataset")
    
    model_data = trained_models[id]
    results = model_data["results"]
    task_type = results["preparation"].get("task_type", "classification")
    
    response = {
        "target": model_data["target"],
        "task_type": task_type,
        "best_model": results["best_model"],
        "best_model_metrics": _to_json_serializable(results["metrics"]),
        "all_metrics": _to_json_serializable(results["all_metrics"]),
        "comparison": _to_json_serializable(results["comparison"]),
        "feature_importance": model_data["trainer"].get_feature_importance(),
        "feature_names": model_data["trainer"].feature_names,
        "dashboard_schema": model_data.get("dashboard_schema"),
        "dashboard_schema_error": model_data.get("dashboard_schema_error"),
    }
    
    return response


@app.post("/generate-dashboard")
async def generate_dashboard(id: str = Form(...)):
    """Generate interactive dashboard schema from trained model outputs."""
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")
    if id not in trained_models:
        raise HTTPException(status_code=404, detail="No trained model found for this dataset")

    model_data = trained_models[id]
    target = model_data.get("target")
    trainer = model_data.get("trainer")
    results = model_data.get("results")
    df_cleaned = datasets[id].get("cleaned")
    if df_cleaned is None:
        df_cleaned = datasets[id].get("original")

    if df_cleaned is None:
        raise HTTPException(status_code=404, detail="Dataset data not available")
    if not target or trainer is None or not results:
        raise HTTPException(status_code=400, detail="Training artifacts are incomplete")

    try:
        generated = generate_dashboard_schema(
            df=df_cleaned,
            target=target,
            results=results,
            trainer=trainer,
        )
        ai_summary = generate_dashboard_summary(target, results, generated)
        if ai_summary:
            generated["ai_summary"] = ai_summary
            generated["ai_summary_source"] = "llm"

        model_data["dashboard_schema"] = generated
        model_data["dashboard_schema_error"] = None

        return {
            "status": "success",
            "message": "Interactive dashboard schema generated",
            "dashboard_schema": generated,
        }
    except Exception as e:
        model_data["dashboard_schema_error"] = str(e)
        raise HTTPException(status_code=400, detail=f"Dashboard generation failed: {str(e)[:300]}")


@app.get("/dashboard-schema/{id}")
async def get_dashboard_schema(id: str, refresh: int = 0):
    """Get cached or newly generated interactive dashboard schema."""
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset id not found")
    if id not in trained_models:
        raise HTTPException(status_code=404, detail="No trained model found for this dataset")

    model_data = trained_models[id]
    if not refresh and model_data.get("dashboard_schema"):
        return {
            "status": "success",
            "dashboard_schema": model_data.get("dashboard_schema"),
            "cached": True,
        }

    target = model_data.get("target")
    trainer = model_data.get("trainer")
    results = model_data.get("results")
    df_cleaned = datasets[id].get("cleaned")
    if df_cleaned is None:
        df_cleaned = datasets[id].get("original")

    if df_cleaned is None:
        raise HTTPException(status_code=404, detail="Dataset data not available")
    if not target or trainer is None or not results:
        raise HTTPException(status_code=400, detail="Training artifacts are incomplete")

    try:
        schema = generate_dashboard_schema(
            df=df_cleaned,
            target=target,
            results=results,
            trainer=trainer,
        )
        model_data["dashboard_schema"] = schema
        model_data["dashboard_schema_error"] = None
        return {
            "status": "success",
            "dashboard_schema": schema,
            "cached": False,
        }
    except Exception as e:
        model_data["dashboard_schema_error"] = str(e)
        raise HTTPException(status_code=400, detail=f"Dashboard schema generation failed: {str(e)[:300]}")


@app.post("/predict")
async def predict(id: str = Form(...), input_data: str = Form(...)):
    """
    Predict using the best trained model
    
    Args:
        id: Dataset ID
        input_data: JSON string with feature values
        
    Returns:
        Prediction with probabilities
    """
    if id not in trained_models:
        raise HTTPException(status_code=404, detail="No trained models for this dataset. Train models first.")
    
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        row_data = json.loads(input_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {str(e)}")
    
    try:
        trainer = trained_models[id]["trainer"]
        classifications = datasets[id].get("categorical_classifications", {})
        binary_mapping = classifications.get("binary_categorical", {})
        
        # Encode binary categorical values in input
        encoded_data = row_data.copy()
        for col, (positive_cat, negative_cat) in binary_mapping.items():
            if col in encoded_data:
                val_lower = str(encoded_data[col]).lower().strip()
                if val_lower == positive_cat.lower():
                    encoded_data[col] = 1
                elif val_lower == negative_cat.lower():
                    encoded_data[col] = 0
        
        # Make prediction
        prediction_result = trainer.predict_single(encoded_data)
        
        # Decode prediction back to categorical format if target is binary
        predictions = prediction_result["predictions"]
        target = trained_models[id]["target"]
        
        decoded_predictions = predictions.copy()
        if target in binary_mapping:
            positive_cat, negative_cat = binary_mapping[target]
            pred_val = predictions[target]
            if isinstance(pred_val, (int, float)):
                if pred_val >= 0.5:
                    decoded_predictions[target] = positive_cat
                else:
                    decoded_predictions[target] = negative_cat
        
        return {
            "prediction": decoded_predictions,
            "model_used": prediction_result["model_used"],
            "probabilities": prediction_result["probabilities"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(id: str = Form(...), file: UploadFile = File(...)):
    """
    Predict on a batch of rows from an uploaded CSV file
    
    Args:
        id: Dataset ID (for accessing model and categorical mappings)
        file: CSV file with feature values (without target column)
        
    Returns:
        List of predictions for each row with original features
    """
    if id not in trained_models:
        raise HTTPException(status_code=404, detail="No trained models for this dataset. Train models first.")
    
    if id not in datasets:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load the batch data
        batch_df = load_csv(file)
        
        if batch_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        trainer = trained_models[id]["trainer"]
        classifications = datasets[id].get("categorical_classifications", {})
        binary_mapping = classifications.get("binary_categorical", {})
        removed_columns = datasets[id].get("removed_columns", [])
        target = trained_models[id]["target"]
        
        # Remove fully categorical columns and target column if present
        cols_to_drop = [col for col in removed_columns if col in batch_df.columns]
        if target in batch_df.columns:
            cols_to_drop.append(target)
        
        if cols_to_drop:
            batch_df = batch_df.drop(columns=cols_to_drop)
        
        # Prepare predictions list
        predictions_list = []
        
        # Process each row
        for idx, row in batch_df.iterrows():
            try:
                row_dict = row.to_dict()
                
                # Convert NaN to None for JSON serialization
                row_dict = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
                
                # Encode binary categorical values
                encoded_data = row_dict.copy()
                for col, (positive_cat, negative_cat) in binary_mapping.items():
                    if col in encoded_data and encoded_data[col] is not None:
                        val_lower = str(encoded_data[col]).lower().strip()
                        if val_lower == positive_cat.lower():
                            encoded_data[col] = 1
                        elif val_lower == negative_cat.lower():
                            encoded_data[col] = 0
                
                # Make prediction
                prediction_result = trainer.predict_single(encoded_data)
                
                # Extract prediction value (it's a list with one element for single row)
                predictions_list_val = prediction_result["predictions"]
                pred_value = predictions_list_val[0] if isinstance(predictions_list_val, list) else predictions_list_val
                
                # Decode prediction if target is binary
                decoded_prediction = pred_value
                if target in binary_mapping:
                    positive_cat, negative_cat = binary_mapping[target]
                    if isinstance(pred_value, (int, float)):
                        if pred_value >= 0.5:
                            decoded_prediction = positive_cat
                        else:
                            decoded_prediction = negative_cat
                
                # Add original features + prediction
                result_row = {**row_dict, "prediction": decoded_prediction}
                predictions_list.append(result_row)
                
            except Exception as row_error:
                # Log error but continue
                result_row = {**row_dict, "prediction": f"Error: {str(row_error)[:50]}"}
                predictions_list.append(result_row)
        
        return {
            "status": "success",
            "total_rows": len(batch_df),
            "predicted_rows": len([p for p in predictions_list if "Error" not in str(p.get("prediction", ""))]),
            "predictions": predictions_list,
            "columns": list(batch_df.columns) + ["prediction"],
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)[:200]}")


if FRONTEND_DIR.exists():
    # Serve static frontend pages and assets from the same origin as the API.
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
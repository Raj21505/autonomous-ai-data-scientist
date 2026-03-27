"""
Machine Learning Model Training Agent
Trains multiple models (classification or regression) and selects the best one.
Automatically detects whether the task is classification or regression.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
)
import warnings

warnings.filterwarnings("ignore")


class ModelTrainer:
    """Trains and evaluates multiple ML models for classification or regression"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        self.feature_names = []
        self.task_type = None  # 'classification' or 'regression'
        self.categorical_cols = []  # Original categorical columns
        self.numeric_cols = []  # Original numeric columns
        self.encoded_feature_mapping = {}  # Maps encoded features to original

    def detect_task_type(self, y):
        """
        Automatically detect if task is classification or regression
        
        Args:
            y: Target variable
            
        Returns:
            'classification' or 'regression'
        """
        y_non_null = y.dropna()
        unique_values = len(y_non_null.unique())

        # Non-numeric targets are always treated as classification labels.
        if not pd.api.types.is_numeric_dtype(y_non_null):
            return 'classification'

        y_numeric = pd.to_numeric(y_non_null, errors='coerce')
        y_numeric = y_numeric.dropna()
        if y_numeric.empty:
            return 'classification'

        integer_like_ratio = float(np.isclose(y_numeric, np.round(y_numeric), atol=1e-8).mean())
        unique_ratio = float(unique_values / max(1, len(y_numeric)))

        # Few unique values typically imply label classes.
        if unique_values <= 20 and (integer_like_ratio >= 0.98 or unique_ratio <= 0.05):
            return 'classification'

        # Very low cardinality even in larger ranges is often classification.
        if unique_values <= 50 and unique_ratio <= 0.02:
            return 'classification'

        # Otherwise treat numeric target as regression.
        return 'regression'

    def prepare_data(self, df, target_column, test_size=0.2):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with data preparation info
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify and encode categorical columns
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # One-hot encode categorical columns
        if self.categorical_cols:
            X_encoded = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)
            # Create mapping from original categorical columns to encoded features
            self.encoded_feature_mapping = {}
            for orig_col in self.categorical_cols:
                encoded_cols = [col for col in X_encoded.columns if col.startswith(f"{orig_col}_")]
                if encoded_cols:
                    self.encoded_feature_mapping[orig_col] = encoded_cols
                else:
                    # If drop_first=True removed all, it means binary with one value
                    # But since we have clustered, it should have multiple
                    pass
            X = X_encoded
        
        # Convert all columns to numeric (in case of any remaining issues)
        X = X.astype(float)
        
        self.feature_names = X.columns.tolist()

        # Remove any rows with missing values
        valid_idx = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        if len(y) < 2:
            raise ValueError("Not enough valid rows after cleaning to train models.")

        # Detect task type
        self.task_type = self.detect_task_type(y)

        # Normalize classification labels so sklearn classifiers don't reject
        # float labels that are effectively classes (e.g., 6.0/7.0 with one imputed 6.3).
        if self.task_type == 'classification' and pd.api.types.is_numeric_dtype(y):
            y_numeric = pd.to_numeric(y, errors='coerce')
            integer_like_ratio = float(np.isclose(y_numeric, np.round(y_numeric), atol=1e-8).mean())
            if integer_like_ratio >= 0.95:
                y = pd.Series(np.round(y_numeric).astype(int), index=y.index)
            else:
                # Fall back to string labels for non-integer but discrete categories.
                y = y.astype(str)

        if self.task_type == 'classification' and len(pd.Series(y).unique()) < 2:
            raise ValueError("Classification target must contain at least 2 classes.")
        
        # Split data based on task type
        if self.task_type == 'classification':
            unique_values = len(y.unique())
            try:
                # Try stratified split for classification
                if unique_values >= 2:
                    value_counts = y.value_counts()
                    if (value_counts >= 2).all():
                        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            X, y, test_size=test_size, random_state=self.random_state, stratify=y
                        )
                    else:
                        # Some classes have < 2 samples
                        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                            X, y, test_size=test_size, random_state=self.random_state
                        )
                else:
                    raise ValueError("Need at least 2 classes for classification")
            except:
                # Fallback to random split
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state
                )

            # Safety fallback: if random split produced single-class train set,
            # use full data for train/test so models can still train.
            if len(pd.Series(self.y_train).unique()) < 2:
                self.X_train, self.X_test, self.y_train, self.y_test = X.copy(), X.copy(), y.copy(), y.copy()
        else:
            # Regression: simple random split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            if len(self.y_train) < 2:
                self.X_train, self.X_test, self.y_train, self.y_test = X.copy(), X.copy(), y.copy(), y.copy()

        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train), columns=X.columns
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test), columns=X.columns
        )

        y_numeric = pd.to_numeric(y, errors='coerce')
        has_numeric_target = y_numeric.notna().any()

        # Prepare return dict
        prep_info = {
            "total_samples": len(y),
            "train_samples": len(self.y_train),
            "test_samples": len(self.y_test),
            "n_features": X.shape[1],
            "feature_names": self.feature_names,
            "task_type": self.task_type,
            "target_stats": {
                "min": float(y_numeric.min()) if has_numeric_target else None,
                "max": float(y_numeric.max()) if has_numeric_target else None,
                "mean": float(y_numeric.mean()) if has_numeric_target else None,
                "std": float(y_numeric.std()) if has_numeric_target else None,
                "unique_values": len(y.unique()),
                "dtype": str(y.dtype),
            }
        }
        
        # Add task-specific info
        if self.task_type == 'classification':
            prep_info["target_classes"] = sorted([str(v) for v in y.unique().tolist()])
        else:
            prep_info["target_min"] = float(y_numeric.min()) if has_numeric_target else None
            prep_info["target_max"] = float(y_numeric.max()) if has_numeric_target else None
        
        return prep_info

    def train_models(self):
        """
        Train appropriate ML models based on task type
        
        Returns:
            List of training results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        if self.task_type == 'classification':
            return self._train_classification_models()
        else:
            return self._train_regression_models()

    def _train_classification_models(self):
        """Train 8 classification models"""
        knn_neighbors = max(1, min(5, len(self.X_train)))
        models_config = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting Classifier": GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            "Support Vector Machine": SVC(
                kernel="rbf", probability=True, random_state=self.random_state
            ),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=self.random_state),
            "K-Nearest Neighbors Classifier": KNeighborsClassifier(n_neighbors=knn_neighbors, n_jobs=-1),
            "Neural Network Classifier": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
            ),
            "AdaBoost Classifier": AdaBoostClassifier(
                n_estimators=100, random_state=self.random_state
            ),
        }

        training_results = []

        for model_name, model in models_config.items():
            try:
                model.fit(self.X_train, self.y_train)
                self.models[model_name] = model
                y_pred = model.predict(self.X_test)
                
                try:
                    accuracy = accuracy_score(self.y_test, y_pred)
                    precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
                except:
                    accuracy = precision = recall = f1 = 0.0
                
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(self.X_test)
                        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                            roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
                        else:
                            roc_auc = 0.0
                    else:
                        roc_auc = 0.0
                except:
                    roc_auc = 0.0

                self.results[model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "roc_auc": float(roc_auc),
                }

                training_results.append({
                    "model": model_name,
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "roc_auc": float(roc_auc),
                    "status": "trained",
                })

            except Exception as e:
                error_msg = str(e)[:100]
                training_results.append({
                    "model": model_name,
                    "status": "failed",
                    "error": error_msg
                })
                self.results[model_name] = {
                    "accuracy": 0.0, "precision": 0.0, "recall": 0.0,
                    "f1_score": 0.0, "roc_auc": 0.0,
                }

        return training_results

    def _train_regression_models(self):
        """Train 8 regression models"""
        knn_neighbors = max(1, min(5, len(self.X_train)))
        models_config = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting Regressor": GradientBoostingRegressor(
                n_estimators=100, random_state=self.random_state
            ),
            "Support Vector Regressor": SVR(kernel="rbf"),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=self.random_state),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=knn_neighbors, n_jobs=-1),
            "Neural Network Regressor": MLPRegressor(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
            ),
            "AdaBoost Regressor": AdaBoostRegressor(
                n_estimators=100, random_state=self.random_state
            ),
        }

        training_results = []

        for model_name, model in models_config.items():
            try:
                model.fit(self.X_train, self.y_train)
                self.models[model_name] = model
                y_pred = model.predict(self.X_test)
                
                try:
                    r2 = r2_score(self.y_test, y_pred)
                    mae = mean_absolute_error(self.y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                    # MAPE can fail if y_test has zeros
                    try:
                        mape = mean_absolute_percentage_error(self.y_test, y_pred)
                    except:
                        mape = 0.0
                except:
                    r2 = mae = rmse = mape = 0.0

                self.results[model_name] = {
                    "r2_score": float(r2),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                }

                training_results.append({
                    "model": model_name,
                    "r2_score": float(r2),
                    "mae": float(mae),
                    "rmse": float(rmse),
                    "mape": float(mape),
                    "status": "trained",
                })

            except Exception as e:
                error_msg = str(e)[:100]
                training_results.append({
                    "model": model_name,
                    "status": "failed",
                    "error": error_msg
                })
                self.results[model_name] = {
                    "r2_score": 0.0, "mae": 0.0, "rmse": 0.0, "mape": 0.0,
                }

        return training_results

    def select_best_model(self, metric=None):
        """
        Select best model based on task type
        
        Args:
            metric: Specific metric to use (auto-selected if None)
            
        Returns:
            Dictionary with best model info
        """
        if not self.results:
            raise ValueError("No trained models. Call train_models first.")

        # Select metric based on task type
        if metric is None:
            metric = "f1_score" if self.task_type == 'classification' else "r2_score"

        # Only consider models that actually trained successfully.
        trained_models = {
            name: self.results.get(name, {}) for name in self.models.keys()
        }

        if not trained_models:
            raise ValueError("No models trained successfully.")

        # For regression, higher R² is better; for classification, higher metric is better
        best_model_name = max(
            trained_models.keys(), key=lambda x: trained_models[x].get(metric, 0)
        )

        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        return {
            "best_model": best_model_name,
            "metrics": self.results[best_model_name],
            "all_results": self.results,
        }

    def get_model_comparison(self):
        """Get comparison of all trained models"""
        comparison = []
        
        if self.task_type == 'classification':
            metric_key = "f1_score"
        else:
            metric_key = "r2_score"

        for model_name, metrics in self.results.items():
            comparison.append({"model": model_name, **metrics})

        # Sort by primary metric
        comparison.sort(key=lambda x: x.get(metric_key, 0), reverse=True)
        return comparison

    def predict(self, X):
        """Make predictions using best model"""
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model first.")

        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)
        predictions = self.best_model.predict(X_scaled)
        
        probabilities = None
        if self.task_type == 'classification':
            if hasattr(self.best_model, "predict_proba"):
                probabilities = self.best_model.predict_proba(X_scaled)
            elif hasattr(self.best_model, "decision_function"):
                probabilities = self.best_model.decision_function(X_scaled)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "model_used": self.best_model_name,
            "task_type": self.task_type,
        }

    def predict_single(self, row_dict):
        """
        Predict for a single row
        Handles categorical encoding if needed
        """
        X = pd.DataFrame([row_dict])
        
        # Apply same categorical encoding if applicable
        if self.categorical_cols:
            # One-hot encode categorical columns
            X = pd.get_dummies(X, columns=self.categorical_cols, drop_first=True, dtype=float)
        
        # Ensure all feature columns exist (fill missing with 0)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        
        # Select only the features used during training (in correct order)
        X = X[self.feature_names]
        
        return self.predict(X)

    def get_feature_importance(self):
        """Get feature importance from best model if available"""
        if self.best_model is None:
            return {}

        feature_importance = {}

        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            for feat, imp in zip(self.feature_names, importances):
                feature_importance[feat] = float(imp)
        elif hasattr(self.best_model, "coef_"):
            coefs = np.abs(self.best_model.coef_)
            if len(coefs.shape) > 1:
                coefs = coefs[0]
            for feat, coef in zip(self.feature_names, coefs):
                feature_importance[feat] = float(coef)

        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def get_model_report(self):
        """Get detailed report of all models"""
        report = {
            "total_models": len(self.results),
            "trained_models": len([m for m in self.results if m in self.models]),
            "best_model": self.best_model_name,
            "best_model_metrics": self.results.get(self.best_model_name, {}),
            "all_models_metrics": self.results,
            "feature_importance": self.get_feature_importance(),
            "feature_names": self.feature_names,
            "task_type": self.task_type,
        }
        return report


def train_and_evaluate_models(df, target_column):
    """
    Convenience function to train models and return results
    
    Args:
        df: DataFrame with cleaned data
        target_column: Name of target column
        
    Returns:
        Tuple of (trainer object, results summary)
    """
    trainer = ModelTrainer()

    # Prepare data
    prep_info = trainer.prepare_data(df, target_column)

    # Train models
    training_results = trainer.train_models()

    # If all models failed, return partial results so caller can handle cleanly.
    successful_models = [r for r in training_results if r.get("status") == "trained"]
    if not successful_models:
        results = {
            "preparation": prep_info,
            "training": training_results,
            "best_model": None,
            "metrics": {},
            "all_metrics": trainer.results,
            "comparison": trainer.get_model_comparison(),
        }
        return trainer, results

    # Select best model
    best_info = trainer.select_best_model()

    results = {
        "preparation": prep_info,
        "training": training_results,
        "best_model": best_info["best_model"],
        "metrics": best_info["metrics"],
        "all_metrics": best_info["all_results"],
        "comparison": trainer.get_model_comparison(),
    }

    return trainer, results

    def train_models(self):
        """
        Train 7-8 different ML models
        
        Returns:
            Dictionary with training results
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data first.")

        # Define models
        models_config = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=self.random_state
            ),
            "Support Vector Machine": SVC(
                kernel="rbf", probability=True, random_state=self.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "Neural Network": MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=1000,
                random_state=self.random_state,
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=100, random_state=self.random_state
            ),
        }

        training_results = []

        for model_name, model in models_config.items():
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                self.models[model_name] = model

                # Make predictions
                y_pred = model.predict(self.X_test)
                
                # Calculate metrics
                try:
                    accuracy = accuracy_score(self.y_test, y_pred)
                    precision = precision_score(self.y_test, y_pred, average="weighted", zero_division=0)
                    recall = recall_score(self.y_test, y_pred, average="weighted", zero_division=0)
                    f1 = f1_score(self.y_test, y_pred, average="weighted", zero_division=0)
                except Exception as metric_error:
                    # If metric calculation fails, use defaults
                    accuracy = 0.0
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0
                
                # Try ROC-AUC (may fail for some models)
                try:
                    if hasattr(model, "predict_proba"):
                        y_proba = model.predict_proba(self.X_test)
                        # Handle both binary and multiclass
                        if len(y_proba.shape) > 1 and y_proba.shape[1] == 2:
                            roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
                        else:
                            roc_auc = 0.0
                    else:
                        try:
                            y_score = model.decision_function(self.X_test)
                            roc_auc = roc_auc_score(self.y_test, y_score)
                        except:
                            roc_auc = 0.0
                except:
                    roc_auc = 0.0

                self.results[model_name] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "roc_auc": float(roc_auc),
                    "training_time": "N/A",
                }

                training_results.append(
                    {
                        "model": model_name,
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1_score": float(f1),
                        "roc_auc": float(roc_auc),
                        "status": "trained",
                    }
                )

            except Exception as e:
                # Log error but continue with other models
                error_msg = str(e)[:100]  # Truncate error message
                training_results.append(
                    {"model": model_name, "status": "failed", "error": error_msg}
                )
                # Assign default poor metrics so model isn't selected
                self.results[model_name] = {
                    "accuracy": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "roc_auc": 0.0,
                    "training_time": "N/A",
                }

        return training_results

    def select_best_model(self, metric="f1_score"):
        """
        Select best model based on specified metric
        
        Args:
            metric: Metric to use for selection (accuracy, f1_score, roc_auc)
            
        Returns:
            Dictionary with best model info
        """
        if not self.results:
            raise ValueError("No trained models. Call train_models first.")

        # Filter models that actually trained (have positive metrics)
        trained_models = {
            name: metrics for name, metrics in self.results.items()
            if metrics.get(metric, 0) > 0
        }

        if not trained_models:
            # If no models succeeded, pick the first one with highest accuracy
            trained_models = self.results

        # Sort models by metric
        best_model_name = max(
            trained_models.keys(), key=lambda x: trained_models[x].get(metric, 0)
        )

        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        return {
            "best_model": best_model_name,
            "metrics": self.results[best_model_name],
            "all_results": self.results,
        }

    def get_model_comparison(self):
        """Get comparison of all trained models"""
        comparison = []
        for model_name, metrics in self.results.items():
            comparison.append({"model": model_name, **metrics})

        # Sort by f1_score
        comparison.sort(key=lambda x: x.get("f1_score", 0), reverse=True)

        return comparison

    def predict(self, X):
        """
        Make predictions using best model
        
        Args:
            X: Features DataFrame
            
        Returns:
            Predictions and probabilities
        """
        if self.best_model is None:
            raise ValueError("No best model selected. Call select_best_model first.")

        # Scale features
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        predictions = self.best_model.predict(X_scaled)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(self.best_model, "predict_proba"):
            probabilities = self.best_model.predict_proba(X_scaled)
        elif hasattr(self.best_model, "decision_function"):
            probabilities = self.best_model.decision_function(X_scaled)

        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "model_used": self.best_model_name,
        }

    def get_feature_importance(self):
        """Get feature importance from best model if available"""
        if self.best_model is None:
            return {}

        feature_importance = {}

        # Random Forest
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
            for feat, imp in zip(self.feature_names, importances):
                feature_importance[feat] = float(imp)

        # Logistic Regression coefficients
        elif hasattr(self.best_model, "coef_"):
            coefs = np.abs(self.best_model.coef_[0])
            for feat, coef in zip(self.feature_names, coefs):
                feature_importance[feat] = float(coef)

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return feature_importance

    def get_model_report(self):
        """Get detailed report of all models"""
        report = {
            "total_models": len(self.results),
            "trained_models": len([m for m in self.results if m in self.models]),
            "best_model": self.best_model_name,
            "best_model_metrics": self.results.get(self.best_model_name, {}),
            "all_models_metrics": self.results,
            "feature_importance": self.get_feature_importance(),
            "feature_names": self.feature_names,
        }
        return report

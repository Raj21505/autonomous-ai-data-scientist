# backend/agents/eda_agent.py

import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats


def plot_to_base64():
    """Convert matplotlib figure to base64 image"""
    # Ensure all EDA outputs are legend-free, regardless of plot type.
    fig = plt.gcf()
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_eda(df, analysis):
    """Generate comprehensive EDA visualizations"""
    results = {}
    
    numeric_features = analysis.get("numeric_features", [])
    categorical_features = analysis.get("categorical_features", [])
    target_column = analysis.get("target_column")
    
    # 1. Distribution Analysis
    results["distributions"] = generate_distribution_analysis(df, numeric_features, categorical_features)
    
    # 2. Missing Data Visualization
    results["missing_data"] = generate_missing_data_report(df)
    
    # 3. Categorical Feature Analysis
    results["categorical_analysis"] = generate_categorical_analysis(df, categorical_features, target_column)
    
    # 4. Statistical Summaries
    results["statistical_summary"] = generate_statistical_summary(df, numeric_features)
    
    # 5. Feature Relationships
    results["feature_relationships"] = generate_feature_relationships(df, numeric_features, categorical_features, target_column)
    
    # 6. Correlation Heatmap (existing)
    if len(numeric_features) >= 2:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            df[numeric_features].corr(), annot=True, cmap="coolwarm", fmt=".2f"
        )
        plt.title("Correlation Heatmap")
        results["correlation_plot_base64"] = plot_to_base64()
    
    return results


def generate_distribution_analysis(df, numeric_features, categorical_features):
    """Generate histograms, KDE plots, box plots, and value counts"""
    histograms = {}
    box_plots = {}
    value_counts = {}
    
    # Histograms and KDE plots for numeric features
    for col in numeric_features:
        try:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            df[col].hist(bins=30, edgecolor='black', alpha=0.7)
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            
            plt.subplot(1, 2, 2)
            df[col].plot(kind='kde')
            plt.title(f"KDE Plot of {col}")
            plt.xlabel(col)
            
            histograms[col] = plot_to_base64()
        except:
            pass
    
    # Box plots for numeric features
    if numeric_features:
        try:
            plt.figure(figsize=(10, 4))
            df[numeric_features].boxplot()
            plt.title("Box Plots - Numeric Features")
            plt.xticks(rotation=45)
            box_plots["all"] = plot_to_base64()
        except:
            pass
    
    # Value counts for categorical features
    for col in categorical_features:
        try:
            plt.figure(figsize=(8, 4))
            df[col].value_counts().head(10).plot(kind='barh')
            plt.title(f"Value Counts - {col}")
            plt.xlabel("Count")
            value_counts[col] = plot_to_base64()
        except:
            pass
    
    return {
        "histograms": histograms,
        "box_plots": box_plots,
        "value_counts": value_counts
    }


def generate_missing_data_report(df):
    """Generate missing data analysis"""
    missing_stats = {}
    
    # Calculate missing percentages
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
    }).sort_values('Missing_Percentage', ascending=False)
    
    missing_stats["summary"] = missing_data.to_dict(orient='records')
    missing_stats["total_missing_cells"] = int(df.isnull().sum().sum())
    missing_stats["total_cells"] = int(df.shape[0] * df.shape[1])
    
    # Missing data heatmap
    if df.isnull().sum().sum() > 0:
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
            plt.title("Missing Data Heatmap")
            missing_stats["heatmap"] = plot_to_base64()
        except:
            pass
    
    # Missing pattern - correlation between missing values
    missing_corr = df.isnull().corr()
    if not missing_corr.empty:
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(missing_corr, annot=True, cmap='coolwarm', center=0)
            plt.title("Missing Value Correlations")
            missing_stats["missing_pattern"] = plot_to_base64()
        except:
            pass
    
    return missing_stats


def generate_categorical_analysis(df, categorical_features, target_column=None):
    """Categorical feature 
      with cardinality and class imbalance"""
    categorical_stats = {}
    
    # Feature cardinality report
    cardinality = {}
    for col in categorical_features:
        unique_count = df[col].nunique()
        cardinality[col] = {
            "unique_values": unique_count,
            "value_list": df[col].value_counts().to_dict()
        }
    categorical_stats["cardinality"] = cardinality
    
    # Class imbalance detection for target
    if target_column and target_column in df.columns:
        try:
            target_dist = df[target_column].value_counts()
            categorical_stats["target_distribution"] = target_dist.to_dict()
            
            plt.figure(figsize=(8, 4))
            target_dist.plot(kind='bar')
            plt.title(f"Target Variable Distribution: {target_column}")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            categorical_stats["target_plot"] = plot_to_base64()
            
            # Calculate imbalance ratio
            imbalance_ratio = target_dist.max() / target_dist.min()
            categorical_stats["imbalance_ratio"] = float(imbalance_ratio)
        except:
            pass
        
        # Contingency tables between categorical features and target
        contingency_tables = {}
        for cat_col in categorical_features:
            if cat_col != target_column:
                try:
                    contingency = pd.crosstab(df[cat_col], df[target_column])
                    contingency_tables[cat_col] = contingency.to_dict()
                except:
                    pass
        categorical_stats["contingency_tables"] = contingency_tables
    
    return categorical_stats


def generate_statistical_summary(df, numeric_features):
    """Generate statistical summaries including skewness, kurtosis, variance"""
    stats_summary = {}
    
    for col in numeric_features:
        try:
            skewness = float(df[col].skew())
            kurtosis = float(df[col].kurtosis())
            variance = float(df[col].var())
            std_dev = float(df[col].std())
            
            # Identify outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(df)) * 100
            
            # Check for constant/near-constant features
            unique_ratio = df[col].nunique() / len(df)
            
            stats_summary[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std_dev": std_dev,
                "variance": variance,
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "skewness": skewness,
                "kurtosis": kurtosis,
                "outlier_count": outlier_count,
                "outlier_percentage": float(outlier_percentage),
                "unique_ratio": float(unique_ratio),
                "is_constant": unique_ratio < 0.01
            }
        except:
            pass
    
    return stats_summary


def generate_feature_relationships(df, numeric_features, categorical_features, target_column=None):
    """Generate visualizations for feature relationships"""
    relationships = {}
    
    # Pairplot for top numeric features (max 5 to avoid overcrowding)
    top_numeric = numeric_features[:5] if len(numeric_features) > 5 else numeric_features
    if len(top_numeric) >= 2:
        try:
            plt.figure(figsize=(10, 8))
            pd.plotting.scatter_matrix(df[top_numeric], alpha=0.3)
            plt.tight_layout()
            relationships["pairplot"] = plot_to_base64()
        except:
            pass
    
    # Target vs numeric features
    if target_column and target_column in df.columns and target_column in numeric_features:
        scatter_plots = {}
        for feat in numeric_features:
            if feat != target_column:
                try:
                    plt.figure(figsize=(7, 4))
                    plt.scatter(df[feat], df[target_column], alpha=0.6)
                    plt.xlabel(feat)
                    plt.ylabel(target_column)
                    plt.title(f"{feat} vs {target_column}")
                    scatter_plots[feat] = plot_to_base64()
                except:
                    pass
        relationships["target_scatter"] = scatter_plots
    
    # Target vs categorical features (count plots)
    if target_column and target_column in df.columns:
        count_plots = {}
        for cat_col in categorical_features:
            if cat_col != target_column:
                try:
                    plt.figure(figsize=(8, 4))
                    cross_tab = pd.crosstab(df[cat_col], df[target_column])
                    cross_tab.plot(kind='bar')
                    plt.title(f"{cat_col} vs {target_column}")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)
                    count_plots[cat_col] = plot_to_base64()
                except:
                    pass
        relationships["target_categorical"] = count_plots
    
    return relationships

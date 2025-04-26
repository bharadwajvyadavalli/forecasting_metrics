"""
Lag-Based Predictions Module

This module provides functionality for analyzing forecast accuracy by forecast horizon (lag).
It helps understand how accuracy degrades as the forecast horizon increases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional, Callable

import forecast_metrics as fm


def compute_metrics_by_lag(
        df: pd.DataFrame,
        metric_fns: Dict[str, Callable] = None
) -> pd.DataFrame:
    """
    Compute metrics grouped by forecast lag (horizon).

    Parameters:
        df (pd.DataFrame): Forecast data with Actual_Month and Prediction_Month
        metric_fns (Dict[str, Callable], optional): Dictionary of metric functions.
            If None, uses default metrics.

    Returns:
        pd.DataFrame: DataFrame with metrics by lag
    """
    # Make a copy of the dataframe
    df = df.copy()

    # Ensure date columns are datetime
    df['Actual_Month'] = pd.to_datetime(df['Actual_Month'])
    df['Prediction_Month'] = pd.to_datetime(df['Prediction_Month'])

    # Calculate lag (in months)
    df['lag'] = ((df.Prediction_Month.dt.year - df.Actual_Month.dt.year) * 12 +
                 (df.Prediction_Month.dt.month - df.Actual_Month.dt.month))

    # Default metric functions if not provided
    if metric_fns is None:
        metric_fns = {
            'Mean_Bias': fm.mean_bias,
            'Tracking_Signal': fm.tracking_signal,
            'Direction_Accuracy': fm.direction_accuracy,
            'Data_Anomaly_Rate': fm.data_anomaly_rate,
            'Residual_Anomaly_Rate': fm.residual_anomaly_rate
        }

    # Compute metrics by lag
    results = []
    for lag, lag_group in df.groupby('lag'):
        row = {'lag': lag, 'count': len(lag_group)}

        # Skip lags with insufficient data
        if len(lag_group) < 5:
            continue

        # Compute metrics
        for name, fn in metric_fns.items():
            try:
                a = lag_group['Prediction_Actual'].to_numpy()
                p = lag_group['Prediction_Value'].to_numpy()
                row[name] = fn(a, p)
            except Exception as e:
                print(f"Error computing {name} for lag {lag}: {e}")
                row[name] = np.nan

        results.append(row)

    # Create DataFrame and round numeric columns
    result_df = pd.DataFrame(results).sort_values('lag')
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)

    return result_df


def compute_metrics_by_lag_and_sku(
        df: pd.DataFrame,
        metric_fns: Dict[str, Callable] = None
) -> pd.DataFrame:
    """
    Compute metrics grouped by SKU and forecast lag (horizon).

    Parameters:
        df (pd.DataFrame): Forecast data with Actual_Month and Prediction_Month
        metric_fns (Dict[str, Callable], optional): Dictionary of metric functions.
            If None, uses default metrics.

    Returns:
        pd.DataFrame: DataFrame with metrics by SKU and lag
    """
    # Make a copy of the dataframe
    df = df.copy()

    # Ensure date columns are datetime
    df['Actual_Month'] = pd.to_datetime(df['Actual_Month'])
    df['Prediction_Month'] = pd.to_datetime(df['Prediction_Month'])

    # Calculate lag (in months)
    df['lag'] = ((df.Prediction_Month.dt.year - df.Actual_Month.dt.year) * 12 +
                 (df.Prediction_Month.dt.month - df.Actual_Month.dt.month))

    # Default metric functions if not provided
    if metric_fns is None:
        metric_fns = {
            'Mean_Bias': fm.mean_bias,
            'Tracking_Signal': fm.tracking_signal,
            'Direction_Accuracy': fm.direction_accuracy,
            'Data_Anomaly_Rate': fm.data_anomaly_rate,
            'Residual_Anomaly_Rate': fm.residual_anomaly_rate
        }

    # Compute metrics by SKU and lag
    results = []
    for (sku, lag), group in df.groupby(['SKU', 'lag']):
        row = {'SKU': sku, 'lag': lag, 'count': len(group)}

        # Skip groups with insufficient data
        if len(group) < 5:
            continue

        # Compute metrics
        for name, fn in metric_fns.items():
            try:
                a = group['Prediction_Actual'].to_numpy()
                p = group['Prediction_Value'].to_numpy()
                row[name] = fn(a, p)
            except Exception as e:
                print(f"Error computing {name} for {sku}, lag {lag}: {e}")
                row[name] = np.nan

        results.append(row)

    # Create DataFrame and round numeric columns
    result_df = pd.DataFrame(results).sort_values(['SKU', 'lag'])
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)

    return result_df


def plot_metric_by_lag(
        lag_metrics: pd.DataFrame,
        metric: str,
        output_dir: str = '.',
        sku: Optional[str] = None
) -> str:
    """
    Plot a metric's value by forecast lag.

    Parameters:
        lag_metrics (pd.DataFrame): DataFrame with metrics by lag
        metric (str): Metric name to plot
        output_dir (str, optional): Directory to save visualization
        sku (str, optional): SKU to filter on. If None, uses all SKUs.

    Returns:
        str: Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Filter by SKU if specified
    if sku and 'SKU' in lag_metrics.columns:
        filtered_df = lag_metrics[lag_metrics['SKU'] == sku].copy()
        title_prefix = f"{sku} - "
    else:
        filtered_df = lag_metrics.copy()
        title_prefix = ""

    # Skip if metric not in DataFrame
    if metric not in filtered_df.columns:
        return None

    # Create figure
    plt.figure(figsize=(10, 6))

    # Determine if we need to group by SKU
    if 'SKU' in filtered_df.columns and sku is None:
        # Plot lines for each SKU
        for sku_name, sku_group in filtered_df.groupby('SKU'):
            plt.plot(sku_group['lag'], sku_group[metric], marker='o', label=sku_name)
        plt.legend(title='SKU', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # Plot single line
        plt.plot(filtered_df['lag'], filtered_df[metric], marker='o', linewidth=2)

        # Add count as bubble size
        if 'count' in filtered_df.columns:
            sizes = filtered_df['count'] / filtered_df['count'].max() * 100
            plt.scatter(filtered_df['lag'], filtered_df[metric], s=sizes, alpha=0.5)

    # Add title and labels
    plt.title(f"{title_prefix}{metric} by Forecast Horizon", fontsize=14)
    plt.xlabel('Forecast Horizon (months)', fontsize=12)
    plt.ylabel(metric, fontsize=12)

    # Format x-axis as integers
    plt.xticks(sorted(filtered_df['lag'].unique()))

    # Add reference line at 0 or 0.5 based on metric type
    if metric.lower() in fm.BIAS_METRICS:
        plt.axhline(0, color='r', linestyle='--', alpha=0.7)
    elif metric.lower() in fm.PERFORMANCE_METRICS:
        plt.axhline(0.5, color='r', linestyle='--', alpha=0.7)

    # Add business context annotation
    if metric.lower() == 'mean_bias':
        plt.figtext(
            0.5, -0.05,
            "Business Impact: Positive bias (over-prediction) leads to excess inventory; " +
            "negative bias (under-prediction) leads to stockouts",
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray")
        )
    elif metric.lower() == 'direction_accuracy':
        plt.figtext(
            0.5, -0.05,
            "Business Impact: Lower direction accuracy at longer horizons " +
            "affects long-term planning reliability",
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray")
        )

    plt.tight_layout()

    # Save the figure
    filename = f"{metric.lower()}_by_lag"
    if sku:
        filename = f"{sku}_{filename}"

    output_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_lag_heatmap(
        lag_metrics: pd.DataFrame,
        metric: str,
        output_dir: str = '.'
) -> str:
    """
    Create a heatmap of a metric by SKU and lag.

    Parameters:
        lag_metrics (pd.DataFrame): DataFrame with metrics by SKU and lag
        metric (str): Metric name to visualize
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if required columns exist
    if 'SKU' not in lag_metrics.columns or 'lag' not in lag_metrics.columns:
        return None

    # Skip if metric not in DataFrame
    if metric not in lag_metrics.columns:
        return None

    # Pivot data for heatmap
    pivot_data = lag_metrics.pivot(index='SKU', columns='lag', values=metric)

    # Create figure
    plt.figure(figsize=(10, max(6, len(pivot_data) * 0.4)))

    # Determine colormap based on metric type
    if metric.lower() in fm.PERFORMANCE_METRICS:
        # For performance metrics, higher is better
        cmap = 'RdYlGn'
    else:
        # For error metrics, lower is better
        cmap = 'RdYlGn_r'

    # Create heatmap
    sns.heatmap(
        pivot_data,
        cmap=cmap,
        linewidths=.5,
        linecolor='gray',
        annot=True,
        fmt='.2f',
        center=0 if metric.lower() in fm.BIAS_METRICS else None
    )

    # Add title and labels
    plt.title(f"{metric} by SKU and Forecast Horizon", fontsize=14)
    plt.ylabel("SKU", fontsize=12)
    plt.xlabel("Forecast Horizon (months)", fontsize=12)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, f"{metric.lower()}_lag_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def generate_lag_report(
        lag_metrics: pd.DataFrame,
        output_file: str = None
) -> str:
    """
    Generate a report analyzing how metrics change with forecast horizon.

    Parameters:
        lag_metrics (pd.DataFrame): DataFrame with metrics by lag
        output_file (str, optional): Path to save the report.
            If None, returns the report as a string.

    Returns:
        str: Report text or path to saved report
    """
    # Build report
    report = []
    report.append("=================================================")
    report.append("FORECAST HORIZON ANALYSIS REPORT")
    report.append("=================================================")
    report.append("")

    # Get metric columns
    metrics = [col for col in lag_metrics.columns if col not in ['lag', 'SKU', 'count']]

    # Overview section
    report.append("OVERVIEW")
    report.append("-------------------------------------------------")
    report.append(f"Analyzed forecast horizons: {min(lag_metrics['lag'])} to {max(lag_metrics['lag'])} months")
    report.append(f"Total forecast horizons: {lag_metrics['lag'].nunique()}")
    if 'SKU' in lag_metrics.columns:
        report.append(f"Total SKUs: {lag_metrics['SKU'].nunique()}")
    report.append("")

    # Analyze each metric
    report.append("METRIC TRENDS BY FORECAST HORIZON")
    report.append("-------------------------------------------------")

    for metric in metrics:
        report.append(f"## {metric}")

        # Group by lag and calculate average
        lag_avg = lag_metrics.groupby('lag')[metric].mean().reset_index()

        # Calculate trend
        first_val = lag_avg.iloc[0][metric]
        last_val = lag_avg.iloc[-1][metric]
        change = last_val
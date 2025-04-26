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
        change = last_val - first_val
        pct_change = change / first_val * 100 if first_val != 0 else float('inf')

        # Determine if higher is better
        higher_is_better = metric.lower() in [m.lower() for m in fm.PERFORMANCE_METRICS]

        # Interpret trend
        if higher_is_better:
            if change > 0:
                trend_desc = "IMPROVING"
                trend_impact = "POSITIVE"
            elif change < 0:
                trend_desc = "DEGRADING"
                trend_impact = "NEGATIVE"
            else:
                trend_desc = "STABLE"
                trend_impact = "NEUTRAL"
        else:
            if change > 0:
                trend_desc = "INCREASING"
                trend_impact = "NEGATIVE"
            elif change < 0:
                trend_desc = "DECREASING"
                trend_impact = "POSITIVE"
            else:
                trend_desc = "STABLE"
                trend_impact = "NEUTRAL"

        # Add metric data
        report.append(f"First horizon value: {first_val:.3f}")
        report.append(f"Last horizon value: {last_val:.3f}")
        report.append(f"Change: {change:.3f} ({pct_change:.1f}%)")
        report.append(f"Trend: {trend_desc}")
        report.append(f"Business Impact: {trend_impact}")

        # Add business interpretation
        if metric.lower() == 'mean_bias':
            report.append("\nBusiness Interpretation:")
            if abs(first_val) < abs(last_val):
                report.append("Bias increases with forecast horizon, leading to greater inventory imbalances for longer-term forecasts.")
                report.append("Recommendation: Adjust safety stock levels for longer horizons to compensate for increased bias.")
            else:
                report.append("Bias is well-controlled across forecast horizons, maintaining consistent inventory balance.")
                report.append("Recommendation: Continue current approach with confidence in long-term forecasts.")

        elif metric.lower() == 'direction_accuracy':
            report.append("\nBusiness Interpretation:")
            if last_val < 0.5:
                report.append("Direction accuracy drops below 50% for long horizons, making long-term trend predictions unreliable.")
                report.append("Recommendation: Limit strategic decisions to shorter horizons where direction accuracy is above 50%.")
            elif last_val < first_val:
                report.append("Direction accuracy degrades with longer horizons but remains above chance level.")
                report.append("Recommendation: Use directional indicators for short-term planning but be cautious with longer horizons.")
            else:
                report.append("Direction accuracy is consistent across horizons, providing reliable trend predictions.")
                report.append("Recommendation: Confidently use trend signals for capacity planning across all horizons.")

        report.append("\n" + "-" * 50 + "\n")

    # Add recommendations section
    report.append("RECOMMENDATIONS")
    report.append("-------------------------------------------------")

    # Generate specific recommendations based on metrics
    bias_metrics = [m for m in metrics if m.lower() in [b.lower() for b in fm.BIAS_METRICS]]
    if bias_metrics:
        # Analyze bias trends
        bias_trends = []
        for metric in bias_metrics:
            lag_avg = lag_metrics.groupby('lag')[metric].mean()
            if lag_avg.iloc[-1] > lag_avg.iloc[0]:
                bias_trends.append("increasing")
            elif lag_avg.iloc[-1] < lag_avg.iloc[0]:
                bias_trends.append("decreasing")
            else:
                bias_trends.append("stable")

        # Generate bias recommendations
        if "increasing" in bias_trends:
            report.append("1. Adjust confidence intervals for longer horizons to account for increasing bias")
            report.append("2. Review forecasting models for systematic errors that compound over time")
            report.append("3. Consider using different models for short-term vs. long-term forecasting")
        else:
            report.append("1. Current bias control methods are effective across forecast horizons")
            report.append("2. Continue monitoring bias trends for early detection of model degradation")

    # Generate performance metric recommendations
    if "Direction_Accuracy" in metrics or "direction_accuracy" in metrics:
        metric_name = "Direction_Accuracy" if "Direction_Accuracy" in metrics else "direction_accuracy"
        lag_avg = lag_metrics.groupby('lag')[metric_name].mean()

        if lag_avg.iloc[-1] < 0.6:
            report.append("4. For long-term planning, focus on overall magnitude rather than directional changes")
            report.append("5. Implement a tiered confidence system that reduces reliance on directional accuracy for horizons > 6 months")
        else:
            report.append("4. The model provides reliable directional guidance across all horizons")
            report.append("5. Use directional signals confidently for trend-based decisions")

    # Generate general recommendations
    report.append("6. Stratify forecast evaluation by horizon, with different metrics emphasized at different horizons")
    report.append("7. For critical business decisions, weight near-term forecast performance more heavily")
    report.append("8. Establish horizon-specific thresholds for performance metrics")

    # Join report lines
    report_text = "\n".join(report)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        return output_file

    return report_text
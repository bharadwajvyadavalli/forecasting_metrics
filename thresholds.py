"""
Thresholds Module

This module provides utilities for calculating and applying performance thresholds
to forecast metrics with clear business context.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

import forecast_metrics as fm


def calculate_thresholds(
        metrics_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        performance_metrics: Optional[List[str]] = None,
        symmetric_metrics: Optional[List[str]] = None,
        percentiles: Tuple[float, float] = (0.25, 0.75),
        sku_level: bool = True
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calculate performance thresholds for metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        metrics (List[str], optional): List of metrics to calculate thresholds for.
            If None, uses all numeric columns except SKU and date columns.
        performance_metrics (List[str], optional): List of metrics where higher values are better.
            If None, defaults to fm.PERFORMANCE_METRICS.
        symmetric_metrics (List[str], optional): List of metrics where absolute value matters.
            If None, defaults to fm.SYMMETRIC_ERROR_METRICS.
        percentiles (Tuple[float, float], optional): Percentiles for thresholds (green, yellow).
            Default is (0.25, 0.75).
        sku_level (bool, optional): Whether to calculate SKU-level thresholds.
            Default is True.

    Returns:
        Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
            If sku_level is True, returns (sku_thresholds, global_thresholds).
            If sku_level is False, returns global_thresholds only.
    """
    # Default lists if not provided
    if performance_metrics is None:
        performance_metrics = fm.PERFORMANCE_METRICS

    if symmetric_metrics is None:
        symmetric_metrics = fm.SYMMETRIC_ERROR_METRICS

    # Identify metrics if not provided
    if metrics is None:
        exclude_cols = ['SKU', 'Actual_Month', 'Prediction_Month']
        metrics = [
            col for col in metrics_df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols
        ]

    # Determine error metrics (those not in performance_metrics)
    error_metrics = [m for m in metrics if m.lower() not in [p.lower() for p in performance_metrics]]

    # Convert symmetric metrics to lowercase for case-insensitive matching
    symmetric_lower = [s.lower() for s in symmetric_metrics]

    # Compute global thresholds
    global_rows = []

    # Process error metrics (lower is better)
    for m in error_metrics:
        series = metrics_df[m].dropna()

        # Apply absolute value for symmetric metrics
        if m.lower() in symmetric_lower:
            series = series.abs()

        # Calculate percentiles
        q_low, q_high = series.quantile(percentiles)

        global_rows.append({
            'Metric': m,
            'Green': round(q_low, 3),
            'Yellow': round(q_high, 3),
            'Red_Condition': f'> {q_high:.3f}',
            'Business_Impact': get_metric_impact(m)
        })

    # Process performance metrics (higher is better)
    for m in [m for m in metrics if m.lower() in [p.lower() for p in performance_metrics]]:
        series = metrics_df[m].dropna()
        q_low, q_high = series.quantile(percentiles)

        global_rows.append({
            'Metric': m,
            'Green': round(q_high, 3),
            'Yellow': round(q_low, 3),
            'Red_Condition': f'< {q_low:.3f}',
            'Business_Impact': get_metric_impact(m)
        })

    global_thresholds = pd.DataFrame(global_rows)

    # If SKU-level thresholds not needed, return global only
    if not sku_level:
        return global_thresholds

    # Compute SKU-level thresholds
    sku_rows = []

    for sku, sku_df in metrics_df.groupby('SKU'):
        # Process error metrics (lower is better)
        for m in error_metrics:
            if m not in sku_df.columns:
                continue

            series = sku_df[m].dropna()

            # Apply absolute value for symmetric metrics
            if m.lower() in symmetric_lower:
                series = series.abs()

            # Calculate percentiles
            q_low, q_high = series.quantile(percentiles)

            sku_rows.append({
                'SKU': sku,
                'Metric': m,
                'Green': round(q_low, 3),
                'Yellow': round(q_high, 3),
                'Red_Condition': f'> {q_high:.3f}',
                'Business_Impact': get_metric_impact(m, sku_df)
            })

        # Process performance metrics (higher is better)
        for m in [m for m in metrics if m.lower() in [p.lower() for p in performance_metrics]]:
            if m not in sku_df.columns:
                continue

            series = sku_df[m].dropna()
            q_low, q_high = series.quantile(percentiles)

            sku_rows.append({
                'SKU': sku,
                'Metric': m,
                'Green': round(q_high, 3),
                'Yellow': round(q_low, 3),
                'Red_Condition': f'< {q_low:.3f}',
                'Business_Impact': get_metric_impact(m, sku_df)
            })

    sku_thresholds = pd.DataFrame(sku_rows)

    return sku_thresholds, global_thresholds


def apply_thresholds(
        metrics_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        sku_specific: bool = True
) -> pd.DataFrame:
    """
    Apply thresholds to metrics and add performance level columns.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        thresholds_df (pd.DataFrame): DataFrame containing thresholds
        sku_specific (bool, optional): Whether to use SKU-specific thresholds.
            Default is True.

    Returns:
        pd.DataFrame: DataFrame with performance level columns added
    """
    result_df = metrics_df.copy()

    # Get list of metrics in thresholds_df
    metrics = thresholds_df['Metric'].unique()

    # Process each metric
    for metric in metrics:
        if metric not in result_df.columns:
            continue

        # Add performance level column
        perf_col = f"{metric}_Performance"
        result_df[perf_col] = None

        # Process each SKU
        for sku in result_df['SKU'].unique():
            # Get data for this SKU
            sku_mask = result_df['SKU'] == sku

            # Get thresholds for this metric
            metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

            # If using SKU-specific thresholds and they exist, use them
            if sku_specific and 'SKU' in metric_thresh.columns:
                sku_thresh = metric_thresh[metric_thresh['SKU'] == sku]
                if not sku_thresh.empty:
                    metric_thresh = sku_thresh

            # If no matching thresholds found, skip
            if metric_thresh.empty:
                continue

            # Get threshold values
            green_val = metric_thresh.iloc[0]['Green']
            yellow_val = metric_thresh.iloc[0]['Yellow']

            # Determine if higher or lower is better
            red_condition = metric_thresh.iloc[0]['Red_Condition']
            higher_is_better = '<' in red_condition

            # Apply thresholds
            if higher_is_better:
                # Higher is better
                result_df.loc[sku_mask & (result_df[metric] >= green_val), perf_col] = 'Green'
                result_df.loc[sku_mask & (result_df[metric] < green_val) &
                              (result_df[metric] >= yellow_val), perf_col] = 'Yellow'
                result_df.loc[sku_mask & (result_df[metric] < yellow_val), perf_col] = 'Red'
            else:
                # Lower is better
                result_df.loc[sku_mask & (result_df[metric] <= green_val), perf_col] = 'Green'
                result_df.loc[sku_mask & (result_df[metric] > green_val) &
                              (result_df[metric] <= yellow_val), perf_col] = 'Yellow'
                result_df.loc[sku_mask & (result_df[metric] > yellow_val), perf_col] = 'Red'

    return result_df


def get_performance_level(
        value: float,
        metric: str,
        thresholds_df: pd.DataFrame,
        sku: Optional[str] = None
) -> str:
    """
    Get performance level for a single metric value.

    Parameters:
        value (float): Metric value
        metric (str): Metric name
        thresholds_df (pd.DataFrame): DataFrame containing thresholds
        sku (str, optional): SKU to use for thresholds. If None, uses global thresholds.

    Returns:
        str: Performance level ('Green', 'Yellow', or 'Red')
    """
    # Get thresholds for this metric
    metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

    # If SKU specified and SKU-specific thresholds exist, use them
    if sku and 'SKU' in metric_thresh.columns:
        sku_thresh = metric_thresh[metric_thresh['SKU'] == sku]
        if not sku_thresh.empty:
            metric_thresh = sku_thresh

    # If no matching thresholds found, return None
    if metric_thresh.empty:
        return None

    # Get threshold values
    green_val = metric_thresh.iloc[0]['Green']
    yellow_val = metric_thresh.iloc[0]['Yellow']

    # Determine if higher or lower is better
    red_condition = metric_thresh.iloc[0]['Red_Condition']
    higher_is_better = '<' in red_condition

    # Determine performance level
    if higher_is_better:
        if value >= green_val:
            return 'Green'
        elif value >= yellow_val:
            return 'Yellow'
        else:
            return 'Red'
    else:
        if value <= green_val:
            return 'Green'
        elif value <= yellow_val:
            return 'Yellow'
        else:
            return 'Red'


def get_metric_impact(metric: str, data: Optional[pd.DataFrame] = None) -> str:
    """
    Get business impact description for a metric.

    Parameters:
        metric (str): Metric name
        data (pd.DataFrame, optional): Data to analyze for specific impacts

    Returns:
        str: Business impact description
    """
    # Map metrics to their business impacts
    impacts = {
        'mean_bias': 'Affects inventory levels and resource allocation efficiency',
        'tracking_signal': 'Indicates persistent forecast misalignment requiring model adjustment',
        'residual_counts': 'Signals systematic over/under-prediction affecting planning reliability',
        'data_anomaly_rate': 'Highlights data quality issues requiring investigation',
        'residual_anomaly_rate': 'Shows forecast sensitivity to outliers affecting reliability',
        'direction_accuracy': 'Critical for trend-based decisions and capacity planning',
        'turning_point_f1': 'Essential for detecting market shifts and strategy adjustments',
        'sliding_jsd': 'Measures distribution stability affecting long-term planning',
        'crps': 'Evaluates probabilistic forecast quality for risk assessment'
    }

    # Normalize metric name (case-insensitive matching)
    metric_lower = metric.lower()
    matched_metric = next((m for m in impacts.keys() if m.lower() == metric_lower), None)

    # If metric not found in impacts, provide a default impact
    if matched_metric is None:
        return 'Affects forecast quality and business decisions'

    return impacts[matched_metric]


def generate_threshold_report(
        metrics_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        output_file: str = None
) -> str:
    """
    Generate a report summarizing metric performance against thresholds.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        thresholds_df (pd.DataFrame): DataFrame containing thresholds
        output_file (str, optional): Path to save the report.
            If None, returns the report as a string.

    Returns:
        str: Report text or path to saved report
    """
    # Build report
    report = []
    report.append("=================================================")
    report.append("FORECAST METRICS THRESHOLD REPORT")
    report.append("=================================================")
    report.append("")

    # Process each metric
    for metric in thresholds_df['Metric'].unique():
        if metric not in metrics_df.columns:
            continue

        # Get threshold and determine if higher is better
        thresh = thresholds_df[thresholds_df['Metric'] == metric]
        if thresh.empty:
            continue

        red_condition = thresh.iloc[0]['Red_Condition']
        higher_is_better = '<' in red_condition

        # Get metric statistics
        metric_mean = metrics_df[metric].mean()
        metric_min = metrics_df[metric].min()
        metric_max = metrics_df[metric].max()

        # Get threshold values
        green_val = thresh.iloc[0]['Green']
        yellow_val = thresh.iloc[0]['Yellow']

        # Get performance level
        if higher_is_better:
            if metric_mean >= green_val:
                performance = "GREEN"
            elif metric_mean >= yellow_val:
                performance = "YELLOW"
            else:
                performance = "RED"
        else:
            if metric_mean <= green_val:
                performance = "GREEN"
            elif metric_mean <= yellow_val:
                performance = "YELLOW"
            else:
                performance = "RED"

        # Build metric section
        report.append(f"## {metric} - {performance}")
        report.append(f"Average: {metric_mean:.3f} (Range: {metric_min:.3f} to {metric_max:.3f})")

        if higher_is_better:
            report.append(f"Thresholds: Green ≥ {green_val:.3f}, Yellow ≥ {yellow_val:.3f}, Red < {yellow_val:.3f}")
        else:
            report.append(f"Thresholds: Green ≤ {green_val:.3f}, Yellow ≤ {yellow_val:.3f}, Red > {yellow_val:.3f}")

        # Add business impact
        if 'Business_Impact' in thresh.columns:
            impact = thresh.iloc[0]['Business_Impact']
            report.append(f"Business Impact: {impact}")

        # Add SKU breakdown
        sku_stats = metrics_df.groupby('SKU')[metric].mean().sort_values()

        if higher_is_better:
            worst_skus = sku_stats.head(3)
            best_skus = sku_stats.tail(3)
        else:
            worst_skus = sku_stats.tail(3)
            best_skus = sku_stats.head(3)

        report.append("\nBest Performing SKUs:")
        for sku, value in best_skus.items():
            report.append(f"- {sku}: {value:.3f}")

        report.append("\nWorst Performing SKUs:")
        for sku, value in worst_skus.items():
            report.append(f"- {sku}: {value:.3f}")

        report.append("\n" + "-" * 50 + "\n")

    # Join report lines
    report_text = "\n".join(report)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        return output_file

    return report_text
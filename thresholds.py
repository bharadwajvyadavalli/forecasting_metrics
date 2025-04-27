"""
Simplified Thresholds Module

This module provides utilities for calculating performance thresholds for forecast metrics.
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
    """Calculate performance thresholds for metrics."""
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
        if m not in metrics_df.columns:
            continue

        series = metrics_df[m].dropna()
        if series.empty:
            continue

        # Apply absolute value for symmetric metrics
        if m.lower() in symmetric_lower:
            series = series.abs()

        try:
            # Calculate percentiles
            q_low = float(series.quantile(percentiles[0]))
            q_high = float(series.quantile(percentiles[1]))

            global_rows.append({
                'Metric': m,
                'Green': round(q_low, 3),
                'Yellow': round(q_high, 3),
                'Red_Condition': f'> {q_high:.3f}',
                'Business_Impact': get_metric_impact(m)
            })
        except Exception as e:
            print(f"Error calculating thresholds for {m}: {e}")
            # Use default values if calculation fails
            global_rows.append({
                'Metric': m,
                'Green': 0.0,
                'Yellow': 0.5,
                'Red_Condition': '> 0.500',
                'Business_Impact': get_metric_impact(m)
            })

    # Process performance metrics (higher is better)
    for m in [m for m in metrics if m.lower() in [p.lower() for p in performance_metrics]]:
        if m not in metrics_df.columns:
            continue

        series = metrics_df[m].dropna()
        if series.empty:
            continue

        try:
            q_low = float(series.quantile(percentiles[0]))
            q_high = float(series.quantile(percentiles[1]))

            global_rows.append({
                'Metric': m,
                'Green': round(q_high, 3),
                'Yellow': round(q_low, 3),
                'Red_Condition': f'< {q_low:.3f}',
                'Business_Impact': get_metric_impact(m)
            })
        except Exception as e:
            print(f"Error calculating thresholds for {m}: {e}")
            # Use default values if calculation fails
            global_rows.append({
                'Metric': m,
                'Green': 0.6,
                'Yellow': 0.4,
                'Red_Condition': '< 0.400',
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
            if series.empty:
                continue

            # Apply absolute value for symmetric metrics
            if m.lower() in symmetric_lower:
                series = series.abs()

            try:
                # Calculate percentiles
                q_low = float(series.quantile(percentiles[0]))
                q_high = float(series.quantile(percentiles[1]))

                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': round(q_low, 3),
                    'Yellow': round(q_high, 3),
                    'Red_Condition': f'> {q_high:.3f}',
                    'Business_Impact': get_metric_impact(m, sku_df)
                })
            except Exception as e:
                print(f"Error calculating thresholds for {m} with SKU {sku}: {e}")
                # Use default values if calculation fails
                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': 0.0,
                    'Yellow': 0.5,
                    'Red_Condition': '> 0.500',
                    'Business_Impact': get_metric_impact(m, sku_df)
                })

        # Process performance metrics (higher is better)
        for m in [m for m in metrics if m.lower() in [p.lower() for p in performance_metrics]]:
            if m not in sku_df.columns:
                continue

            series = sku_df[m].dropna()
            if series.empty:
                continue

            try:
                q_low = float(series.quantile(percentiles[0]))
                q_high = float(series.quantile(percentiles[1]))

                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': round(q_high, 3),
                    'Yellow': round(q_low, 3),
                    'Red_Condition': f'< {q_low:.3f}',
                    'Business_Impact': get_metric_impact(m, sku_df)
                })
            except Exception as e:
                print(f"Error calculating thresholds for {m} with SKU {sku}: {e}")
                # Use default values if calculation fails
                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': 0.6,
                    'Yellow': 0.4,
                    'Red_Condition': '< 0.400',
                    'Business_Impact': get_metric_impact(m, sku_df)
                })

    sku_thresholds = pd.DataFrame(sku_rows)

    return sku_thresholds, global_thresholds


def get_metric_impact(metric: str, data: Optional[pd.DataFrame] = None) -> str:
    """Get business impact description for a metric."""
    # Map metrics to their business impacts
    impacts = {
        'mean_bias': 'Affects inventory levels and resource allocation efficiency',
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
"""
Simplified Lag-Based Predictions Module

This module provides functionality for analyzing forecast accuracy by forecast horizon (lag).
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

import forecast_metrics as fm


def compute_metrics_by_lag(
        df: pd.DataFrame,
        metric_fns: Dict[str, Callable] = None
) -> pd.DataFrame:
    """Compute metrics grouped by forecast lag (horizon)."""
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
    """Compute metrics grouped by SKU and forecast lag (horizon)."""
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
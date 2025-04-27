"""
Lag-Based Predictions Module
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable

import forecast_metrics as fm


def compute_metrics_by_lag(
        df: pd.DataFrame,
        metric_fns: Dict[str, Callable] = None
) -> pd.DataFrame:
    """
    Compute bias metrics grouped by forecast lag (horizon).

    Lag represents how many months ahead a prediction was made.
    For example, prediction in Jan 2025 for Apr 2025 has lag = 3.
    """
    # Default metric functions if not provided
    if metric_fns is None:
        metric_fns = {
            'Mean_Bias': fm.mean_bias
        }

    # Make a copy of the dataframe
    df = df.copy()

    # Ensure date columns are datetime
    df['Actual_Month'] = pd.to_datetime(df['Actual_Month'])
    df['Prediction_Month'] = pd.to_datetime(df['Prediction_Month'])

    # Calculate lag - negate to get positive values
    df['lag'] = -1 * ((df.Actual_Month.dt.year - df.Prediction_Month.dt.year) * 12 +
                    (df.Actual_Month.dt.month - df.Prediction_Month.dt.month))

    # Filter to include only lags 1-12
    df = df[df['lag'].between(1, 12)]

    # Create container for results
    rows = []

    # Process lag values from 1 to 12
    for lag_value in range(1, 13):
        # Filter data for this specific lag
        lag_data = df[df['lag'] == lag_value]

        # Skip if no data for this lag
        if len(lag_data) == 0:
            continue

        # Create a row with the lag value
        row = {'lag': lag_value}

        # Calculate metrics for this lag
        for name, fn in metric_fns.items():
            try:
                actual_values = lag_data['Prediction_Actual'].to_numpy()
                predicted_values = lag_data['Prediction_Value'].to_numpy()
                row[name] = fn(actual_values, predicted_values)
            except Exception as e:
                row[name] = np.nan

        rows.append(row)

    # Create DataFrame from the results
    if not rows:
        return pd.DataFrame(columns=['lag'] + list(metric_fns.keys()))

    result_df = pd.DataFrame(rows)

    # Round numeric columns for readability
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)

    return result_df


def compute_metrics_by_lag_and_sku(
        df: pd.DataFrame,
        metric_fns: Dict[str, Callable] = None
) -> pd.DataFrame:
    """
    Compute bias metrics grouped by SKU and forecast lag (horizon).

    Lag represents how many months ahead a prediction was made.
    For example, prediction in Jan 2025 for Apr 2025 has lag = 3.
    """
    # Default metric functions if not provided
    if metric_fns is None:
        metric_fns = {
            'Mean_Bias': fm.mean_bias
        }

    # Make a copy of the dataframe
    df = df.copy()

    # Ensure date columns are datetime
    df['Actual_Month'] = pd.to_datetime(df['Actual_Month'])
    df['Prediction_Month'] = pd.to_datetime(df['Prediction_Month'])

    # Calculate lag - negate to get positive values
    df['lag'] = -1 * ((df.Actual_Month.dt.year - df.Prediction_Month.dt.year) * 12 +
                    (df.Actual_Month.dt.month - df.Prediction_Month.dt.month))

    # Filter to include only lags 1-12
    df = df[df['lag'].between(1, 12)]

    # Create container for results
    rows = []

    # Get unique SKUs
    skus = sorted(df['SKU'].unique())

    # Process each SKU
    for sku in skus:
        # Get data for just this SKU
        sku_data = df[df['SKU'] == sku]

        # Process lag values from 1 to 12
        for lag_value in range(1, 13):
            # Filter data for this specific lag
            lag_data = sku_data[sku_data['lag'] == lag_value]

            # Skip if no data for this lag
            if len(lag_data) == 0:
                continue

            # Create a row with the SKU and lag value
            row = {'SKU': sku, 'lag': lag_value}

            # Calculate metrics
            for name, fn in metric_fns.items():
                try:
                    actual_values = lag_data['Prediction_Actual'].to_numpy()
                    predicted_values = lag_data['Prediction_Value'].to_numpy()
                    row[name] = fn(actual_values, predicted_values)
                except Exception as e:
                    row[name] = np.nan

            rows.append(row)

    # Create DataFrame from the results
    if not rows:
        return pd.DataFrame(columns=['SKU', 'lag'] + list(metric_fns.keys()))

    result_df = pd.DataFrame(rows)

    # Round numeric columns for readability
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)

    return result_df
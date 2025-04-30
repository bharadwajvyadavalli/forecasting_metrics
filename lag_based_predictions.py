"""
Lag-Based Predictions Module

This module provides functionality to calculate forecast metrics based on lag (forecast horizon).
Analyzing forecast accuracy by lag helps understand how forecast quality degrades with
longer prediction horizons.
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Optional

import forecast_metrics as fm


def compute_metrics_by_lag_and_sku(
        df: pd.DataFrame,
        metric_fns: Optional[Dict[str, Callable]] = None,
        timeline_output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute forecast metrics grouped by SKU and forecast lag (horizon).

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing forecast data with columns:
        - SKU: Stock Keeping Unit identifier
        - Actual_Month: The month for which the forecast is made
        - Prediction_Month: The month when the forecast was made
        - Prediction_Value: The forecasted value
        - Prediction_Actual: The actual observed value

    metric_fns : Dict[str, Callable], optional
        Dictionary mapping metric names to metric functions.
        Each function should accept arrays of actuals and predictions.
        Default is {'Mean_Bias': fm.mean_bias}

    timeline_output_file : str, optional
        If provided, writes a timeline CSV with prediction details to this path.

    Returns:
    --------
    pd.DataFrame:
        DataFrame with metrics calculated for each SKU and lag combination.
    """
    # Default metric functions if not provided
    if metric_fns is None:
        metric_fns = {
            'Mean_Bias': fm.mean_bias,
            'Turning_Point_F1': fm.turning_point_f1,
            'CRPS': fm.crps
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
    df_filtered = df[df['lag'].between(1, 12)]

    if df_filtered.empty:
        print("Warning: No data found within the 1-12 month lag range.")
        return pd.DataFrame(columns=['SKU', 'lag'] + list(metric_fns.keys()))

    # If timeline output file is specified, write prediction timeline data
    if timeline_output_file and not df_filtered.empty:
        # Create simple timeline dataframe with the required columns
        timeline = df_filtered.copy()

        # Format date columns and create lag name
        timeline['Prediction_Month'] = timeline['Prediction_Month'].dt.strftime('%Y-%m')
        timeline['Actual_Month'] = timeline['Actual_Month'].dt.strftime('%Y-%m')
        timeline['Lag_Name'] = timeline['lag'].apply(lambda x: f"{x}-month ahead")

        # Select columns - don't try to sort
        columns = ['SKU', 'Prediction_Month', 'Lag_Name', 'Actual_Month',
                   'Prediction_Value', 'Prediction_Actual']
        timeline = timeline[columns]

        # Write to CSV without sorting
        timeline.to_csv(timeline_output_file, index=False)
        print(f"Timeline data written to {timeline_output_file}")

    # Create container for results
    rows = []

    # Get unique SKUs
    skus = sorted(df_filtered['SKU'].unique())

    # Process each SKU
    for sku in skus:
        # Get data for just this SKU
        sku_data = df_filtered[df_filtered['SKU'] == sku]

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
                    print(f"Error calculating {name} for SKU {sku}, lag {lag_value}: {str(e)}")
                    row[name] = np.nan

            rows.append(row)

    # Create DataFrame from the results
    if not rows:
        print("Warning: No results were calculated. Check your data and metric functions.")
        return pd.DataFrame(columns=['SKU', 'lag'] + list(metric_fns.keys()))

    result_df = pd.DataFrame(rows)

    # Round numeric columns for readability
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].round(3)

    return result_df
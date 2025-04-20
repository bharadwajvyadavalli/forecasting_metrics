import numpy as np
import pandas as pd
from forecast_metrics_calculations import (
    compute_mean_bias,
    compute_tracking_signal,
    compute_residual_counts,
    # … import any other metric functions you need …
)


def compute_metrics_by_prediction_month(
        df: pd.DataFrame,
        metric_fns: dict[str, callable]
) -> pd.DataFrame:
    """
    For each SKU and Prediction_Month, compute:
      - count of forecasts (horizons)
      - each metric in metric_fns(actuals, preds)
    Returns a DataFrame with columns:
      ['SKU','Prediction_Month','count',<metrics...>]
    """
    df = df.copy()
    df['Actual_Month'] = pd.to_datetime(df['Actual_Month'], format='%Y-%m')
    df['Prediction_Month'] = pd.to_datetime(df['Prediction_Month'], format='%Y-%m')
    df['lag'] = (
            df.Prediction_Month.dt.year * 12 + df.Prediction_Month.dt.month
            - df.Actual_Month.dt.year * 12 - df.Actual_Month.dt.month
    )

    rows = []
    for (sku, pred_month), grp in df.groupby(['SKU', 'Prediction_Month']):
        a = grp['Prediction_Actual'].to_numpy()
        p = grp['Prediction_Value'].to_numpy()
        row = {
            'SKU': sku,
            'Prediction_Month': pred_month,
            'count': len(grp),
        }
        for name, fn in metric_fns.items():
            row[name] = fn(a, p)
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(['SKU', 'Prediction_Month'])
        .reset_index(drop=True)
    )


if __name__ == '__main__':
    # 1. Load your forecast data
    df_pred = pd.read_csv(
        'sample_data.csv',
        parse_dates=['Actual_Month', 'Prediction_Month']
    )

    # 2. Define metrics to compute per SKU & Prediction_Month
    metrics_to_run = {
        'Mean_Bias': compute_mean_bias,
        'Tracking Signal': compute_tracking_signal,
        'Residual Counts': compute_residual_counts,
        # … add more metric functions as needed …
    }

    # 3. Compute the summary
    summary = compute_metrics_by_prediction_month(df_pred, metrics_to_run)
    numeric_cols = summary.select_dtypes(include=[np.number]).columns
    summary[numeric_cols] = summary[numeric_cols].round(2)
    # 4. Inspect or save
    summary.to_csv('lag_based_predictions.csv', index=False)
    print("Saved lag based predictions to lag_based_predictions.csv")
import numpy as np
import pandas as pd


def generate_date_list(start_ym="2025-01", n_months=24):
    """
    Creates a list of YYYY-MM strings starting from `start_ym` for `n_months`.
    Example: start_ym='2025-01', n_months=24 -> ['2025-01', '2025-02', ..., '2026-12'].
    """
    start_date = pd.to_datetime(start_ym, format="%Y-%m")
    date_range = pd.date_range(start=start_date, periods=n_months, freq="MS")
    return [d.strftime("%Y-%m") for d in date_range]


def generate_baseline_data_ym(num_skus=10, n_months=24, start_ym="2025-01", seed=42):
    """
    1) Generates 'Actual' data for `num_skus` SKUs over `n_months` months in YYYY-MM format.
    2) Applies a constant monthly trend percentage (±2%) for each SKU.
    3) Creates forecasts up to 12 months ahead within ±5% noise.

    Returns DataFrame with columns:
    [SKU, Actual_Month, Actual_Value, Prediction_Month, Prediction_Value, Prediction_Actual].
    """
    np.random.seed(seed)

    # Generate actual months list
    all_months = generate_date_list(start_ym, n_months)

    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    actual_rows = []

    for sku in skus:
        # Initialize actual value and trend percentage
        val = 50 + np.random.normal(0, 10)
        trend_pct = np.random.uniform(-0.02, 0.02)  # monthly trend ±2%

        for ym in all_months:
            actual_rows.append([sku, ym, max(val, 0)])
            # Update for next month
            val = val * (1 + trend_pct) + np.random.normal(0, 5)

    df_actual = pd.DataFrame(actual_rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    df_actual["dt"] = pd.to_datetime(df_actual["Actual_Month"], format="%Y-%m")
    dt_set = set(df_actual["dt"])

    # Build forecasts with ±5% noise
    forecast_rows = []
    for row in df_actual.itertuples(index=False):
        for horizon in range(1, 13):
            future_dt = row.dt + pd.DateOffset(months=horizon)
            if future_dt in dt_set:
                future_val = df_actual.loc[df_actual["dt"] == future_dt, "Actual_Value"].values[0]
                noise_pct = np.random.uniform(-0.05, 0.05)
                pred_val = future_val * (1 + noise_pct)
                forecast_rows.append([
                    row.SKU,
                    row.Actual_Month,
                    row.Actual_Value,
                    future_dt.strftime("%Y-%m"),
                    pred_val,
                    future_val
                ])

    df_pred = pd.DataFrame(
        forecast_rows,
        columns=["SKU", "Actual_Month", "Actual_Value", "Prediction_Month", "Prediction_Value", "Prediction_Actual"]
    )
    return df_pred


def inject_anomalies_and_bias(
    df_pred,
    data_anomaly_prob=0.05,
    data_anomaly_factor=2.0,
    residual_anomaly_prob=0.05,
    residual_anomaly_factor=3.0,
    bias_sku_list=("SKU_1", "SKU_2"),
    bias_month_range=("2025-10", "2025-12"),
    bias_shift=25.0,
    seed=999
):
    """
    Inject data anomalies, residual anomalies, and systematic bias.
    """
    np.random.seed(seed)
    df = df_pred.copy()

    # Data anomalies on Prediction_Actual
    mask_da = np.random.rand(len(df)) < data_anomaly_prob
    for idx in df[mask_da].index:
        if np.random.rand() < 0.5:
            df.at[idx, "Prediction_Actual"] *= data_anomaly_factor
        else:
            df.at[idx, "Prediction_Actual"] /= data_anomaly_factor

    # Residual anomalies on Prediction_Value
    mask_ra = np.random.rand(len(df)) < residual_anomaly_prob
    signs = np.random.choice([1, -1], size=mask_ra.sum())
    df.loc[mask_ra, "Prediction_Value"] += signs * residual_anomaly_factor

    # Systematic bias for specified SKUs/months
    df["AM_dt"] = pd.to_datetime(df["Actual_Month"], format="%Y-%m")
    start_dt = pd.to_datetime(bias_month_range[0], format="%Y-%m")
    end_dt = pd.to_datetime(bias_month_range[1], format="%Y-%m")
    for idx, row in df.iterrows():
        if row["SKU"] in bias_sku_list and start_dt <= row["AM_dt"] <= end_dt:
            df.at[idx, "Prediction_Value"] += bias_shift
    df.drop(columns="AM_dt", inplace=True)

    return df


if __name__ == "__main__":
    df_pred = generate_baseline_data_ym()
    df_aug = inject_anomalies_and_bias(df_pred)
    df_aug["Actual_Value"] = df_aug["Actual_Value"].round().astype(int)
    df_aug["Prediction_Value"] = df_aug["Prediction_Value"].round().astype(int)
    df_aug["Prediction_Actual"] = df_aug["Prediction_Actual"].round().astype(int)
    df_aug.to_csv("sample_data.csv", index=False)
    print("sample_data.csv written with ±2% trend and ±5% forecast noise.")

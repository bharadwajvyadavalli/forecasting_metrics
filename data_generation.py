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
    1) Generates 'Actual' data for `num_skus` SKUs over `n_months` months in YYYY-MM format, starting from `start_ym`.
    2) For each (SKU, Actual_Month), creates naive forecasts for up to 12 months ahead IF that future date is within the 24-month range.

    Returns a DataFrame (df_pred) with columns:
      [SKU, Actual_Month, Actual_Value, Prediction_Month, Prediction_Value, Prediction_Actual].
    """
    np.random.seed(seed)

    # Create a date list for the 24 months
    all_months = generate_date_list(start_ym, n_months)
    
    # Generate Actual data
    skus = [f"SKU_{i+1}" for i in range(num_skus)]
    
    actual_rows = []
    for sku in skus:
        # Random baseline & trend
        baseline = 100 + np.random.normal(0, 10)
        trend = np.random.normal(0, 2)
        
        for i, ym in enumerate(all_months):
            month_idx = i + 1  # 1..24
            val = baseline + (trend * month_idx) + np.random.normal(0, 5)
            val = max(val, 0)  # enforce non-negative
            actual_rows.append([sku, ym, val])
    
    df_actual = pd.DataFrame(actual_rows, columns=["SKU", "Actual_Month", "Actual_Value"])
    # Convert to datetime for easy offset checks
    df_actual["dt"] = pd.to_datetime(df_actual["Actual_Month"], format="%Y-%m")
    
    dt_set = set(df_actual["dt"].unique())  # set of valid datetimes
    forecast_rows = []
    
    # Naive forecasts up to 12 months ahead
    for row in df_actual.itertuples(index=False):
        origin_sku = row.SKU
        origin_ym = row.Actual_Month
        origin_val = row.Actual_Value
        origin_dt = row.dt
        
        for horizon in range(1, 13):
            future_dt = origin_dt + pd.DateOffset(months=horizon)
            if future_dt in dt_set:
                # future's actual
                future_val = df_actual.loc[df_actual["dt"] == future_dt, "Actual_Value"].values[0]
                # naive forecast = future_val + small random noise
                pred_val = future_val + np.random.normal(0, 5)
                
                forecast_rows.append([
                    origin_sku,
                    origin_ym,
                    origin_val,
                    future_dt.strftime("%Y-%m"),
                    pred_val,
                    future_val
                ])
    
    df_pred = pd.DataFrame(forecast_rows, columns=[
        "SKU",
        "Actual_Month",
        "Actual_Value",
        "Prediction_Month",
        "Prediction_Value",
        "Prediction_Actual"
    ])
    
    return df_pred

def inject_anomalies_and_bias(
    df_pred,
    data_anomaly_prob=0.05,
    data_anomaly_factor=5.0,
    residual_anomaly_prob=0.05,
    residual_anomaly_factor=30.0,
    bias_sku_list=("SKU_1", "SKU_2"),
    bias_month_range=("2025-10", "2025-12"),  # origin months for applying bias
    bias_shift=25.0,
    seed=999
):
    """
    Inject anomalies & systematic bias spikes into df_pred:
    
    1) Data anomalies: ~data_anomaly_prob fraction of rows -> multiply/divide 'Prediction_Actual' by data_anomaly_factor.
    2) Residual anomalies: ~residual_anomaly_prob fraction of rows -> add +/-residual_anomaly_factor to 'Prediction_Value'.
    3) Systematic bias: For SKUs in bias_sku_list AND Actual_Month in [bias_month_range], add bias_shift to 'Prediction_Value'.

    Returns a new DataFrame (floats).
    """
    np.random.seed(seed)
    df = df_pred.copy()
    
    # 1) Data anomalies
    mask_data_anom = np.random.rand(len(df)) < data_anomaly_prob
    for i in df[mask_data_anom].index:
        if np.random.rand() < 0.5:
            df.at[i, "Prediction_Actual"] *= data_anomaly_factor
        else:
            df.at[i, "Prediction_Actual"] /= data_anomaly_factor
    
    # 2) Residual anomalies
    mask_res_anom = np.random.rand(len(df)) < residual_anomaly_prob
    signs = np.random.choice([1, -1], size=mask_res_anom.sum())
    df.loc[mask_res_anom, "Prediction_Value"] += signs * residual_anomaly_factor
    
    # 3) Bias for certain SKUs & origin months
    df["AM_dt"] = pd.to_datetime(df["Actual_Month"], format="%Y-%m")
    start_dt = pd.to_datetime(bias_month_range[0], format="%Y-%m")
    end_dt   = pd.to_datetime(bias_month_range[1], format="%Y-%m")
    
    for idx, row in df.iterrows():
        if row["SKU"] in bias_sku_list:
            if start_dt <= row["AM_dt"] <= end_dt:
                df.at[idx, "Prediction_Value"] += bias_shift
    
    df.drop(columns="AM_dt", inplace=True)
    return df


if __name__ == "__main__":
    # 1) Generate baseline data for 10 SKUs, 24 months
    df_pred = generate_baseline_data_ym(num_skus=10, n_months=24, start_ym="2025-01", seed=42)
    
    print("=== df_pred (Before Anomalies) shape ===")
    print(df_pred.shape)  # Should be non-empty
    
    # 2) Inject anomalies/bias
    df_pred_aug = inject_anomalies_and_bias(
        df_pred,
        data_anomaly_prob=0.08,        # 8% data anomalies
        data_anomaly_factor=5.0,
        residual_anomaly_prob=0.10,    # 10% big forecast errors
        residual_anomaly_factor=30.0,
        bias_sku_list=["SKU_1", "SKU_2"],
        bias_month_range=("2025-10", "2026-02"),
        bias_shift=25.0,
        seed=999
    )
    
    print("\n=== df_pred_aug (After Anomalies & Bias) shape ===")
    print(df_pred_aug.shape)
    
    # 3) Show sample for one SKU, e.g. SKU_1
    sku_filter = "SKU_1"
    df_sku1 = df_pred_aug[df_pred_aug["SKU"] == sku_filter].copy()
    
    # Let's see 15 rows of SKU_1
    print(f"\n=== Sample data for {sku_filter} (showing 15 rows) ===")
    print(df_sku1.head(15))
    
    # Optional: If you want integer columns, round them here:
    # df_pred_aug["Actual_Value"] = df_pred_aug["Actual_Value"].round().astype(int)
    # df_pred_aug["Prediction_Value"] = df_pred_aug["Prediction_Value"].round().astype(int)
    # df_pred_aug["Prediction_Actual"] = df_pred_aug["Prediction_Actual"].round().astype(int)
    #
    # If you do that, then print or save to CSV:
    # df_pred_aug.to_csv("sample_data_with_anomalies.csv", index=False)
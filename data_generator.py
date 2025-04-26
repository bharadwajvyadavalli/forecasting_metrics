"""
Forecast Data Generator

This module provides functions to generate synthetic forecast data for testing
and demonstrating the forecast metrics framework.
"""

import numpy as np
import pandas as pd


def generate_date_list(start_ym="2025-01", n_months=24):
    """Create a list of YYYY-MM strings starting from `start_ym` for `n_months`."""
    start = pd.to_datetime(start_ym, format="%Y-%m")
    dr = pd.date_range(start=start, periods=n_months, freq="MS")
    return [d.strftime("%Y-%m") for d in dr]


def generate_forecast_data(
        num_skus=10,
        n_months=24,
        start_ym="2025-01",
        seed=42,
        trend_pct=0.02,
        noise_pct=0.005,
        with_anomalies=True
):
    """
    Generate synthetic forecast data with controlled characteristics.

    Parameters:
        num_skus (int): Number of SKUs to generate
        n_months (int): Number of months to generate
        start_ym (str): Starting year-month (YYYY-MM)
        seed (int): Random seed for reproducibility
        trend_pct (float): Maximum trend percentage
        noise_pct (float): Maximum noise percentage
        with_anomalies (bool): Whether to inject anomalies and bias

    Returns:
        pd.DataFrame: DataFrame with forecast data
    """
    np.random.seed(seed)
    months = generate_date_list(start_ym, n_months)
    rows = []

    # Create baseline data for each SKU
    for sku in [f"SKU_{i + 1}" for i in range(num_skus)]:
        # Generate base value
        base = np.random.uniform(40, 60)

        # Generate time series with trend
        vals = [base]
        for _ in range(1, n_months):
            vals.append(vals[-1] * (1 + np.random.uniform(-trend_pct, trend_pct)))
        actuals = np.array(vals)

        # Generate forecasts for each month
        for i, am in enumerate(months):
            for h in range(1, 13):  # 12-month forecast horizon
                if i + h >= n_months:
                    break
                # Get the target (future) month for forecasting
                pred_month_idx = i + h
                pred_month = months[pred_month_idx]

                # Generate a forecast for the future month
                # Base the forecast on the current month's actual value, not the future month's value
                forecast_value = actuals[i] * (1 + np.random.uniform(-noise_pct, noise_pct))

                rows.append({
                    "SKU": sku,
                    "Actual_Month": am,
                    "Actual_Value": actuals[i],
                    "Prediction_Month": pred_month,
                    "Prediction_Value": forecast_value,
                    # Set the Prediction_Actual to the actual value for the prediction month
                    "Prediction_Actual": actuals[pred_month_idx]
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Inject anomalies and bias if requested
    if with_anomalies:
        df = inject_anomalies_and_bias(df, seed=seed + 1)

    # Round values to integers
    df["Actual_Value"] = df["Actual_Value"].round().astype(int)
    df["Prediction_Value"] = df["Prediction_Value"].round().astype(int)
    df["Prediction_Actual"] = df["Prediction_Actual"].round().astype(int)

    return df


def inject_anomalies_and_bias(df, seed=123):
    """
    Inject anomalies and bias into forecast data.

    Parameters:
        df (pd.DataFrame): Baseline forecast data
        seed (int): Random seed for reproducibility

    Returns:
        pd.DataFrame: DataFrame with anomalies and bias
    """
    np.random.seed(seed)
    df = df.copy()

    # Calculate month index for each SKU
    df["Month_Idx"] = df.groupby("SKU")["Actual_Month"].transform(
        lambda x: pd.Categorical(x, sorted(x.unique())).codes
    )

    # 1. Apply bias patterns by month index
    def apply_bias(month_idx):
        if month_idx < 4:
            return np.random.uniform(-0.01, 0.01)  # Minimal bias
        elif month_idx < 9:
            return np.random.uniform(0.03, 0.05)  # Medium bias
        else:
            return np.random.choice([0.1, -0.1])  # Spike bias

    df["Bias"] = df["Month_Idx"].apply(apply_bias)
    df["Prediction_Value"] *= (1 + df["Bias"])

    # 2. Add data anomalies (2% per month)
    for m, grp in df.groupby("Month_Idx"):
        idxs = grp.sample(frac=0.02, random_state=seed + m).index
        for idx in idxs:
            df.at[idx, "Actual_Value"] *= (1 + np.random.choice([-1, 1]) * 0.2)

    # 3. Add forecast anomalies (2% per month)
    for m, grp in df.groupby("Month_Idx"):
        idxs = grp.sample(frac=0.02, random_state=seed + m + 100).index
        for idx in idxs:
            df.at[idx, "Prediction_Value"] *= (1 + np.random.choice([-1, 1]) * 0.2)

    # Clean up temporary columns
    df.drop(columns=["Month_Idx", "Bias"], inplace=True)

    return df


if __name__ == "__main__":
    # Generate sample data and save to CSV
    df = generate_forecast_data(num_skus=15)
    df.to_csv("sample_data.csv", index=False)
    print("Sample forecast data generated and saved to sample_data.csv")
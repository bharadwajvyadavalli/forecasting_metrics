"""
Enhanced Forecast Data Generator

This module generates synthetic forecast data with SKUs that specifically demonstrate
different forecast quality issues and metrics.
"""

import numpy as np
import pandas as pd


def generate_date_list(start_ym="2025-01", n_months=24):
    """Create a list of YYYY-MM strings."""
    start = pd.to_datetime(start_ym, format="%Y-%m")
    dr = pd.date_range(start=start, periods=n_months, freq="MS")
    return [d.strftime("%Y-%m") for d in dr]


def generate_data(
        start_ym="2025-01",
        n_months=24,
        seed=42
):
    """Generate synthetic forecast data with pairs of SKUs highlighting each metric."""
    np.random.seed(seed)
    months = generate_date_list(start_ym, n_months)
    rows = []

    # Create sets of SKUs that demonstrate specific metric properties
    skus_data = {
        # Bias metric demonstration
        "BIAS_Good": {"base": 50, "trend": 0.01, "noise": 0.01, "bias": 0.0},
        "BIAS_Bad": {"base": 50, "trend": 0.01, "noise": 0.01, "bias": 0.15},  # Consistent over-forecasting

        # Data Anomaly metric demonstration
        "DATA_ANOMALY_Good": {"base": 60, "trend": 0.01, "noise": 0.01, "data_anomaly_rate": 0.0},
        "DATA_ANOMALY_Bad": {"base": 60, "trend": 0.01, "noise": 0.01, "data_anomaly_rate": 0.25},  # High rate of data outliers

        # Residual Anomaly metric demonstration
        "RESIDUAL_ANOMALY_Good": {"base": 65, "trend": 0.01, "noise": 0.01, "residual_anomaly_rate": 0.0},
        "RESIDUAL_ANOMALY_Bad": {"base": 65, "trend": 0.01, "noise": 0.01, "residual_anomaly_rate": 0.25},  # High rate of prediction errors

        # Turning point demonstration with improved parameters
        "TURNING_Good": {"base": 80, "trend": 0.0, "cycle_amplitude": 25, "noise": 0.02, "turning_noise": 0.1},  # More pronounced cycles
        "TURNING_Bad": {"base": 80, "trend": 0.0, "cycle_amplitude": 25, "noise": 0.02, "turning_noise": 0.95, "delay": 2},  # High turning point miss rate + delay

        # Distribution stability demonstration
        "DISTRIBUTION_Good": {"base": 90, "trend": 0.01, "noise": 0.02, "distribution_shift": False},
        "DISTRIBUTION_Bad": {"base": 90, "trend": 0.01, "noise": 0.02, "distribution_shift": True},  # Shifts midway

        # Calibration demonstration
        "CALIBRATION_Good": {"base": 100, "trend": 0.01, "noise": 0.02, "var_consistency": True},
        "CALIBRATION_Bad": {"base": 100, "trend": 0.01, "noise": 0.02, "var_consistency": False}  # Inconsistent error variance
    }

    # Process each SKU
    for sku, params in skus_data.items():
        base = params["base"]
        trend_pct = params.get("trend", 0.01)

        # Generate actuals with appropriate patterns
        actuals = [base]

        # Add trend
        for i in range(1, n_months):
            next_val = actuals[-1] * (1 + np.random.uniform(-trend_pct, trend_pct))

            # Add cycle for turning point SKUs
            if "TURNING" in sku and "cycle_amplitude" in params:
                cycle = params["cycle_amplitude"] * np.sin(i * np.pi / 6)  # 12-month cycle
                next_val += cycle

            # Add distribution shift for distribution SKUs
            if "DISTRIBUTION" in sku and params.get("distribution_shift", False):
                if "Bad" in sku and i > n_months // 2:
                    if i == n_months // 2 + 1:  # Initial step change
                        next_val = next_val * 1.5  # Step change
                    else:
                        # Progressive distribution change after step change
                        # Add increasing variance and occasional outliers
                        variance_factor = (i - n_months // 2) / (n_months // 4)  # Grows over time
                        next_val = next_val * (1 + np.random.normal(0, 0.1 * variance_factor))

                        # Add occasional extreme values (15% chance)
                        if np.random.random() < 0.15:
                            next_val = next_val * np.random.choice([0.7, 1.4])
                # For good SKUs, keep steady and predictable pattern
                elif "Good" in sku:
                    # Small, consistent random variations
                    next_val = next_val * (1 + np.random.normal(0, 0.01))

            actuals.append(next_val)

        actuals = np.array(actuals)

        # Add data anomalies for data anomaly SKUs
        if "DATA_ANOMALY" in sku and params.get("data_anomaly_rate", 0) > 0:
            anomaly_count = int(n_months * params["data_anomaly_rate"])
            anomaly_indices = np.random.choice(range(n_months), anomaly_count, replace=False)
            for idx in anomaly_indices:
                actuals[idx] = actuals[idx] * (1 + np.random.choice([-1, 1]) * 0.5)  # 50% spikes

        # Generate forecasts for each month
        for i, am in enumerate(months):
            for h in range(1, 13):  # 12-month forecast horizon
                if i + h >= n_months:
                    break

                # Get the target (future) month for forecasting
                pred_month_idx = i + h
                pred_month = months[pred_month_idx]

                # Base forecast - start with the actual for current month
                forecast_value = actuals[i]

                # Add basic noise
                noise_pct = params.get("noise", 0.01)
                forecast_value = forecast_value * (1 + np.random.uniform(-noise_pct, noise_pct))

                # Add bias for bias SKUs
                if "BIAS" in sku and "bias" in params:
                    forecast_value = forecast_value * (1 + params["bias"])

                # Add turning point noise for turning point SKUs
                if "TURNING" in sku and "turning_noise" in params:
                    # Check if we're near a turning point
                    is_turning = False
                    if pred_month_idx > 1 and pred_month_idx < n_months - 1:
                        # Check current and adjacent points for turning points
                        for offset in [-1, 0, 1]:
                            check_idx = pred_month_idx + offset
                            if 1 <= check_idx < n_months - 1:
                                prev_diff = actuals[check_idx] - actuals[check_idx-1]
                                next_diff = actuals[check_idx+1] - actuals[check_idx]
                                if np.sign(prev_diff) != np.sign(next_diff):
                                    is_turning = True
                                    break

                    # Apply turning noise and delay for bad SKUs
                    if is_turning and np.random.random() < params["turning_noise"]:
                        # For bad SKUs, completely miss the turning point
                        if "Bad" in sku:
                            if pred_month_idx > 1:
                                # Apply a delay if specified
                                delay = params.get("delay", 0)
                                if pred_month_idx - delay > 0:
                                    # Continue the previous trend direction instead of changing
                                    prev_diff = actuals[pred_month_idx-1] - actuals[pred_month_idx-2]
                                    # Exaggerate the wrong direction
                                    forecast_value = actuals[pred_month_idx-1] + (prev_diff * 1.5)
                                else:
                                    # Fallback if we can't apply delay
                                    forecast_value = actuals[i] * 0.8

                # Add calibration issues for calibration SKUs
                if "CALIBRATION" in sku and not params.get("var_consistency", True):
                    # Error variance grows with horizon
                    if h > 6:
                        forecast_value = forecast_value * (1 + np.random.uniform(-0.2, 0.2))

                # Add residual anomaly for residual anomaly SKUs
                if "RESIDUAL_ANOMALY" in sku and params.get("residual_anomaly_rate", 0) > 0:
                    if np.random.random() < params["residual_anomaly_rate"]:
                        # Add large random error to this prediction (not to the actual)
                        forecast_value = forecast_value * (1 + np.random.choice([-1, 1]) * 0.8)  # 80% error spikes

                rows.append({
                    "SKU": sku,
                    "Actual_Month": am,
                    "Actual_Value": round(actuals[i]),
                    "Prediction_Month": pred_month,
                    "Prediction_Value": round(forecast_value),
                    "Prediction_Actual": round(actuals[pred_month_idx])
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Round values to integers
    df["Actual_Value"] = df["Actual_Value"].round().astype(int)
    df["Prediction_Value"] = df["Prediction_Value"].round().astype(int)
    df["Prediction_Actual"] = df["Prediction_Actual"].round().astype(int)

    return df


if __name__ == "__main__":
    # Generate sample data and save to CSV
    df = generate_data()
    df.to_csv("sample_metric_data.csv", index=False)
    print("Sample forecast data generated and saved to sample_metric_data.csv")
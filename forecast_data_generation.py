import numpy as np
import pandas as pd

def generate_date_list(start_ym="2025-01", n_months=24):
    """
    Creates a list of YYYY-MM strings starting from `start_ym` for `n_months`.
    """
    start = pd.to_datetime(start_ym, format="%Y-%m")
    dr = pd.date_range(start=start, periods=n_months, freq="MS")
    return [d.strftime("%Y-%m") for d in dr]


def generate_baseline_data_ym(
    num_skus=10,
    n_months=24,
    start_ym="2025-01",
    seed=42,
    trend_pct=0.02,
    noise_pct=0.005,          # much smaller noise to boost DirAcc
):
    """
    1) Generates 'Actual' data for each SKU over `n_months`.
    2) Applies ±trend_pct drift.
    3) Creates 12-month-ahead forecasts with ±noise_pct noise.
    """
    np.random.seed(seed)
    months = generate_date_list(start_ym, n_months)
    rows = []
    for sku in [f"SKU_{i+1}" for i in range(num_skus)]:
        base = np.random.uniform(40, 60)
        vals = [base]
        for _ in range(1, n_months):
            vals.append(vals[-1] * (1 + np.random.uniform(-trend_pct, trend_pct)))
        actuals = np.array(vals)
        for i, am in enumerate(months):
            for h in range(1, 13):
                if i + h - 1 >= n_months:
                    break
                rows.append({
                    "SKU": sku,
                    "Actual_Month": am,
                    "Actual_Value": actuals[i],
                    "Prediction_Month": months[i+h-1],
                    "Prediction_Value": actuals[i] * (1 + np.random.uniform(-noise_pct, noise_pct)),
                })
    df = pd.DataFrame(rows)
    df["Prediction_Actual"] = df["Actual_Value"]
    return df


def inject_anomalies_and_bias(
    df,
    minimal_bias_range=(-0.01, 0.01),    # tighter “low” block
    medium_bias_range=(0.03, 0.05),      # moderate block
    spike_bias_values=(0.1, -0.1),       # ±10% spikes only
    data_anomaly_rate=0.02,              # 2% per month
    data_anomaly_mag=0.2,                # ±20% data spikes
    residual_anomaly_rate=0.02,          # 2% per month
    residual_anomaly_mag=0.2,            # ±20% residual spikes
    drift_start=12,
    drift_rate=0.001,                    # mild drift (0.1%)
    seed=123
):
    """
    1) Mild distributional drift after month≥drift_start
    2) Block bias per month index:
       0–3: uniform(minimal_bias_range)
       4–8: uniform(medium_bias_range)
       ≥9: choice(spike_bias_values)
    3) Data anomalies per month (2%)
    4) Residual anomalies per month (2%)
    """
    np.random.seed(seed)
    df = df.copy()
    # compute month index per SKU
    df["Month_Idx"] = df.groupby("SKU")["Actual_Month"] \
                       .transform(lambda x: pd.Categorical(x, sorted(x.unique())).codes)

    # 1) drift on actuals
    df.loc[df["Month_Idx"] >= drift_start, "Actual_Value"] *= (1 + drift_rate)

    # 2) apply block bias
    def pick_bias(m):
        if m < 4:
            return np.random.uniform(*minimal_bias_range)
        elif m < 9:
            return np.random.uniform(*medium_bias_range)
        else:
            return np.random.choice(spike_bias_values)
    df["Bias"] = df["Month_Idx"].apply(pick_bias)
    df["Prediction_Value"] *= (1 + df["Bias"])

    # 3) data anomalies _within each month_
    for m, grp in df.groupby("Month_Idx"):
        idxs = grp.sample(frac=data_anomaly_rate, random_state=seed+m).index
        for idx in idxs:
            df.at[idx, "Actual_Value"] *= (1 + np.random.choice([-1,1]) * data_anomaly_mag)

    # 4) residual anomalies _within each month_
    for m, grp in df.groupby("Month_Idx"):
        idxs = grp.sample(frac=residual_anomaly_rate, random_state=seed+m+100).index
        for idx in idxs:
            df.at[idx, "Prediction_Value"] *= (1 + np.random.choice([-1,1]) * residual_anomaly_mag)

    # finalize
    df["Actual_Value"] = df["Actual_Value"].round().astype(int)
    df["Prediction_Value"] = df["Prediction_Value"].round().astype(int)
    df.drop(columns=["Month_Idx", "Bias"], inplace=True)
    return df


def generate_full_sample(
    num_skus=10,
    n_months=24,
    start_ym="2025-01",
    seed=42
):
    base = generate_baseline_data_ym(
        num_skus=num_skus,
        n_months=n_months,
        start_ym=start_ym,
        seed=seed
    )
    return inject_anomalies_and_bias(base, seed=seed+1)


if __name__ == "__main__":
    df = generate_full_sample()
    df.to_csv("sample_data.csv", index=False)
    print("→ sample_data.csv generated with higher DirAcc, clear bias blocks, and stable 2% anomaly rates.")
from typing import Callable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import norm

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, norm, entropy
from scipy.spatial.distance import jensenshannon

def generate_integer_forecast_data(n_months=24, horizon=12,
                                   low=30, high=50, max_offset=5, seed=42):
    """
    Generate synthetic univariate forecast data:
      - actuals: integers in [low, high], length n_months
      - preds: for each t, horizon-months-ahead forecast = actual[t+h] ± offset
    Returns:
      actuals, preds as 1D numpy arrays of equal length
    """
    np.random.seed(seed)
    series = np.random.randint(low, high+1, size=n_months)
    actuals, preds = [], []
    for i in range(n_months):
        for h in range(1, horizon+1):
            if i + h < n_months:
                actuals.append(series[i + h])
                preds.append(series[i + h] + np.random.randint(-max_offset, max_offset+1))
    return np.array(actuals), np.array(preds)

# 1. Systematic Bias (revised)
def compute_mean_bias(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Mean bias: average(preds - actuals).
    """
    return (preds - actuals).mean()

def compute_tracking_signal(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Tracking Signal: sum of residuals divided by mean absolute residual.
    """
    resid = preds - actuals
    mad = np.mean(np.abs(resid)) or 1.0
    return resid.sum() / mad

def compute_residual_counts(actuals: np.ndarray, preds: np.ndarray) -> int:
    """
    Counts of positive and negative residuals.
    """
    resid = preds - actuals
    pos = int((resid > 0).sum())
    neg = int((resid < 0).sum())
    return pos - neg

def compute_area_under_sparsification(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Area Under Sparsification curve:
    area under the residual-magnitude threshold vs coverage curve.
    """
    resid = np.abs(preds - actuals)
    sorted_resid = np.sort(resid)
    coverage = np.arange(1, len(sorted_resid) + 1) / len(sorted_resid)
    # approximate ∫ coverage(t) dt
    return np.trapz(coverage, sorted_resid)

# 2. Outlier & Anomaly Behavior
def compute_data_anomaly_rate(actuals: np.ndarray, k: float = 3.0) -> float:
    """
    Fraction of actuals where |value - median|/MAD > k.
    """
    med = np.median(actuals)
    mad = np.median(np.abs(actuals - med)) or 1.0
    return np.mean(np.abs(actuals - med) / mad > k)

def compute_residual_anomaly_rate(actuals: np.ndarray, preds: np.ndarray, k: float = 3.0) -> float:
    """
    Fraction of residuals where |(pred-actual - mean)|/std > k.
    """
    resid = preds - actuals
    return np.mean(np.abs(resid - resid.mean()) / (resid.std() or 1.0) > k)

def compute_mean_anomaly_magnitude(actuals: np.ndarray, preds: np.ndarray, k: float = 3.0) -> float:
    """
    Average magnitude of flagged residual anomalies.
    """
    resid = preds - actuals
    z = np.abs(resid - resid.mean()) / (resid.std() or 1.0)
    return np.mean(np.abs(resid)[z > k])

def compute_time_to_detect(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Average index at which the first non-zero residual occurs.
    """
    flags = (preds - actuals) != 0
    idx = np.where(flags)[0]
    return idx.mean() if idx.size else np.nan

def compute_persistence(actuals: np.ndarray, preds: np.ndarray) -> int:
    """
    Longest run of consecutive non-zero residuals.
    """
    flags = (preds - actuals) != 0
    run = max_run = 0
    for f in flags:
        run = run + 1 if f else 0
        max_run = max(max_run, run)
    return max_run

# 3. Direction & Velocity Dynamics
def compute_direction_accuracy(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Fraction of correct up/down movements.
    """
    return np.mean(
        np.sign(np.diff(actuals, prepend=actuals[0])) ==
        np.sign(np.diff(preds, prepend=preds[0]))
    )

def compute_velocity_error(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    MAE of first differences.
    """
    return np.mean(np.abs(np.diff(preds) - np.diff(actuals)))

def compute_acceleration_error(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    MAE of second differences.
    """
    return np.mean(np.abs(np.diff(preds, n=2) - np.diff(actuals, n=2)))

def compute_turning_point_f1(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    F1 score on local peaks/valleys.
    """
    da = np.sign(np.diff(actuals)); dp = np.sign(np.diff(preds))
    tp_a = np.concatenate(([False], da[1:] != da[:-1], [False]))
    tp_p = np.concatenate(([False], dp[1:] != dp[:-1], [False]))
    true = set(np.where(tp_a)[0]); pred = set(np.where(tp_p)[0])
    inter = true & pred
    prec = len(inter)/len(pred) if pred else np.nan
    rec = len(inter)/len(true) if true else np.nan
    return 2 * prec * rec / (prec + rec) if prec and rec else np.nan

def compute_trend_change_delay(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Avg lag between true vs predicted trend reversals.
    """
    da = np.sign(np.diff(actuals)); dp = np.sign(np.diff(preds))
    tp_a = np.where(np.concatenate(([0], da[1:] != da[:-1], [0])))[0]
    tp_p = np.where(np.concatenate(([0], dp[1:] != dp[:-1], [0])))[0]
    delays = [next((t - ta for t in tp_p if t >= ta), np.nan) for ta in tp_a]
    return np.nanmean(delays) if delays else np.nan

# 4. Probabilistic Calibration & Sharpness
def compute_picp(actuals: np.ndarray, preds: np.ndarray, z: float = 1.96) -> float:
    """
    Coverage percent within ±z·σ residual interval.
    """
    resid = preds - actuals
    half = z * resid.std()
    return np.mean((actuals >= preds-half) & (actuals <= preds+half)) * 100

def compute_interval_width(actuals: np.ndarray, preds: np.ndarray, z: float = 1.96) -> float:
    """
    Avg width of ±z·σ intervals.
    """
    return 2 * z * (preds - actuals).std()

def compute_interval_score(actuals: np.ndarray, preds: np.ndarray, alpha: float = 0.1) -> float:
    """
    Winkler‐style interval score.
    """
    resid = preds - actuals; std = resid.std()
    half = norm.ppf(1-alpha/2) * std
    lower, upper = preds-half, preds+half
    miss_low = np.where(actuals < lower, lower - actuals, 0)
    miss_high = np.where(actuals > upper, actuals - upper, 0)
    return np.mean((upper-lower) + (2/alpha)*miss_low + (2/alpha)*miss_high)

def compute_winkler_score(actuals: np.ndarray, preds: np.ndarray, alpha: float = 0.1) -> float:
    return compute_interval_score(actuals, preds, alpha)

def compute_crps(actuals: np.ndarray, preds: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score.
    """
    std = (preds - actuals).std()
    z = (actuals - preds) / std
    return np.mean(std * (z * (2*norm.cdf(z)-1) + 2*norm.pdf(z) - 1/np.sqrt(np.pi)))

# 5. Distributional Drift & Stability
def compute_sliding_jsd(actuals: np.ndarray, preds: np.ndarray, w: int = 6) -> float:
    """
    Mean Jensen–Shannon Distance over sliding windows.
    """
    jsd_vals = []
    for i in range(len(actuals)-w+1):
        ha, _ = np.histogram(actuals[i:i+w], bins=10, density=True)
        hp, _ = np.histogram(preds[i:i+w],    bins=10, density=True)
        jsd_vals.append(jensenshannon(ha, hp))
    return np.mean(jsd_vals)

def compute_psi(actuals: np.ndarray, preds: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    """
    ha, edges = np.histogram(actuals, bins=bins, density=True)
    hp, _     = np.histogram(preds,    bins=edges, density=True)
    ha += 1e-6; hp += 1e-6
    return np.sum((ha-hp) * np.log(ha/hp))

def compute_kl_divergence(actuals: np.ndarray, preds: np.ndarray, bins: int = 10) -> float:
    """
    KL divergence between distributions.
    """
    ha, edges = np.histogram(actuals, bins=bins, density=True)
    hp, _     = np.histogram(preds,    bins=edges, density=True)
    ha += 1e-6; hp += 1e-6
    return np.sum(ha * np.log(ha/hp))

def compute_rolling_error_variance(actuals: np.ndarray, preds: np.ndarray, w: int = 6) -> float:
    return pd.Series(preds-actuals).rolling(w).var().dropna().mean()

def compute_mmd(actuals: np.ndarray, preds: np.ndarray, sigma: float = 1.0) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    """
    X, Y = actuals, preds
    xx = np.exp(-np.subtract.outer(X, X)**2/(2*sigma**2)).mean()
    yy = np.exp(-np.subtract.outer(Y, Y)**2/(2*sigma**2)).mean()
    xy = np.exp(-np.subtract.outer(X, Y)**2/(2*sigma**2)).mean()
    return xx + yy - 2*xy

# 6. Model Robustness & Change‑Sensitivity (placeholders)
def compute_feature_drift_rate(actuals: np.ndarray, preds: np.ndarray) -> float:
    return compute_psi(actuals, preds)

def compute_parameter_sensitivity_index(*args, **kwargs) -> float:
    raise NotImplementedError

def compute_retraining_gain(*args, **kwargs) -> float:
    raise NotImplementedError

def compute_forecast_variability_index(*args, **kwargs) -> float:
    raise NotImplementedError

def compute_out_of_sample_gap(*args, **kwargs) -> float:
    raise NotImplementedError


if __name__ == '__main__':
    # 1. Load the sample data
    df = pd.read_csv('sample_data.csv', parse_dates=['Actual_Month', 'Prediction_Month'])

    # 2. Compute metrics per SKU & Actual_Month
    results = []
    for (sku, actual_month), grp in df.groupby(['SKU', 'Actual_Month']):
        y_true = grp['Prediction_Actual'].to_numpy()
        y_pred = grp['Prediction_Value'].to_numpy()
        base_actuals = grp['Actual_Value'].to_numpy()

        metrics = {
            'SKU': sku,
            'Actual_Month': actual_month,
            # 1. Systematic Bias (updated)
            'Mean_Bias': compute_mean_bias(y_true, y_pred),
            'Tracking_Signal': compute_tracking_signal(y_true, y_pred),
            'Residual Counts': compute_residual_counts(y_true, y_pred),
            'Area_Under_Sparsification': compute_area_under_sparsification(y_true, y_pred),
            # 2. Outlier & Anomaly
            'Data_Anomaly_Rate': compute_data_anomaly_rate(base_actuals),
            'Residual_Anomaly_Rate': compute_residual_anomaly_rate(y_true, y_pred),
            'Mean_Anomaly_Magnitude': compute_mean_anomaly_magnitude(y_true, y_pred),
            'Time_to_Detect': compute_time_to_detect(y_true, y_pred),
            'Persistence': compute_persistence(y_true, y_pred),
            # 3. Direction & Velocity
            'Direction_Accuracy': compute_direction_accuracy(y_true, y_pred),
            'Velocity_Error': compute_velocity_error(y_true, y_pred),
            'Acceleration_Error': compute_acceleration_error(y_true, y_pred),
            'Turning_Point_F1': compute_turning_point_f1(y_true, y_pred),
            'Trend_Change_Delay': compute_trend_change_delay(y_true, y_pred),
            # 4. Probabilistic Calibration
            'PICP (%)': compute_picp(y_true, y_pred),
            'Interval_Width': compute_interval_width(y_true, y_pred),
            'Interval_Score': compute_interval_score(y_true, y_pred),
            'Winkler_Score': compute_winkler_score(y_true, y_pred),
            'CRPS': compute_crps(y_true, y_pred),
            # 5. Distributional Drift & Stability
            'Sliding_JSD': compute_sliding_jsd(y_true, y_pred),
            'PSI': compute_psi(y_true, y_pred),
            'KL_Divergence': compute_kl_divergence(y_true, y_pred),
            'Rolling_Error_Var': compute_rolling_error_variance(y_true, y_pred),
            'MMD': compute_mmd(y_true, y_pred),
            # 6. Model Robustness & Change‑Sensitivity
            'Feature_Drift_Rate': compute_feature_drift_rate(y_true, y_pred)
        }
        results.append(metrics)

    # 3. Build DataFrame and round numeric columns to 2 decimals
    output_df = pd.DataFrame(results)
    numeric_cols = output_df.select_dtypes(include=[np.number]).columns
    output_df[numeric_cols] = output_df[numeric_cols].round(2)

    # 4. Save to CSV
    output_df.to_csv('sample_data_outputs.csv', index=False)

    print("Metrics computed and written to sample_data_outputs.csv")
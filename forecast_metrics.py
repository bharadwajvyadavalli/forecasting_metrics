"""
Simplified Forecast Metrics Library

This module contains forecast metrics organized by business purpose.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon


# BIAS METRICS
def mean_bias(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate mean bias (average prediction error)."""
    return np.mean(predictions - actuals)


# ANOMALY METRICS
def data_anomaly_rate(actuals: np.ndarray, k: float = 3.0) -> float:
    """Calculate fraction of actuals where |value - median|/MAD > k."""
    med = np.median(actuals)
    mad = np.median(np.abs(actuals - med)) or 1.0
    return np.mean(np.abs(actuals - med) / mad > k)


def residual_anomaly_rate(actuals: np.ndarray, predictions: np.ndarray, k: float = 3.0) -> float:
    """Calculate fraction of residuals where |(pred-actual - mean)|/std > k."""
    resid = predictions - actuals
    return np.mean(np.abs(resid - resid.mean()) / (resid.std() or 1.0) > k)


# DIRECTIONAL METRICS
def direction_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate percentage of correctly predicted up/down movements."""
    actual_dir = np.sign(np.diff(actuals, prepend=actuals[0]))
    pred_dir = np.sign(np.diff(predictions, prepend=predictions[0]))
    return np.mean(actual_dir == pred_dir)


def turning_point_f1(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate F1 score for turning point detection."""
    # Calculate directions
    da = np.sign(np.diff(actuals))
    dp = np.sign(np.diff(predictions))

    # Identify turning points (where direction changes)
    tp_a = np.concatenate(([False], da[1:] != da[:-1], [False]))
    tp_p = np.concatenate(([False], dp[1:] != dp[:-1], [False]))

    # Calculate precision and recall
    true = set(np.where(tp_a)[0])
    pred = set(np.where(tp_p)[0])

    # Handle edge cases
    if not true and not pred:
        return 1.0  # Perfect agreement - no turning points
    if not true or not pred:
        return 0.0  # One has turning points, other doesn't

    # Calculate intersection
    inter = true & pred

    # Calculate precision and recall
    precision = len(inter) / len(pred)
    recall = len(inter) / len(true)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# DISTRIBUTION METRICS
def sliding_jsd(actuals: np.ndarray, window_size: int = 4, bins: int = 6) -> float:
    """Calculate average Jensen-Shannon divergence between sequential time windows."""
    # Check if we have enough data
    if len(actuals) < 2 * window_size:
        return 0.0

    jsd_scores = []

    # Calculate JSD for each pair of sequential windows
    for i in range(len(actuals) - 2 * window_size + 1):
        # Extract consecutive windows
        window1 = actuals[i:i + window_size]
        window2 = actuals[i + window_size:i + 2 * window_size]

        # Create probability distributions using histograms
        p, _ = np.histogram(window1, bins=bins, density=True)
        q, _ = np.histogram(window2, bins=bins, density=True)

        # Normalize to ensure valid probability distributions
        p = p / (p.sum() + 1e-12)
        q = q / (q.sum() + 1e-12)

        # Calculate Jensen-Shannon divergence
        jsd = jensenshannon(p, q, base=2)
        jsd_scores.append(jsd)

    # Return average JSD
    return float(np.mean(jsd_scores)) if jsd_scores else 0.0


# CALIBRATION METRICS
def crps(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """Calculate Continuous Ranked Probability Score."""
    # Estimate standard deviation of forecast error
    std = (predictions - actuals).std()

    # If standard deviation is zero, return mean absolute error
    if std == 0:
        return float(np.mean(np.abs(predictions - actuals)))

    # Calculate standardized error
    z = (actuals - predictions) / std

    # CRPS formula for normal distribution
    return std * (z * (2 * norm.cdf(z) - 1)
               + 2 * norm.pdf(z)
               - 1 / np.sqrt(np.pi))


# Define metrics groups for easy reference
BIAS_METRICS = ['mean_bias']
ANOMALY_METRICS = ['data_anomaly_rate', 'residual_anomaly_rate']
DIRECTIONAL_METRICS = ['direction_accuracy', 'turning_point_f1']
DISTRIBUTION_METRICS = ['sliding_jsd']
CALIBRATION_METRICS = ['crps']

# Higher is better vs. lower is better
PERFORMANCE_METRICS = ['direction_accuracy', 'turning_point_f1']  # Higher is better
ERROR_METRICS = BIAS_METRICS + ANOMALY_METRICS + DISTRIBUTION_METRICS + CALIBRATION_METRICS  # Lower is better

# Metrics where absolute value matters for thresholds
SYMMETRIC_ERROR_METRICS = ['mean_bias']


def calculate_all_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """Calculate all forecast metrics for the given actuals and predictions."""
    return {
        # Bias metrics
        'mean_bias': mean_bias(actuals, predictions),

        # Anomaly metrics
        'data_anomaly_rate': data_anomaly_rate(actuals),
        'residual_anomaly_rate': residual_anomaly_rate(actuals, predictions),

        # Directional metrics
        'direction_accuracy': direction_accuracy(actuals, predictions),
        'turning_point_f1': turning_point_f1(actuals, predictions),

        # Distribution metrics
        'sliding_jsd': sliding_jsd(actuals),

        # Calibration metrics
        'crps': crps(actuals, predictions)
    }
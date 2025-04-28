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
def turning_point_f1(actuals: np.ndarray, predictions: np.ndarray, tolerance: int = 1) -> float:
    """
    Calculate F1 score for turning point detection with improved algorithm.

    This metric identifies how well a forecast predicts changes in trend direction
    (turning points). It's particularly valuable for seasonal forecasting as it captures
    the model's ability to predict when upward trends become downward and vice versa.

    Parameters:
    -----------
    actuals : np.ndarray
        Array of actual values
    predictions : np.ndarray
        Array of predicted values
    tolerance : int, default=1
        Number of time steps tolerance for considering a turning point correctly predicted

    Returns:
    --------
    float:
        F1 score for turning point detection (0.0 to 1.0)
        - 1.0 means all turning points correctly identified with no false positives
        - 0.0 means no turning points correctly identified or no turning points exist
    """
    # Must have at least 3 points to detect turning points
    if len(actuals) < 3 or len(predictions) < 3:
        return 0.0

    # Calculate first differences (directions)
    actual_diff = np.diff(actuals)
    pred_diff = np.diff(predictions)

    # Identify sign changes (turning points)
    # A turning point occurs where the sign changes from positive to negative or vice versa
    actual_tp = np.where(np.sign(actual_diff[:-1]) != np.sign(actual_diff[1:]))[0] + 1
    pred_tp = np.where(np.sign(pred_diff[:-1]) != np.sign(pred_diff[1:]))[0] + 1

    # Handle edge cases - if there are no turning points
    if len(actual_tp) == 0 and len(pred_tp) == 0:
        return 1.0  # Perfect agreement - no turning points
    if len(actual_tp) == 0 or len(pred_tp) == 0:
        return 0.0  # One has turning points, other doesn't

    # Count true positives with tolerance
    true_positives = 0
    used_pred_tp = set()

    for tp in actual_tp:
        # Check if any predicted turning point is within tolerance
        for i, p_tp in enumerate(pred_tp):
            if i in used_pred_tp:
                continue  # Skip already matched predictions

            if abs(tp - p_tp) <= tolerance:
                true_positives += 1
                used_pred_tp.add(i)
                break

    # Calculate precision and recall
    precision = true_positives / len(pred_tp)
    recall = true_positives / len(actual_tp)

    # Calculate F1 score
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# DISTRIBUTION METRICS
def distribution_jsd(actuals: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Jensen-Shannon divergence between first half and second half
    of the time series to detect distribution shifts.

    Returns a value between 0 (identical distributions) and 1 (completely different).
    """
    # Check if we have enough data
    if len(actuals) < 6:  # Need at least a few points in each half
        return 0.0

    # If all values are identical, distribution is perfectly stable
    if np.all(actuals == actuals[0]):
        return 0.0

    # Split the time series into first half and second half
    half_point = len(actuals) // 2
    first_half = actuals[:half_point]
    second_half = actuals[half_point:]

    # Create adaptive bin edges covering both distributions
    data_min = min(np.min(first_half), np.min(second_half)) * 0.95
    data_max = max(np.max(first_half), np.max(second_half)) * 1.05

    # Ensure range isn't zero
    if data_max == data_min:
        data_max = data_min + 1e-8

    # Calculate histograms with identical bin edges
    bin_edges = np.linspace(data_min, data_max, bins + 1)
    p, _ = np.histogram(first_half, bins=bin_edges, density=True)
    q, _ = np.histogram(second_half, bins=bin_edges, density=True)

    # Add small constant to avoid zeros
    p = p + 1e-10
    q = q + 1e-10

    # Normalize
    p = p / p.sum()
    q = q / q.sum()

    # Calculate JSD with base 2 (results in 0-1 range)
    jsd = jensenshannon(p, q, base=2)

    return float(0.0 if np.isnan(jsd) else jsd)


# CALIBRATION METRICS
def crps(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate Continuous Ranked Probability Score for point forecasts.

    Parameters:
    actuals: np.ndarray - Array of observed values
    predictions: np.ndarray - 1D array of mean predictions

    Returns:
    float: Mean CRPS (single scalar value)
    """
    # Estimate standard deviation of forecast error
    std = max((predictions - actuals).std(), 1e-8)  # Avoid division by zero

    # Calculate standardized error
    z = (actuals - predictions) / std

    # CRPS formula for normal distribution
    crps_values = std * (z * (2 * norm.cdf(z) - 1) +
                         2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    # Return mean CRPS as a single scalar value
    return float(np.mean(crps_values))


# Define metrics groups for easy reference
BIAS_METRICS = ['mean_bias']
ANOMALY_METRICS = ['data_anomaly_rate', 'residual_anomaly_rate']
DIRECTIONAL_METRICS = ['turning_point_f1']
DISTRIBUTION_METRICS = ['sliding_jsd']
CALIBRATION_METRICS = ['crps']

# Higher is better vs. lower is better
PERFORMANCE_METRICS = ['turning_point_f1']  # Higher is better
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
        'turning_point_f1': turning_point_f1(actuals, predictions),

        # Distribution metrics
        'sliding_jsd': distribution_jsd(actuals),

        # Calibration metrics
        'crps': crps(actuals, predictions)
    }
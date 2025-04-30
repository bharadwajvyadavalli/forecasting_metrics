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
    #return np.mean(np.abs(resid - resid.mean()) / (resid.std() or 1.0) > k)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) or 1.0
    return np.mean(np.abs(resid - med) / mad > k)


# DIRECTIONAL METRICS
def turning_point_f1(actuals: np.ndarray,
                     predictions: np.ndarray,
                     tolerance: int = 1,
                     magnitude_threshold: float = 0.15,
                     smoothing: bool = True,
                     smoothing_window: int = 3) -> float:
    """
    Calculate F1 score for turning point detection optimized for monthly forecasts.

    Parameters:
    -----------
    actuals : np.ndarray
        Array of actual monthly values
    predictions : np.ndarray
        Array of predicted monthly values
    tolerance : int, default=1
        Number of months tolerance for considering a turning point correctly predicted
    magnitude_threshold : float, default=0.15
        Minimum relative magnitude to consider as a significant trend change
    smoothing : bool, default=True
        Whether to apply light smoothing to reduce noise in monthly data
    smoothing_window : int, default=3
        Window size for smoothing (3 = quarter-centered moving average)

    Returns:
    --------
    float:
        F1 score (0.0 to 1.0)

    Thresholds
    ---------------------------------
        F1 Score Interpretation
    ---------------------------------
    0.80 - 1.00: Excellent forecast (accurately captures trend changes)
    0.60 - 0.80: Good forecast
    0.40 - 0.60: Average forecast
    0.20 - 0.40: Poor forecast
    0.00 - 0.20: Very poor forecast
    """
    # Need at least 3 points to detect turning points
    if len(actuals) < 3 or len(predictions) < 3:
        return 0.0

    # Apply smoothing if enabled
    if smoothing and len(actuals) >= smoothing_window:
        weights = np.ones(smoothing_window) / smoothing_window
        actuals_smooth = np.convolve(actuals, weights, mode='valid')
        predictions_smooth = np.convolve(predictions, weights, mode='valid')
        offset = smoothing_window // 2
    else:
        actuals_smooth = actuals
        predictions_smooth = predictions
        offset = 0

    # Calculate month-over-month changes
    actual_diff = np.diff(actuals_smooth)
    pred_diff = np.diff(predictions_smooth)

    # Calculate average magnitude of changes
    actual_magnitude = np.abs(actual_diff)
    pred_magnitude = np.abs(pred_diff)

    actual_mean_magnitude = np.mean(actual_magnitude) or 1.0  # Avoid division by zero
    pred_mean_magnitude = np.mean(pred_magnitude) or 1.0  # Avoid division by zero

    # Determine significant changes
    actual_significant = actual_magnitude > (magnitude_threshold * actual_mean_magnitude)
    pred_significant = pred_magnitude > (magnitude_threshold * pred_mean_magnitude)

    # Find direction changes - Note these will be one element shorter than actual_diff
    if len(actual_diff) >= 2:
        actual_sign_changes = np.sign(actual_diff[:-1]) != np.sign(actual_diff[1:])

        # Create correctly shaped significant arrays for masking
        # Use actual_significant[:-1] and actual_significant[1:] to match sign_changes length
        if len(actual_significant) >= 2:
            actual_significant_prev = actual_significant[:-1]
            actual_significant_next = actual_significant[1:]

            # A turning point must have direction change AND significant magnitude
            actual_tp_mask = actual_sign_changes & (actual_significant_prev | actual_significant_next)
            actual_tp = np.where(actual_tp_mask)[0] + 1 + offset  # +1 because sign change is between points
        else:
            actual_tp = np.array([])
    else:
        actual_tp = np.array([])

    # Same logic for predictions
    if len(pred_diff) >= 2:
        pred_sign_changes = np.sign(pred_diff[:-1]) != np.sign(pred_diff[1:])

        if len(pred_significant) >= 2:
            pred_significant_prev = pred_significant[:-1]
            pred_significant_next = pred_significant[1:]

            pred_tp_mask = pred_sign_changes & (pred_significant_prev | pred_significant_next)
            pred_tp = np.where(pred_tp_mask)[0] + 1 + offset
        else:
            pred_tp = np.array([])
    else:
        pred_tp = np.array([])

    # Handle edge cases
    if len(actual_tp) == 0 and len(pred_tp) == 0:
        return 1.0  # Perfect agreement - no turning points

    if len(actual_tp) == 0 or len(pred_tp) == 0:
        return 0.0  # One has turning points, other doesn't

    # Match turning points with tolerance
    true_positives = 0
    used_pred_tp = set()

    for tp in actual_tp:
        for i, p_tp in enumerate(pred_tp):
            if i in used_pred_tp:
                continue  # Skip already matched predictions

            if abs(tp - p_tp) <= tolerance:
                true_positives += 1
                used_pred_tp.add(i)
                break

    # Calculate F1 score
    precision = true_positives / len(pred_tp)
    recall = true_positives / len(actual_tp)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# DISTRIBUTION METRICS
def distribution_simple(actuals: np.ndarray) -> float:
    """
    Calculate distribution shift between first and second half of time series.
    Returns a value between 0 and 1 where higher values indicate greater drift.
    """
    # Check if we have enough data
    if len(actuals) < 6:
        return 0.0

    # If all values are identical, distribution is perfectly stable
    if np.all(actuals == actuals[0]):
        return 0.0

    # Split the time series into first half and second half
    half_point = len(actuals) // 2
    first_half = actuals[:half_point]
    second_half = actuals[half_point:]

    # Calculate absolute change in mean (normalized)
    # This detects shifts in the central tendency of the data
    mean_first = np.mean(first_half)
    mean_second = np.mean(second_half)
    mean_max = max(abs(mean_first), abs(mean_second), 1e-8)
    mean_change = abs(mean_second - mean_first) / mean_max

    # Calculate absolute change in variance (normalized)
    # This detects changes in the spread or dispersion of the data
    var_first = np.var(first_half)
    var_second = np.var(second_half)
    var_max = max(var_first, var_second, 1e-8)
    var_change = abs(var_second - var_first) / var_max

    # Calculate ratio of means to detect step changes
    # This is particularly effective at catching sudden level shifts
    if mean_first > 0 and mean_second > 0:
        mean_ratio = max(mean_second / mean_first, mean_first / mean_second)
        # Higher ratio indicates larger step change
        step_change = min((mean_ratio - 1) / 2, 1.0)  # Cap at 1.0
    else:
        step_change = 0.0

    # Combined score with weights for different types of drift
    # 50% emphasis on step changes (sudden shifts)
    # 30% emphasis on mean changes (gradual level shifts)
    # 20% emphasis on variance changes (changes in volatility)
    score = 0.5 * step_change + 0.3 * mean_change + 0.2 * var_change

    return min(score, 1.0)

# CALIBRATION METRICS
def crps(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate scale-independent Continuous Ranked Probability Score for point forecasts.

    This implementation centers errors to remove bias impact and normalizes
    by the scale of the data to make the metric comparable across different datasets.

    Parameters:
    -----------
    actuals: np.ndarray
        Array of observed values
    predictions: np.ndarray
        1D array of mean predictions

    Returns:
    --------
    float:
        Scale-independent CRPS (single scalar value)

    Thresholds
    ---------------
    Excellent calibration: CRPS ≤ 0.05
    Good calibration: 0.05 < CRPS ≤ 0.10
    Acceptable calibration: 0.10 < CRPS ≤ 0.20
    Poor calibration: 0.20 < CRPS ≤ 0.35
    Very poor calibration: CRPS > 0.35
    """
    # Calculate errors
    errors = predictions - actuals

    # Get scale factor for normalization (using mean of absolute actuals)
    scale_factor = np.mean(np.abs(actuals)) or 1.0  # Avoid division by zero

    # Estimate standard deviation of forecast error (robust to outliers)
    median_error = np.median(errors)
    mad = np.median(np.abs(errors - median_error))
    # Convert MAD to approximate standard deviation
    std = mad * 1.4826  # Constant for normal distribution
    std = max(std, 1e-8)  # Avoid division by zero

    # Center the errors to remove bias impact
    centered_errors = errors - np.mean(errors)

    # Calculate standardized centered error
    z = centered_errors / std

    # CRPS formula for normal distribution (centered)
    crps_values = std * (z * (2 * norm.cdf(z) - 1) +
                         2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

    # Scale-independent CRPS
    scale_independent_crps = np.mean(np.abs(crps_values)) / scale_factor

    return float(scale_independent_crps)


# Define metrics groups for easy reference
BIAS_METRICS = ['mean_bias']
ANOMALY_METRICS = ['data_anomaly_rate', 'residual_anomaly_rate']
DIRECTIONAL_METRICS = ['turning_point_f1']
DISTRIBUTION_METRICS = ['data_drift']
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
        'data_drift': distribution_simple(actuals),

        # Calibration metrics
        'crps': crps(actuals, predictions)
    }
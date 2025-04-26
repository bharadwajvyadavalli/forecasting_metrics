"""
Forecast Metrics Library

This module contains all forecast metrics organized by business purpose.
Each metric includes documentation about its business relevance and interpretation.

Categories:
1. Bias Metrics - Detect systematic over/under-prediction
2. Anomaly Metrics - Identify outliers and unusual patterns
3. Directional Metrics - Evaluate trend and directional accuracy
4. Distribution Metrics - Assess distributional shifts and stability
5. Calibration Metrics - Evaluate probabilistic accuracy
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.spatial.distance import jensenshannon


# ============================================================================
# BIAS METRICS
# ============================================================================

def mean_bias(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate mean bias (average prediction error).

    Business Impact:
    - In inventory forecasting: Positive bias leads to excess inventory costs,
      negative bias leads to stockouts and lost sales
    - In resource planning: Bias affects staffing levels and capacity decisions

    Returns:
        float: Mean bias value
    """
    return np.mean(predictions - actuals)


def tracking_signal(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate tracking signal (sum of errors divided by mean absolute error).

    Values between -4 and +4 indicate acceptable balance.
    Values outside this range indicate significant bias.

    Business Impact:
    - In supply chain: Indicates if forecast is systematically missing demand signals
    - In financial forecasting: Helps identify persistent budget variances

    Returns:
        float: Tracking signal value
    """
    resid = predictions - actuals
    mad = np.mean(np.abs(resid)) or 1.0  # Avoid division by zero
    return resid.sum() / mad


def residual_counts(actuals: np.ndarray, predictions: np.ndarray) -> int:
    """
    Count imbalance between positive and negative residuals.

    Business Impact:
    - In retail forecasting: Shows if a model consistently over/under-predicts
      during specific periods (e.g., holidays, promotions)

    Returns:
        int: Difference between count of positive and negative residuals
    """
    resid = predictions - actuals
    pos = int((resid > 0).sum())
    neg = int((resid < 0).sum())
    return pos - neg


# ============================================================================
# ANOMALY METRICS
# ============================================================================

def data_anomaly_rate(actuals: np.ndarray, k: float = 3.0) -> float:
    """
    Calculate fraction of actuals where |value - median|/MAD > k.

    Business Impact:
    - In retail: Identifies unusual demand patterns requiring investigation
    - In manufacturing: Flags potential data quality issues

    Returns:
        float: Fraction of actuals identified as anomalies (0.0 to 1.0)
    """
    med = np.median(actuals)
    mad = np.median(np.abs(actuals - med)) or 1.0
    return np.mean(np.abs(actuals - med) / mad > k)


def residual_anomaly_rate(actuals: np.ndarray, predictions: np.ndarray, k: float = 3.0) -> float:
    """
    Calculate fraction of residuals where |(pred-actual - mean)|/std > k.

    Business Impact:
    - In capacity planning: Flags periods where forecast reliability is compromised
    - In inventory management: Identifies problematic forecast periods

    Returns:
        float: Fraction of residuals identified as anomalies (0.0 to 1.0)
    """
    resid = predictions - actuals
    return np.mean(np.abs(resid - resid.mean()) / (resid.std() or 1.0) > k)


# ============================================================================
# DIRECTIONAL METRICS
# ============================================================================

def direction_accuracy(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate percentage of correctly predicted up/down movements.

    Business Impact:
    - In financial forecasting: Drives buy/sell decisions
    - In staffing: Guides hiring ahead of demand increases

    Returns:
        float: Direction accuracy (0.0 to 1.0)
    """
    actual_dir = np.sign(np.diff(actuals, prepend=actuals[0]))
    pred_dir = np.sign(np.diff(predictions, prepend=predictions[0]))
    return np.mean(actual_dir == pred_dir)


def turning_point_f1(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate F1 score for turning point detection.

    Business Impact:
    - In market analysis: Critical for timing market entries/exits
    - In inventory management: Drives cycle stock adjustments

    Returns:
        float: F1 score for turning point detection (0.0 to 1.0)
    """
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


# ============================================================================
# DISTRIBUTION METRICS
# ============================================================================

def sliding_jsd(actuals: np.ndarray, window_size: int = 4, bins: int = 6) -> float:
    """
    Calculate average Jensen-Shannon divergence between sequential time windows.

    Business Impact:
    - In model management: Indicates when models need retraining
    - In retail: Detects evolving customer preferences

    Returns:
        float: Average Jensen-Shannon divergence (0.0 to 1.0)
    """
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


# ============================================================================
# CALIBRATION METRICS
# ============================================================================

def crps(actuals: np.ndarray, predictions: np.ndarray) -> float:
    """
    Calculate Continuous Ranked Probability Score.

    Business Impact:
    - In risk management: Evaluates uncertainty quantification
    - In financial forecasting: Evaluates accuracy of risk models

    Returns:
        float: CRPS score (lower is better)
    """
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



# ============================================================================
# METRIC GROUPS
# ============================================================================

# Define metrics groups for easy reference
BIAS_METRICS = ['mean_bias', 'tracking_signal', 'residual_counts']
ANOMALY_METRICS = ['data_anomaly_rate', 'residual_anomaly_rate']
DIRECTIONAL_METRICS = ['direction_accuracy', 'turning_point_f1']
DISTRIBUTION_METRICS = ['sliding_jsd']
CALIBRATION_METRICS = ['crps']

# Higher is better vs. lower is better
PERFORMANCE_METRICS = ['direction_accuracy', 'turning_point_f1']  # Higher is better
ERROR_METRICS = BIAS_METRICS + ANOMALY_METRICS + DISTRIBUTION_METRICS + CALIBRATION_METRICS  # Lower is better

# Metrics where absolute value matters for thresholds
SYMMETRIC_ERROR_METRICS = ['mean_bias', 'tracking_signal']


def calculate_all_metrics(actuals: np.ndarray, predictions: np.ndarray) -> dict:
    """
    Calculate all forecast metrics for the given actuals and predictions.

    Parameters:
        actuals (np.ndarray): Array of actual values
        predictions (np.ndarray): Array of predicted values

    Returns:
        dict: Dictionary of all metrics with their values
    """
    return {
        # Bias metrics
        'mean_bias': mean_bias(actuals, predictions),
        'tracking_signal': tracking_signal(actuals, predictions),
        'residual_counts': residual_counts(actuals, predictions),

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
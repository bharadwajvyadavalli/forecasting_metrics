"""
Simplified Metrics Calculator Module

This module provides a class for calculating and analyzing forecast metrics.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple, Optional

import forecast_metrics as fm


class MetricsCalculator:
    """A class for calculating and analyzing forecast metrics."""

    def __init__(self, data_path: Optional[str] = None, data_df: Optional[pd.DataFrame] = None):
        if data_df is not None:
            self.data = data_df
        elif data_path is not None:
            self.data = pd.read_csv(data_path, parse_dates=['Actual_Month', 'Prediction_Month'])
        else:
            self.data = None

        self.metrics_df: Optional[pd.DataFrame] = None
        self.sku_thresholds: Optional[pd.DataFrame] = None
        # Removed global_thresholds as it's no longer needed

    def compute_metrics(self) -> pd.DataFrame:
        """Compute forecast metrics for all SKUs and time periods."""
        if self.data is None:
            raise ValueError("No data loaded. Load data first.")

        results = []
        for (sku, actual_month), grp in self.data.groupby(['SKU', 'Actual_Month']):
            y_true = grp['Prediction_Actual'].to_numpy()
            y_pred = grp['Prediction_Value'].to_numpy()

            metrics = {'SKU': sku, 'Actual_Month': actual_month}
            metrics.update(fm.calculate_all_metrics(y_true, y_pred))
            results.append(metrics)

        self.metrics_df = pd.DataFrame(results)
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns
        self.metrics_df[numeric_cols] = self.metrics_df[numeric_cols].round(3)
        return self.metrics_df

    def compute_thresholds(self) -> pd.DataFrame:
        """Compute SKU-level performance thresholds for all metrics."""
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")

        # Import the thresholds module here to avoid circular imports
        import thresholds as th
        self.sku_thresholds = th.calculate_thresholds(self.metrics_df)
        return self.sku_thresholds

    def _get_metric_impact(self, metric_name: str, data: Optional[pd.DataFrame] = None) -> str:
        """Get business impact description for a metric."""
        # Import the thresholds module here to avoid circular imports
        import thresholds as th
        return th.get_metric_impact(metric_name, data)

    def generate_report(self, output_dir: str = '.') -> None:
        """Save metrics and thresholds to CSV files."""
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")

        os.makedirs(output_dir, exist_ok=True)
        self.metrics_df.to_csv(f"{output_dir}/metrics_results.csv", index=False)

        if self.sku_thresholds is None:
            self.compute_thresholds()

        self.sku_thresholds.to_csv(f"{output_dir}/sku_thresholds.csv", index=False)
        # Removed saving global_thresholds.csv
        # Removed call to _generate_sku_performance_summary
        print(f"Reports saved to {output_dir}")

    # Removed _generate_sku_performance_summary method since we don't need it anymore
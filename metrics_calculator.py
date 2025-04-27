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
        self.global_thresholds: Optional[pd.DataFrame] = None

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

    def compute_thresholds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute performance thresholds for all metrics."""
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")

        # Import the thresholds module here to avoid circular imports
        import thresholds as th
        self.sku_thresholds, self.global_thresholds = th.calculate_thresholds(self.metrics_df)
        return self.sku_thresholds, self.global_thresholds

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

        if self.sku_thresholds is None or self.global_thresholds is None:
            self.compute_thresholds()

        self.sku_thresholds.to_csv(f"{output_dir}/sku_thresholds.csv", index=False)
        self.global_thresholds.to_csv(f"{output_dir}/global_thresholds.csv", index=False)
        self._generate_sku_performance_summary(output_dir)
        print(f"Reports saved to {output_dir}")

    def _generate_sku_performance_summary(self, output_dir: str) -> None:
        """Generate a summary of SKU performance based on thresholds."""
        sku_performance: Dict[str, Dict[str, any]] = {}
        for sku, sku_df in self.metrics_df.groupby('SKU'):
            thresholds = self.sku_thresholds[self.sku_thresholds['SKU'] == sku]
            red_count = 0
            red_metrics = []
            for _, row in thresholds.iterrows():
                metric = row['Metric']
                # Skip if the metric is not in the DataFrame
                if metric not in sku_df.columns:
                    continue

                cond = row['Red_Condition']
                try:
                    if '<' in cond:
                        val = float(cond.split('< ')[1])
                        if sku_df[metric].mean() < val:
                            red_count += 1
                            red_metrics.append(metric)
                    elif '>' in cond:
                        val = float(cond.split('> ')[1])
                        if sku_df[metric].mean() > val:
                            red_count += 1
                            red_metrics.append(metric)
                except Exception as e:
                    print(f"Error applying condition {cond} for {metric} with SKU {sku}: {e}")
                    continue

            sku_performance[sku] = {'red_count': red_count, 'red_metrics': red_metrics,
                                    'total_metrics': len(thresholds)}

        rows = []
        for sku, perf in sku_performance.items():
            pct = perf['red_count'] / perf['total_metrics'] * 100 if perf['total_metrics'] > 0 else 0
            level = 'Poor' if pct > 30 else 'Moderate' if pct > 10 else 'Good'
            rows.append({
                'SKU': sku,
                'Red_Metrics_Count': perf['red_count'],
                'Red_Metrics_Percent': round(pct, 1),
                'Problematic_Metrics': ', '.join(perf['red_metrics'][:3]) + (
                    '...' if len(perf['red_metrics']) > 3 else ''),
                'Performance_Level': level
            })

        if rows:
            perf_df = pd.DataFrame(rows).sort_values('Red_Metrics_Percent', ascending=False)
            perf_df.to_csv(f"{output_dir}/sku_performance_summary.csv", index=False)
        else:
            # Create an empty DataFrame if there are no rows
            pd.DataFrame(columns=['SKU', 'Red_Metrics_Count', 'Red_Metrics_Percent',
                                  'Problematic_Metrics', 'Performance_Level']).to_csv(
                f"{output_dir}/sku_performance_summary.csv", index=False)
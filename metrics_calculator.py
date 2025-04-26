import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Optional

import forecast_metrics as fm


class MetricsCalculator:
    """
    A class for calculating and analyzing forecast metrics.

    Attributes:
        data (pd.DataFrame): Forecast data
        metrics_df (pd.DataFrame): Calculated metrics results
        sku_thresholds (pd.DataFrame): SKU-level performance thresholds
        global_thresholds (pd.DataFrame): Global performance thresholds
    """

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
        if self.data is None:
            raise ValueError("No data loaded. Load data first.")

        results = []
        for (sku, actual_month), grp in self.data.groupby(['SKU', 'Actual_Month']):
            y_true = grp['Prediction_Actual'].to_numpy()
            y_pred = grp['Prediction_Value'].to_numpy()

            metrics = fm.calculate_all_metrics(y_true, y_pred)
            metrics['SKU'] = sku
            metrics['Actual_Month'] = actual_month
            results.append(metrics)

        self.metrics_df = pd.DataFrame(results)
        numeric_cols = self.metrics_df.select_dtypes(include=[np.number]).columns
        self.metrics_df[numeric_cols] = self.metrics_df[numeric_cols].round(3)
        return self.metrics_df

    def compute_thresholds(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")

        # Identify metric columns (exclude identifiers)
        metrics = [c for c in self.metrics_df.columns if c not in ['SKU', 'Actual_Month']]

        # Split into performance vs. error
        perf_metrics = [m for m in metrics if m.lower() in {p.lower() for p in fm.PERFORMANCE_METRICS}]
        error_metrics = [m for m in metrics if m not in perf_metrics]

        # Lower-case set for symmetric error metrics
        symmetric_list = {s.lower() for s in fm.SYMMETRIC_ERROR_METRICS}

        # Compute global thresholds
        global_rows = []
        for m in error_metrics:
            series = self.metrics_df[m].dropna()
            if series.empty:
                continue
            if m.lower() in symmetric_list:
                series = series.abs()
            # Compute quartiles
            quantiles = series.quantile([0.25, 0.75])
            q25, q75 = quantiles.tolist()
            global_rows.append({
                'Metric': m,
                'Green': round(q25, 3),
                'Yellow': round(q75, 3),
                'Red_Condition': f'> {q75:.3f}',
                'Business_Impact': self._get_metric_impact(m)
            })

        for m in perf_metrics:
            series = self.metrics_df[m].dropna()
            if series.empty:
                continue
            quantiles = series.quantile([0.25, 0.75])
            q25, q75 = quantiles.tolist()
            # For performance metrics, higher is better
            global_rows.append({
                'Metric': m,
                'Green': round(q75, 3),
                'Yellow': round(q25, 3),
                'Red_Condition': f'< {q25:.3f}',
                'Business_Impact': self._get_metric_impact(m)
            })

        self.global_thresholds = pd.DataFrame(global_rows)

        # Compute SKU-level thresholds
        sku_rows = []
        for sku, sku_df in self.metrics_df.groupby('SKU'):
            for m in error_metrics:
                series = sku_df[m].dropna()
                if series.empty:
                    continue
                if m.lower() in symmetric_list:
                    series = series.abs()
                quantiles = series.quantile([0.25, 0.75])
                q25, q75 = quantiles.tolist()
                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': round(q25, 3),
                    'Yellow': round(q75, 3),
                    'Red_Condition': f'> {q75:.3f}',
                    'Business_Impact': self._get_metric_impact(m, sku_df)
                })

            for m in perf_metrics:
                series = sku_df[m].dropna()
                if series.empty:
                    continue
                quantiles = series.quantile([0.25, 0.75])
                q25, q75 = quantiles.tolist()
                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': round(q75, 3),
                    'Yellow': round(q25, 3),
                    'Red_Condition': f'< {q25:.3f}',
                    'Business_Impact': self._get_metric_impact(m, sku_df)
                })

        self.sku_thresholds = pd.DataFrame(sku_rows)
        return self.sku_thresholds, self.global_thresholds

    def _get_metric_impact(self, metric_name: str, data: Optional[pd.DataFrame] = None) -> str:
        impacts = {
            'mean_bias': 'Affects inventory levels and resource allocation efficiency',
            'tracking_signal': 'Indicates persistent forecast misalignment requiring model adjustment',
            'residual_counts': 'Signals systematic over/under-prediction affecting planning reliability',
            'data_anomaly_rate': 'Highlights data quality issues requiring investigation',
            'residual_anomaly_rate': 'Shows forecast sensitivity to outliers affecting reliability',
            'direction_accuracy': 'Critical for trend-based decisions and capacity planning',
            'turning_point_f1': 'Essential for detecting market shifts and strategy adjustments',
            'sliding_jsd': 'Measures distribution stability affecting long-term planning',
            'crps': 'Evaluates probabilistic forecast quality for risk assessment'
        }
        key = next((m for m in impacts if m.lower() == metric_name.lower()), None)
        return impacts.get(key, 'Affects forecast quality and business decisions')

    def generate_report(self, output_dir: str = '.') -> None:
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
        sku_performance: Dict[str, Dict[str, any]] = {}
        for sku, sku_df in self.metrics_df.groupby('SKU'):
            thresholds = self.sku_thresholds[self.sku_thresholds['SKU'] == sku]
            red_count = 0
            red_metrics: List[str] = []
            for _, row in thresholds.iterrows():
                metric = row['Metric']
                cond = row['Red_Condition']
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
            sku_performance[sku] = {'red_count': red_count, 'red_metrics': red_metrics, 'total_metrics': len(thresholds)}
        rows = []
        for sku, perf in sku_performance.items():
            pct = perf['red_count'] / perf['total_metrics'] * 100 if perf['total_metrics'] > 0 else 0
            level = 'Poor' if pct > 30 else 'Moderate' if pct > 10 else 'Good'
            rows.append({
                'SKU': sku,
                'Red_Metrics_Count': perf['red_count'],
                'Red_Metrics_Percent': round(pct, 1),
                'Problematic_Metrics': ', '.join(perf['red_metrics'][:3]) + ('...' if len(perf['red_metrics']) > 3 else ''),
                'Performance_Level': level
            })
        perf_df = pd.DataFrame(rows).sort_values('Red_Metrics_Percent', ascending=False)
        perf_df.to_csv(f"{output_dir}/sku_performance_summary.csv", index=False)

    def visualize_key_metrics(self, output_dir: str = '.') -> None:
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.1)
        self._plot_metric_distribution('mean_bias', output_dir)
        self._plot_metric_by_sku('direction_accuracy', output_dir)
        self._plot_anomaly_comparison(output_dir)
        self._plot_metric_correlations(output_dir)
        print(f"Visualizations saved to {output_dir}")

    def _plot_metric_distribution(self, metric: str, output_dir: str) -> None:
        plt.figure(figsize=(10, 6))
        sns.histplot(self.metrics_df[metric].dropna(), kde=True)
        plt.title(f"Distribution of {metric}")
        plt.xlabel(metric)
        plt.ylabel('Frequency')
        if metric in fm.BIAS_METRICS:
            plt.axvline(0, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_distribution.png")
        plt.close()

    def _plot_metric_by_sku(self, metric: str, output_dir: str) -> None:
        plt.figure(figsize=(12, 6))
        sku_means = self.metrics_df.groupby('SKU')[metric].mean().sort_values()
        sns.barplot(x=sku_means.index, y=sku_means.values)
        plt.title(f"{metric} by SKU")
        plt.ylabel(metric)
        plt.xlabel('SKU')
        plt.xticks(rotation=90)
        plt.axhline(self.metrics_df[metric].mean(), linestyle='--',
                    label=f'Global Avg: {self.metrics_df[metric].mean():.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{metric}_by_sku.png")
        plt.close()

    def _plot_anomaly_comparison(self, output_dir: str) -> None:
        if 'data_anomaly_rate' not in self.metrics_df.columns or 'residual_anomaly_rate' not in self.metrics_df.columns:
            return
        plt.figure(figsize=(10, 6))
        data = self.metrics_df[['SKU', 'data_anomaly_rate', 'residual_anomaly_rate']].melt(
            id_vars=['SKU'], var_name='Anomaly Type', value_name='Rate')
        sns.boxplot(x='Anomaly Type', y='Rate', data=data)
        plt.title('Data vs Residual Anomaly Rates')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/anomaly_comparison.png")
        plt.close()

    def _plot_metric_correlations(self, output_dir: str) -> None:
        plt.figure(figsize=(12, 10))
        metric_cols = [c for c in self.metrics_df.columns if c not in ['SKU', 'Actual_Month'] and not self.metrics_df[c].isna().all()]
        corr = self.metrics_df[metric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
        plt.title('Correlation Between Metrics')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/metric_correlations.png")
        plt.close()

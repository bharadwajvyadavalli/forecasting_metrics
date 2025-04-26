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
            # Skip if the metric is not in the DataFrame
            if m not in self.metrics_df.columns:
                continue

            series = self.metrics_df[m].dropna()
            # Skip if series is empty
            if series.empty:
                continue

            # Apply absolute value for symmetric metrics
            if m.lower() in symmetric_list:
                series = series.abs()

            # Check if series has enough values for quantiles
            if len(series) < 2:
                # Not enough data points for quantiles, use default values
                q25 = 0
                q75 = 0.5
            else:
                try:
                    # Compute quartiles with error handling
                    q25 = float(series.quantile(0.25))
                    q75 = float(series.quantile(0.75))
                except Exception as e:
                    print(f"Error computing quantiles for {m}: {e}")
                    # Fallback to default values
                    q25 = 0
                    q75 = 0.5

            global_rows.append({
                'Metric': m,
                'Green': round(q25, 3),
                'Yellow': round(q75, 3),
                'Red_Condition': f'> {q75:.3f}',
                'Business_Impact': self._get_metric_impact(m)
            })

        for m in perf_metrics:
            # Skip if the metric is not in the DataFrame
            if m not in self.metrics_df.columns:
                continue

            series = self.metrics_df[m].dropna()
            # Skip if series is empty
            if series.empty:
                continue

            # Check if series has enough values for quantiles
            if len(series) < 2:
                # Not enough data points for quantiles, use default values
                q25 = 0.4
                q75 = 0.6
            else:
                try:
                    # Compute quartiles with error handling
                    q25 = float(series.quantile(0.25))
                    q75 = float(series.quantile(0.75))
                except Exception as e:
                    print(f"Error computing quantiles for {m}: {e}")
                    # Fallback to default values
                    q25 = 0.4
                    q75 = 0.6

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
                # Skip if the metric is not in the DataFrame
                if m not in sku_df.columns:
                    continue

                series = sku_df[m].dropna()
                # Skip if series is empty
                if series.empty:
                    continue

                # Apply absolute value for symmetric metrics
                if m.lower() in symmetric_list:
                    series = series.abs()

                # Check if series has enough values for quantiles
                if len(series) < 2:
                    # Not enough data points for quantiles, use default values
                    q25 = 0
                    q75 = 0.5
                else:
                    try:
                        # Compute quartiles with error handling
                        q25 = float(series.quantile(0.25))
                        q75 = float(series.quantile(0.75))
                    except Exception as e:
                        print(f"Error computing quantiles for {m} with SKU {sku}: {e}")
                        # Fallback to default values
                        q25 = 0
                        q75 = 0.5

                sku_rows.append({
                    'SKU': sku,
                    'Metric': m,
                    'Green': round(q25, 3),
                    'Yellow': round(q75, 3),
                    'Red_Condition': f'> {q75:.3f}',
                    'Business_Impact': self._get_metric_impact(m, sku_df)
                })

            for m in perf_metrics:
                # Skip if the metric is not in the DataFrame
                if m not in sku_df.columns:
                    continue

                series = sku_df[m].dropna()
                # Skip if series is empty
                if series.empty:
                    continue

                # Check if series has enough values for quantiles
                if len(series) < 2:
                    # Not enough data points for quantiles, use default values
                    q25 = 0.4
                    q75 = 0.6
                else:
                    try:
                        # Compute quartiles with error handling
                        q25 = float(series.quantile(0.25))
                        q75 = float(series.quantile(0.75))
                    except Exception as e:
                        print(f"Error computing quantiles for {m} with SKU {sku}: {e}")
                        # Fallback to default values
                        q25 = 0.4
                        q75 = 0.6

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

    def visualize_key_metrics(self, output_dir: str = '.') -> None:
        if self.metrics_df is None:
            raise ValueError("No metrics data available. Call compute_metrics() first.")
        os.makedirs(output_dir, exist_ok=True)

        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            # For newer versions of seaborn
            try:
                plt.style.use('seaborn-whitegrid')
            except Exception:
                print("Could not set seaborn style, using default style")

        try:
            sns.set(font_scale=1.1)
        except Exception:
            print("Could not set seaborn font scale")

        self._plot_metric_distribution('mean_bias', output_dir)
        self._plot_metric_by_sku('direction_accuracy', output_dir)
        self._plot_anomaly_comparison(output_dir)
        self._plot_metric_correlations(output_dir)
        print(f"Visualizations saved to {output_dir}")

    def _plot_metric_distribution(self, metric: str, output_dir: str) -> None:
        # Skip if metric not in DataFrame
        if metric not in self.metrics_df.columns:
            print(f"Metric {metric} not found in metrics DataFrame")
            return

        # Check if there's enough data
        if len(self.metrics_df[metric].dropna()) < 2:
            print(f"Not enough data for metric {metric} to create distribution plot")
            return

        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.metrics_df[metric].dropna(), kde=True)
            plt.title(f"Distribution of {metric}")
            plt.xlabel(metric)
            plt.ylabel('Frequency')
            if metric.lower() in [m.lower() for m in fm.BIAS_METRICS]:
                plt.axvline(0, linestyle='--')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{metric}_distribution.png")
        except Exception as e:
            print(f"Error creating distribution plot for {metric}: {e}")
        finally:
            plt.close()

    def _plot_metric_by_sku(self, metric: str, output_dir: str) -> None:
        # Skip if metric not in DataFrame
        if metric not in self.metrics_df.columns:
            print(f"Metric {metric} not found in metrics DataFrame")
            return

        try:
            plt.figure(figsize=(12, 6))
            sku_means = self.metrics_df.groupby('SKU')[metric].mean().sort_values()

            # Check if there's enough data
            if len(sku_means) < 1:
                print(f"Not enough SKUs with data for metric {metric} to create bar plot")
                return

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
        except Exception as e:
            print(f"Error creating SKU bar plot for {metric}: {e}")
        finally:
            plt.close()

    def _plot_anomaly_comparison(self, output_dir: str) -> None:
        if 'data_anomaly_rate' not in self.metrics_df.columns or 'residual_anomaly_rate' not in self.metrics_df.columns:
            print("Anomaly rate metrics not found in metrics DataFrame")
            return

        try:
            plt.figure(figsize=(10, 6))
            data = self.metrics_df[['SKU', 'data_anomaly_rate', 'residual_anomaly_rate']].melt(
                id_vars=['SKU'], var_name='Anomaly Type', value_name='Rate')

            # Check if there's enough data
            if len(data.dropna()) < 2:
                print("Not enough anomaly rate data to create boxplot")
                return

            sns.boxplot(x='Anomaly Type', y='Rate', data=data)
            plt.title('Data vs Residual Anomaly Rates')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/anomaly_comparison.png")
        except Exception as e:
            print(f"Error creating anomaly comparison plot: {e}")
        finally:
            plt.close()

    def _plot_metric_correlations(self, output_dir: str) -> None:
        # Get metric columns that have valid data
        metric_cols = [c for c in self.metrics_df.columns
                       if c not in ['SKU', 'Actual_Month']
                       and self.metrics_df[c].dtype.kind in 'fc'  # float or complex types
                       and not self.metrics_df[c].isna().all()
                       and len(self.metrics_df[c].dropna()) > 1]

        if not metric_cols:
            print("No valid metric columns found for correlation analysis")
            return

        try:
            plt.figure(figsize=(12, 10))
            # Calculate correlation matrix
            corr = self.metrics_df[metric_cols].corr()
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))
            # Plot heatmap
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                        vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
            plt.title('Correlation Between Metrics')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/metric_correlations.png")
        except Exception as e:
            print(f"Error plotting correlation matrix: {e}")
        finally:
            plt.close()
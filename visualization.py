"""
Visualization Module

This module provides functions for visualizing forecast metrics, thresholds, and lag-based predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional

# Import from other modules
import forecast_metrics as fm
from thresholds import get_performance_level


def create_threshold_chart(
        metrics_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        metric: str,
        sku: Optional[str] = None,
        output_dir: str = '.'
) -> str:
    """
    Create a visualization of metric values with green/yellow/red threshold zones.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        thresholds_df (pd.DataFrame): DataFrame containing threshold values
        metric (str): Metric name to visualize
        sku (str, optional): SKU to filter on. If None, uses global thresholds.
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Skip if metric not in DataFrame
    if metric not in metrics_df.columns:
        print(f"Metric {metric} not found in metrics DataFrame")
        return None

    # Filter thresholds for the specified metric
    metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

    # Skip if no thresholds found
    if metric_thresh.empty:
        print(f"No thresholds found for metric {metric}")
        return None

    # If SKU is specified, filter for that SKU
    if sku:
        if 'SKU' in metric_thresh.columns:
            sku_thresh = metric_thresh[metric_thresh['SKU'] == sku]
            if not sku_thresh.empty:
                metric_thresh = sku_thresh
            title_prefix = f"{sku} - "
            sku_data = metrics_df[metrics_df['SKU'] == sku]
            # Skip if no data for this SKU
            if sku_data.empty:
                print(f"No data found for SKU {sku}")
                return None
            metrics_df = sku_data
        else:
            # If SKU not in thresholds_df, might be using global thresholds
            title_prefix = f"{sku} vs Global - "
            sku_data = metrics_df[metrics_df['SKU'] == sku]
            # Skip if no data for this SKU
            if sku_data.empty:
                print(f"No data found for SKU {sku}")
                return None
            metrics_df = sku_data
    else:
        title_prefix = "Global - "

    # Extract threshold values
    green_val = metric_thresh.iloc[0]['Green']
    yellow_val = metric_thresh.iloc[0]['Yellow']

    # Determine if higher or lower is better
    red_condition = metric_thresh.iloc[0]['Red_Condition']
    higher_is_better = '<' in red_condition

    try:
        # Create figure
        plt.figure(figsize=(12, 6))

        # Determine if we're plotting over time or as a histogram
        if 'Actual_Month' in metrics_df.columns and not metrics_df['Actual_Month'].isna().all():
            # Get data sorted by month
            plot_data = metrics_df.sort_values('Actual_Month')

            # Plot line
            plt.plot(plot_data['Actual_Month'], plot_data[metric], marker='o', linewidth=2)

            # Format x-axis for dates
            plt.xticks(rotation=45)
            plt.xlabel('Month', fontsize=12)

        else:
            # Create histogram
            sns.histplot(metrics_df[metric].dropna(), kde=True)
            plt.xlabel(metric, fontsize=12)

        # Add threshold lines and zones
        if higher_is_better:
            # For metrics where higher is better
            plt.axhline(yellow_val, color='gold', linestyle='--', label=f'Yellow Threshold: {yellow_val:.3f}')
            plt.axhline(green_val, color='green', linestyle='--', label=f'Green Threshold: {green_val:.3f}')

            # Add colored background zones
            ymin, ymax = plt.ylim()
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], yellow_val, ymin, color='red', alpha=0.1)
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], yellow_val, green_val, color='gold', alpha=0.1)
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], green_val, ymax, color='green', alpha=0.1)

        else:
            # For metrics where lower is better
            plt.axhline(yellow_val, color='gold', linestyle='--', label=f'Yellow Threshold: {yellow_val:.3f}')
            plt.axhline(green_val, color='green', linestyle='--', label=f'Green Threshold: {green_val:.3f}')

            # Add colored background zones
            ymin, ymax = plt.ylim()
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], yellow_val, ymax, color='red', alpha=0.1)
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], green_val, yellow_val, color='gold', alpha=0.1)
            plt.fill_between([plt.xlim()[0], plt.xlim()[1]], ymin, green_val, color='green', alpha=0.1)

        # Add title and labels
        plt.title(f"{title_prefix}{metric} Performance vs Thresholds", fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.legend(loc='best')

        # Add business impact annotation if available
        if 'Business_Impact' in metric_thresh.columns:
            impact = metric_thresh.iloc[0]['Business_Impact']
            plt.figtext(
                0.5, -0.05,
                f"Business Impact: {impact}",
                ha='center',
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray")
            )

        plt.tight_layout()

        # Save the figure
        if sku:
            output_path = os.path.join(output_dir, f"{sku}_{metric}_threshold.png")
        else:
            output_path = os.path.join(output_dir, f"global_{metric}_threshold.png")

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating threshold chart for {metric}: {e}")
        plt.close()
        return None


def create_performance_heatmap(
        metrics_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        output_dir: str = '.'
) -> str:
    """
    Create a heatmap showing performance levels across SKUs and metrics.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        thresholds_df (pd.DataFrame): DataFrame containing threshold values
        metrics (List[str], optional): List of metrics to include. If None, uses all metrics.
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # If metrics not specified, get all metrics from thresholds_df
        if metrics is None:
            metrics = thresholds_df['Metric'].unique().tolist()

        # Filter to ensure metrics exist in the DataFrame
        metrics = [m for m in metrics if m in metrics_df.columns]

        # Skip if no valid metrics
        if not metrics:
            print("No valid metrics found for performance heatmap")
            return None

        # Get list of SKUs
        skus = metrics_df['SKU'].unique().tolist()

        # Skip if no SKUs
        if not skus:
            print("No SKUs found in the metrics DataFrame")
            return None

        # Create performance matrix (SKU x Metric)
        # 2 = Green, 1 = Yellow, 0 = Red
        performance_matrix = pd.DataFrame(index=skus, columns=metrics)

        # Populate matrix with performance levels
        for sku in skus:
            sku_data = metrics_df[metrics_df['SKU'] == sku]

            # Skip if there's no data for this SKU
            if sku_data.empty:
                continue

            for metric in metrics:
                # Skip if metric not in SKU data
                if metric not in sku_data.columns:
                    continue

                # Skip if all metric values are NaN for this SKU
                if sku_data[metric].isna().all():
                    continue

                try:
                    # Get average metric value for this SKU
                    metric_value = sku_data[metric].mean()

                    # Skip if value is NaN
                    if pd.isna(metric_value):
                        continue

                    # Get thresholds for this metric
                    metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

                    # Skip if no thresholds found
                    if metric_thresh.empty:
                        continue

                    # Get SKU-specific thresholds if available, otherwise use global
                    if 'SKU' in metric_thresh.columns:
                        sku_thresh = metric_thresh[metric_thresh['SKU'] == sku]
                        if not sku_thresh.empty:
                            metric_thresh = sku_thresh

                    # Get threshold values
                    green_val = metric_thresh.iloc[0]['Green']
                    yellow_val = metric_thresh.iloc[0]['Yellow']

                    # Determine if higher or lower is better
                    red_condition = metric_thresh.iloc[0]['Red_Condition']
                    higher_is_better = '<' in red_condition

                    # Determine performance level
                    if higher_is_better:
                        if metric_value >= green_val:
                            performance_level = 2  # Green
                        elif metric_value >= yellow_val:
                            performance_level = 1  # Yellow
                        else:
                            performance_level = 0  # Red
                    else:
                        if metric_value <= green_val:
                            performance_level = 2  # Green
                        elif metric_value <= yellow_val:
                            performance_level = 1  # Yellow
                        else:
                            performance_level = 0  # Red

                    # Add to matrix
                    performance_matrix.at[sku, metric] = performance_level

                except Exception as e:
                    print(f"Error processing SKU {sku}, metric {metric}: {e}")
                    continue

        # Check if performance matrix has any valid data
        if performance_matrix.isna().all().all():
            print("No valid data to create performance heatmap")
            return None

        # Create figure
        plt.figure(figsize=(max(8, len(metrics) * 1.2), max(6, len(skus) * 0.4)))

        # Fill NaN values with a neutral value for the heatmap
        performance_matrix = performance_matrix.fillna(-1)

        # Create heatmap
        sns.heatmap(
            performance_matrix,
            cmap=['red', 'gold', 'green', 'gray'],  # Added gray for NaN values
            linewidths=.5,
            linecolor='gray',
            cbar=False,
            vmin=-1,
            vmax=2
        )

        # Add color bar legend
        cbar = plt.colorbar(ticks=[-0.5, 0.33, 1, 1.67])
        cbar.set_ticklabels(['N/A', 'Red', 'Yellow', 'Green'])

        # Add title and labels
        plt.title("Performance Heatmap by SKU and Metric", fontsize=14)
        plt.ylabel("SKU", fontsize=12)
        plt.xlabel("Metric", fontsize=12)

        # Rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, "performance_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating performance heatmap: {e}")
        plt.close()
        return None


def create_business_impact_chart(
        metrics_df: pd.DataFrame,
        output_dir: str = '.'
) -> str:
    """
    Create a business impact visualization showing the relationship
    between key metrics and their potential business outcomes.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Select key business impact metrics
        impact_metrics = {
            'mean_bias': {
                'title': 'Inventory Impact',
                'x_label': 'Mean Bias',
                'positive_impact': 'Excess Inventory',
                'negative_impact': 'Stockouts'
            },
            'direction_accuracy': {
                'title': 'Planning Impact',
                'x_label': 'Direction Accuracy',
                'low_impact': 'Missed Opportunities',
                'high_impact': 'Effective Planning'
            },
            'sliding_jsd': {
                'title': 'Model Relevance',
                'x_label': 'Distribution Shift (JSD)',
                'low_impact': 'Stable Patterns',
                'high_impact': 'Model Degradation'
            }
        }

        # Filter metrics that exist in the DataFrame
        avail_metrics = [m for m in impact_metrics.keys() if m in metrics_df.columns]

        # If no valid metrics, return None
        if not avail_metrics:
            print("No valid impact metrics found in the DataFrame")
            return None

        # Create a multi-panel figure
        n_metrics = len(avail_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, n_metrics * 4))

        # If only one metric, axes is not an array
        if n_metrics == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(avail_metrics):
            try:
                # Skip if all values for this metric are NaN
                if metrics_df[metric].isna().all():
                    print(f"All values for metric {metric} are NaN, skipping")
                    continue

                # Calculate SKU averages
                sku_avg = metrics_df.groupby('SKU')[metric].mean().dropna().reset_index()

                # Skip if no data
                if sku_avg.empty:
                    print(f"No valid data for metric {metric}, skipping")
                    continue

                # Sort by metric value
                sku_avg = sku_avg.sort_values(metric)

                # Get metric config
                config = impact_metrics[metric]

                # Create horizontal bar chart
                bars = sns.barplot(
                    y='SKU',
                    x=metric,
                    data=sku_avg,
                    palette='RdYlGn_r' if metric in ['mean_bias', 'sliding_jsd'] else 'RdYlGn',
                    ax=axes[i]
                )

                # Add color gradient legend
                if metric in ['mean_bias', 'sliding_jsd']:
                    axes[i].text(
                        sku_avg[metric].min() * 0.9,
                        -0.5,
                        config['negative_impact'] if 'negative_impact' in config else config['low_impact'],
                        ha='center',
                        fontsize=10,
                        bbox=dict(facecolor='green', alpha=0.2)
                    )
                    axes[i].text(
                        sku_avg[metric].max() * 0.9,
                        -0.5,
                        config['positive_impact'] if 'positive_impact' in config else config['high_impact'],
                        ha='center',
                        fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.2)
                    )
                else:
                    axes[i].text(
                        sku_avg[metric].min() * 0.9,
                        -0.5,
                        config['low_impact'],
                        ha='center',
                        fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.2)
                    )
                    axes[i].text(
                        sku_avg[metric].max() * 0.9,
                        -0.5,
                        config['high_impact'],
                        ha='center',
                        fontsize=10,
                        bbox=dict(facecolor='green', alpha=0.2)
                    )

                # Add title and labels
                axes[i].set_title(config['title'], fontsize=14)
                axes[i].set_xlabel(config['x_label'], fontsize=12)
                axes[i].set_ylabel('SKU', fontsize=12)

                # Add vertical line at zero for Mean_Bias
                if metric == 'mean_bias':
                    axes[i].axvline(0, color='black', linestyle='--', alpha=0.7)

            except Exception as e:
                print(f"Error creating business impact visualization for {metric}: {e}")
                # If the subplot fails, make it empty
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"Error plotting {metric}", ha='center', va='center')

        # Add overall title
        fig.suptitle("Business Impact Analysis", fontsize=16, y=1.02)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, "business_impact.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating business impact chart: {e}")
        plt.close()
        return None


def create_lag_comparison_chart(
        lag_metrics: pd.DataFrame,
        metrics: List[str] = None,
        output_dir: str = '.'
) -> str:
    """
    Create a visualization comparing multiple metrics across forecast horizons.

    Parameters:
        lag_metrics (pd.DataFrame): DataFrame with metrics by lag
        metrics (List[str], optional): List of metrics to include.
            If None, uses default key metrics.
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Ensure lag column exists
        if 'lag' not in lag_metrics.columns:
            print("'lag' column not found in lag_metrics DataFrame")
            return None

        # Default metrics if not provided
        if metrics is None:
            metrics = ['Mean_Bias', 'Direction_Accuracy', 'Tracking_Signal']

        # Filter to metrics that exist in the DataFrame
        metrics = [m for m in metrics if m in lag_metrics.columns]

        # Skip if no valid metrics
        if not metrics:
            print("No valid metrics found for lag comparison chart")
            return None

        # Create figure with multiple subplots
        fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)

        # Handle case where metrics is of length 1
        if len(metrics) == 1:
            axes = [axes]

        # Plot each metric
        for i, metric in enumerate(metrics):
            try:
                # Skip if all values for this metric are NaN
                if lag_metrics[metric].isna().all():
                    print(f"All values for metric {metric} are NaN, skipping")
                    axes[i].axis('off')
                    axes[i].text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                    continue

                # Group by lag
                lag_grouped = lag_metrics.groupby('lag')[metric].mean().dropna().reset_index()

                # Skip if no data
                if lag_grouped.empty:
                    print(f"No valid data for metric {metric}, skipping")
                    axes[i].axis('off')
                    axes[i].text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
                    continue

                # Plot line
                axes[i].plot(lag_grouped['lag'], lag_grouped[metric], marker='o', linewidth=2)

                # Add counts as bubble size if available
                if 'count' in lag_grouped.columns:
                    sizes = lag_grouped['count'] / lag_grouped['count'].max() * 100
                    axes[i].scatter(lag_grouped['lag'], lag_grouped[metric], s=sizes, alpha=0.5)

                # Add reference line at 0 or 0.5 based on metric type
                if metric.lower() in [m.lower() for m in fm.BIAS_METRICS]:
                    axes[i].axhline(0, color='r', linestyle='--', alpha=0.7)
                elif metric.lower() in [m.lower() for m in fm.PERFORMANCE_METRICS]:
                    axes[i].axhline(0.5, color='r', linestyle='--', alpha=0.7)

                # Add title and labels
                axes[i].set_title(f"{metric} by Forecast Horizon", fontsize=12)
                axes[i].set_ylabel(metric, fontsize=10)

                # Add business context for specific metrics
                if metric.lower() == 'mean_bias':
                    axes[i].text(
                        0.05, 0.05,
                        "Positive bias → Excess inventory\nNegative bias → Stockouts",
                        transform=axes[i].transAxes,
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
                elif metric.lower() == 'direction_accuracy':
                    axes[i].text(
                        0.05, 0.05,
                        "High accuracy → Reliable trend signals",
                        transform=axes[i].transAxes,
                        fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )

            except Exception as e:
                print(f"Error creating lag comparison visualization for {metric}: {e}")
                # If the subplot fails, make it empty
                axes[i].axis('off')
                axes[i].text(0.5, 0.5, f"Error plotting {metric}", ha='center', va='center')

        # Add common x-axis label
        fig.text(0.5, 0.04, "Forecast Horizon (months)", ha='center', fontsize=12)

        # Add overall title
        fig.suptitle("Forecast Accuracy by Horizon", fontsize=14, y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave room for common x-label

        # Format x-axis as integers if we have data
        if not lag_metrics['lag'].isna().all():
            unique_lags = sorted(lag_metrics['lag'].dropna().unique())
            if unique_lags:
                plt.xticks(unique_lags)

        # Save the figure
        output_path = os.path.join(output_dir, "lag_comparison_chart.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating lag comparison chart: {e}")
        plt.close()
        return None


def plot_forecast_accuracy_matrix(
        metrics_df: pd.DataFrame,
        thresholds_df: pd.DataFrame,
        lag_metrics_df: pd.DataFrame,
        output_dir: str = '.'
) -> str:
    """
    Create an integrated visualization showing forecast accuracy
    across multiple dimensions: SKUs, metrics, and forecast horizons.

    Parameters:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        thresholds_df (pd.DataFrame): DataFrame containing threshold values
        lag_metrics_df (pd.DataFrame): DataFrame with metrics by SKU and lag
        output_dir (str): Directory to save visualization

    Returns:
        str: Path to the saved visualization file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Select key metrics to visualize
        key_metrics = ['mean_bias', 'direction_accuracy', 'tracking_signal']
        key_metrics = [m for m in key_metrics if m in metrics_df.columns]

        # If no valid metrics, return None
        if not key_metrics:
            print("No valid key metrics found for forecast accuracy matrix")
            return None

        # Create figure with grid layout
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3)

        # 1. Top left: Performance heatmap
        ax1 = fig.add_subplot(gs[0, 0])

        # Create mini performance matrix (SKU x Key Metrics)
        skus = metrics_df['SKU'].unique().tolist()
        perf_matrix = pd.DataFrame(index=skus, columns=key_metrics)

        # Populate performance matrix
        for sku in skus:
            sku_data = metrics_df[metrics_df['SKU'] == sku]

            for metric in key_metrics:
                if metric not in sku_data.columns or sku_data[metric].isna().all():
                    continue

                try:
                    # Get average metric value for this SKU
                    metric_value = sku_data[metric].mean()

                    # Skip if value is NaN
                    if pd.isna(metric_value):
                        continue

                    # Get performance level using thresholds
                    try:
                        # Get thresholds for this metric
                        metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

                        # Skip if no thresholds found
                        if metric_thresh.empty:
                            continue

                        # Get SKU-specific thresholds if available, otherwise use global
                        if 'SKU' in metric_thresh.columns:
                            sku_thresh = metric_thresh[metric_thresh['SKU'] == sku]
                            if not sku_thresh.empty:
                                metric_thresh = sku_thresh

                        # Get threshold values
                        green_val = metric_thresh.iloc[0]['Green']
                        yellow_val = metric_thresh.iloc[0]['Yellow']

                        # Determine if higher or lower is better
                        red_condition = metric_thresh.iloc[0]['Red_Condition']
                        higher_is_better = '<' in red_condition

                        # Determine performance level
                        if higher_is_better:
                            if metric_value >= green_val:
                                level_value = 2  # Green
                            elif metric_value >= yellow_val:
                                level_value = 1  # Yellow
                            else:
                                level_value = 0  # Red
                        else:
                            if metric_value <= green_val:
                                level_value = 2  # Green
                            elif metric_value <= yellow_val:
                                level_value = 1  # Yellow
                            else:
                                level_value = 0  # Red

                        # Add to matrix
                        perf_matrix.at[sku, metric] = level_value

                    except Exception as e:
                        print(f"Error determining performance level for {sku}, {metric}: {e}")
                        continue

                except Exception as e:
                    print(f"Error calculating mean for {sku}, {metric}: {e}")
                    continue

        # Check if perf_matrix has data
        if perf_matrix.isna().all().all():
            # No data, display a message
            ax1.axis('off')
            ax1.text(0.5, 0.5, "No performance data available",
                     ha='center', va='center', fontsize=10)
        else:
            # Fill NA values with a neutral value (-1) for visualization
            perf_matrix = perf_matrix.fillna(-1)

            # Create heatmap
            sns.heatmap(
                perf_matrix,
                cmap=['red', 'gold', 'green', 'gray'],  # Gray for NA values
                linewidths=.5,
                linecolor='gray',
                cbar=False,
                vmin=-1,
                vmax=2,
                ax=ax1
            )

            ax1.set_title("Performance by SKU")
            ax1.set_xlabel("Metric")
            ax1.set_ylabel("SKU")

        # 2. Top middle and right: Lag Impact
        ax2 = fig.add_subplot(gs[0, 1:])

        # Check if lag_metrics_df has the required data
        if 'lag' not in lag_metrics_df.columns:
            # No lag column, display a message
            ax2.axis('off')
            ax2.text(0.5, 0.5, "No forecast horizon data available",
                     ha='center', va='center', fontsize=10)
        else:
            # Filter lag_metrics_df for a representative SKU or use global averages
            valid_metrics = [m for m in key_metrics if m in lag_metrics_df.columns
                            and not lag_metrics_df[m].isna().all()]

            if not valid_metrics:
                # No valid metrics, display a message
                ax2.axis('off')
                ax2.text(0.5, 0.5, "No valid metrics for forecast horizon analysis",
                         ha='center', va='center', fontsize=10)
            else:
                # Use most common SKU if SKU column exists, otherwise use all data
                if 'SKU' in lag_metrics_df.columns and not lag_metrics_df['SKU'].isna().all():
                    # Get most common SKU
                    rep_sku = lag_metrics_df['SKU'].value_counts().idxmax()
                    lag_data = lag_metrics_df[lag_metrics_df['SKU'] == rep_sku]
                    title = f"Metric Changes by Forecast Horizon (SKU: {rep_sku})"
                else:
                    lag_data = lag_metrics_df
                    title = "Metric Changes by Forecast Horizon (Global Average)"

                # Group by lag for each metric
                for i, metric in enumerate(valid_metrics):
                    try:
                        # Group by lag and calculate average
                        lag_avg = lag_data.groupby('lag')[metric].mean().dropna().reset_index()

                        # Skip if no data
                        if lag_avg.empty:
                            continue

                        # Get color from a palette
                        colors = ['blue', 'green', 'red', 'purple', 'orange']
                        color = colors[i % len(colors)]

                        # Plot line
                        ax2.plot(
                            lag_avg['lag'],
                            lag_avg[metric],
                            marker='o',
                            color=color,
                            label=metric
                        )
                    except Exception as e:
                        print(f"Error plotting {metric} by lag: {e}")

                ax2.set_title(title)
                ax2.set_xlabel("Forecast Horizon (months)")
                ax2.set_ylabel("Metric Value")
                ax2.legend(loc='best')

        # 3. Middle row: Metric distributions
        distributions = []
        for i, metric in enumerate(key_metrics[:3]):  # Up to 3 metrics
            ax = fig.add_subplot(gs[1, i])
            distributions.append(ax)

            try:
                # Check if metric exists and has data
                if metric not in metrics_df.columns or metrics_df[metric].isna().all():
                    # No data, display a message
                    ax.axis('off')
                    ax.text(0.5, 0.5, f"No data for {metric}",
                           ha='center', va='center', fontsize=10)
                else:
                    # Create histogram
                    sns.histplot(metrics_df[metric].dropna(), kde=True, ax=ax)

                    # Add threshold lines if available
                    metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]
                    if not metric_thresh.empty:
                        green_val = metric_thresh.iloc[0]['Green']
                        yellow_val = metric_thresh.iloc[0]['Yellow']

                        ax.axvline(green_val, color='green', linestyle='--', label='Green')
                        ax.axvline(yellow_val, color='gold', linestyle='--', label='Yellow')

                    ax.set_title(f"{metric} Distribution")
                    ax.set_xlabel(metric)
                    ax.set_ylabel("Frequency")
            except Exception as e:
                print(f"Error creating distribution for {metric}: {e}")
                ax.axis('off')
                ax.text(0.5, 0.5, f"Error with {metric} distribution",
                       ha='center', va='center', fontsize=10)

        # 4. Bottom row: Business impact summaries
        impact_summaries = []
        impacts = {
            'mean_bias': "Affects inventory levels and resource allocation efficiency",
            'direction_accuracy': "Critical for trend-based decisions and capacity planning",
            'tracking_signal': "Indicates persistent forecast misalignment requiring adjustment",
            'data_anomaly_rate': "Highlights data quality issues requiring investigation",
            'turning_point_f1': "Essential for detecting market shifts and strategy changes"
        }

        for i, metric in enumerate(key_metrics[:3]):  # Up to 3 metrics
            ax = fig.add_subplot(gs[2, i])
            impact_summaries.append(ax)

            try:
                # Remove axes
                ax.axis('off')

                # Check if metric exists and has data
                if metric not in metrics_df.columns or metrics_df[metric].isna().all():
                    ax.text(0.5, 0.5, f"No data available for {metric}",
                           ha='center', va='center', fontsize=10)
                else:
                    # Calculate key statistics
                    avg = metrics_df[metric].mean()

                    # Find worst SKU
                    try:
                        worst_sku_data = metrics_df.groupby('SKU')[metric].mean().dropna()

                        if not worst_sku_data.empty:
                            if metric.lower() in [m.lower() for m in fm.PERFORMANCE_METRICS]:
                                # For performance metrics, lower is worse
                                worst_sku = worst_sku_data.idxmin()
                                trend = "higher is better"
                            else:
                                # For error metrics, higher is worse
                                worst_sku = worst_sku_data.idxmax()
                                trend = "lower is better"
                        else:
                            worst_sku = "N/A"
                            trend = "undetermined"
                    except Exception:
                        worst_sku = "Error determining"
                        trend = "undetermined"

                    # Add business impact text
                    impact_text = f"""
                    Metric: {metric}
                    Average: {avg:.3f}
                    Direction: {trend}
                    
                    Business Impact:
                    {impacts.get(metric.lower(), 'Affects forecast quality')}
                    
                    Focus Area:
                    SKU {worst_sku} needs attention
                    """

                    ax.text(0.05, 0.95, impact_text, va='top', fontsize=9)
            except Exception as e:
                print(f"Error creating impact summary for {metric}: {e}")
                ax.text(0.5, 0.5, f"Error processing {metric} impact",
                       ha='center', va='center', fontsize=10)

        # Add overall title
        fig.suptitle("Integrated Forecast Accuracy Matrix", fontsize=16)
        plt.tight_layout()

        # Save the figure
        output_path = os.path.join(output_dir, "forecast_accuracy_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path

    except Exception as e:
        print(f"Error creating forecast accuracy matrix: {e}")
        plt.close()
        return None
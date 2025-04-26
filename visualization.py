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

    # Filter thresholds for the specified metric
    metric_thresh = thresholds_df[thresholds_df['Metric'] == metric]

    # If SKU is specified, filter for that SKU
    if sku:
        if 'SKU' in metric_thresh.columns:
            metric_thresh = metric_thresh[metric_thresh['SKU'] == sku]
            title_prefix = f"{sku} - "
            metrics_df = metrics_df[metrics_df['SKU'] == sku]
        else:
            # If SKU not in thresholds_df, might be using global thresholds
            title_prefix = f"{sku} vs Global - "
            metrics_df = metrics_df[metrics_df['SKU'] == sku]
    else:
        title_prefix = "Global - "

    # If no matching thresholds were found, return None
    if metric_thresh.empty:
        return None

    # Extract threshold values
    green_val = metric_thresh.iloc[0]['Green']
    yellow_val = metric_thresh.iloc[0]['Yellow']

    # Determine if higher or lower is better
    red_condition = metric_thresh.iloc[0]['Red_Condition']
    higher_is_better = '<' in red_condition

    # Create figure
    plt.figure(figsize=(12, 6))

    # Determine if we're plotting over time or as a histogram
    if 'Actual_Month' in metrics_df.columns:
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If metrics not specified, get all metrics from thresholds_df
    if metrics is None:
        metrics = thresholds_df['Metric'].unique().tolist()

    # Get list of SKUs
    skus = metrics_df['SKU'].unique().tolist()

    # Create performance matrix (SKU x Metric)
    # 2 = Green, 1 = Yellow, 0 = Red
    performance_matrix = pd.DataFrame(index=skus, columns=metrics)

    # Populate matrix with performance levels
    for sku in skus:
        sku_data = metrics_df[metrics_df['SKU'] == sku]

        for metric in metrics:
            if metric not in sku_data.columns:
                continue

            # Get average metric value for this SKU
            metric_value = sku_data[metric].mean()

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

    # Create figure
    plt.figure(figsize=(max(8, len(metrics) * 1.2), max(6, len(skus) * 0.4)))

    # Create heatmap
    sns.heatmap(
        performance_matrix,
        cmap=['red', 'gold', 'green'],
        linewidths=.5,
        linecolor='gray',
        cbar=False,
        vmin=0,
        vmax=2
    )

    # Add color bar legend
    cbar = plt.colorbar(ticks=[0.33, 1, 1.67])
    cbar.set_ticklabels(['Red', 'Yellow', 'Green'])

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
        return None

    # Create a multi-panel figure
    n_metrics = len(avail_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, n_metrics * 4))

    # If only one metric, axes is not an array
    if n_metrics == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(avail_metrics):
        # Calculate SKU averages
        sku_avg = metrics_df.groupby('SKU')[metric].mean().reset_index()

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

    # Add overall title
    fig.suptitle("Business Impact Analysis", fontsize=16, y=1.02)
    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, "business_impact.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Default metrics if not provided
    if metrics is None:
        metrics = ['Mean_Bias', 'Direction_Accuracy', 'Tracking_Signal']
        metrics = [m for m in metrics if m in lag_metrics.columns]

    # Skip if no valid metrics
    if not metrics:
        return None

    # Create figure with multiple subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)

    # Handle case where metrics is of length 1
    if len(metrics) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric in enumerate(metrics):
        # Group by lag
        lag_grouped = lag_metrics.groupby('lag')[metric].mean().reset_index()

        # Plot line
        axes[i].plot(lag_grouped['lag'], lag_grouped[metric], marker='o', linewidth=2)

        # Add counts as bubble size if available
        if 'count' in lag_grouped.columns:
            sizes = lag_grouped['count'] / lag_grouped['count'].max() * 100
            axes[i].scatter(lag_grouped['lag'], lag_grouped[metric], s=sizes, alpha=0.5)

        # Add reference line at 0 or 0.5 based on metric type
        if metric.lower() in fm.BIAS_METRICS:
            axes[i].axhline(0, color='r', linestyle='--', alpha=0.7)
        elif metric.lower() in fm.PERFORMANCE_METRICS:
            axes[i].axhline(0.5, color='r', linestyle='--', alpha=0.7)
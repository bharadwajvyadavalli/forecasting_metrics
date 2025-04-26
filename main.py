"""
Forecast Metrics Framework

This script provides a simplified entry point for analyzing forecast accuracy
through multiple metric dimensions, with clear explanations of each metric's
importance and business relevance.

Usage:
    python main.py [--data=sample_data.csv] [--output=output]
"""

import os
import argparse
import pandas as pd
import numpy as np

import metrics_calculator
from data_generator import generate_forecast_data
import visualization as viz
import thresholds as th
import lag_based_predictions as lag


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forecast Metrics Analysis')
    parser.add_argument('--data', type=str, default='sample_data.csv',
                        help='Path to forecast data CSV (default: sample_data.csv)')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save output (default: output)')
    parser.add_argument('--generate', action='store_true',
                        help='Generate new sample data')
    return parser.parse_args()


def explain_metrics():
    """Print explanation of metrics and their business relevance."""
    print("\n===== FORECAST METRICS EXPLANATION =====")
    print("\n1. BIAS METRICS - Detect Systematic Under/Over Prediction")
    print("   • Mean Bias: Average error, indicates systematic over/under forecasting")
    print("     Business Impact: Affects inventory levels, resource allocation efficiency")
    print("   • Tracking Signal: Normalized bias, flags persistent misalignment")
    print("     Business Impact: Indicates when model parameters need adjustment")

    print("\n2. ANOMALY METRICS - Identify Outliers and Unusual Patterns")
    print("   • Data Anomaly Rate: Proportion of unusual actual values")
    print("     Business Impact: Highlights data quality issues requiring investigation")
    print("   • Residual Anomaly Rate: Proportion of unusual forecast errors")
    print("     Business Impact: Shows forecast sensitivity to outliers affecting reliability")

    print("\n3. DIRECTIONAL METRICS - Evaluate Trend and Pattern Accuracy")
    print("   • Direction Accuracy: Percentage of correctly predicted up/down movements")
    print("     Business Impact: Critical for trend-based decisions and capacity planning")
    print("   • Turning Point F1: Accuracy in detecting trend reversals")
    print("     Business Impact: Essential for detecting market shifts and strategy changes")

    print("\n4. DISTRIBUTION METRICS - Assess Shifts and Stability")
    print("   • Sliding JSD: Measures distribution stability over time")
    print("     Business Impact: Identifies when past data becomes less relevant to future")

    print("\n5. CALIBRATION METRICS - Evaluate Probabilistic Accuracy")
    print("   • CRPS: Continuous Ranked Probability Score for probabilistic forecasts")
    print("     Business Impact: Evaluates forecast uncertainty quantification for risk assessment")


def main():
    """Main function to run the forecast metrics analysis."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Generate sample data if requested
    if args.generate:
        print("Generating sample forecast data...")
        data = generate_forecast_data()
        data.to_csv(args.data, index=False)
        print(f"Sample data saved to {args.data}")

    # Explain metrics and their business relevance
    explain_metrics()

    # Initialize metrics calculator
    print(f"\nAnalyzing forecast data from {args.data}...")
    calculator = metrics_calculator.MetricsCalculator(args.data)

    # Compute metrics and thresholds
    metrics_df = calculator.compute_metrics()
    sku_thresholds, global_thresholds = calculator.compute_thresholds()

    # Generate reports
    calculator.generate_report(args.output)

    # Generate visualizations
    calculator.visualize_key_metrics(args.output)

    # Create additional visualizations
    print("Creating threshold and performance visualizations...")

    # Performance heatmap
    viz.create_performance_heatmap(metrics_df, global_thresholds, output_dir=args.output)

    # Key metric threshold charts
    for metric in ['mean_bias', 'direction_accuracy', 'data_anomaly_rate']:
        if metric in metrics_df.columns:
            viz.create_threshold_chart(metrics_df, global_thresholds, metric, output_dir=args.output)

    # Business impact chart
    viz.create_business_impact_chart(metrics_df, output_dir=args.output)

    # Calculate lag-based metrics
    print("\nAnalyzing forecast accuracy by forecast horizon...")
    lag_metrics = lag.compute_metrics_by_lag(calculator.data)
    lag_metrics_by_sku = lag.compute_metrics_by_lag_and_sku(calculator.data)

    # Save lag-based metrics
    lag_metrics.to_csv(f"{args.output}/lag_metrics.csv", index=False)
    lag_metrics_by_sku.to_csv(f"{args.output}/lag_metrics_by_sku.csv", index=False)

    # Generate lag-based visualizations
    for metric in ['mean_bias', 'direction_accuracy']:
        if metric in lag_metrics.columns:
            lag.plot_metric_by_lag(lag_metrics, metric, args.output)
            lag.plot_lag_heatmap(lag_metrics_by_sku, metric, args.output)

    # Create lag comparison chart
    viz.create_lag_comparison_chart(lag_metrics, output_dir=args.output)

    # Create integrated accuracy matrix
    viz.plot_forecast_accuracy_matrix(
        metrics_df, global_thresholds, lag_metrics_by_sku, args.output
    )

    # Generate lag report
    lag_report = lag.generate_lag_report(lag_metrics)
    with open(f"{args.output}/lag_analysis_report.txt", 'w') as f:
        f.write(lag_report)

    # Generate threshold report
    threshold_report = th.generate_threshold_report(metrics_df, global_thresholds)
    with open(f"{args.output}/threshold_report.txt", 'w') as f:
        f.write(threshold_report)

    # Print summary
    print("\n===== ANALYSIS SUMMARY =====")
    print(f"Analyzed {metrics_df['SKU'].nunique()} SKUs over {metrics_df['Actual_Month'].nunique()} time periods")

    # SKUs with highest bias
    if 'mean_bias' in metrics_df.columns:
        high_bias = metrics_df.groupby('SKU')['mean_bias'].mean().abs().sort_values(ascending=False).head(3)
        print("\nSKUs with highest absolute bias:")
        for sku, value in high_bias.items():
            print(f"  • {sku}: {value:.3f}")

    # Top performing SKUs by Direction Accuracy
    if 'direction_accuracy' in metrics_df.columns:
        top_skus = metrics_df.groupby('SKU')['direction_accuracy'].mean().sort_values(ascending=False).head(3)
        print("\nTop performing SKUs by Direction Accuracy:")
        for sku, value in top_skus.items():
            print(f"  • {sku}: {value:.3f}")

    # Overall recommendation
    print("\n===== RECOMMENDATIONS =====")
    print("1. Review forecast models for SKUs with high bias")
    print("2. Investigate data quality for SKUs with high anomaly rates")
    print("3. For SKUs where Direction Accuracy < 0.6, consider trend-focused models")
    print("4. Monitor distribution stability (Sliding_JSD) for early warning of concept drift")
    print("5. For longer forecast horizons, use wider confidence intervals")
    print("6. Consider different models for short-term vs. long-term forecasting")

    print(f"\nDetailed reports and visualizations saved to {args.output}/")


if __name__ == "__main__":
    main()
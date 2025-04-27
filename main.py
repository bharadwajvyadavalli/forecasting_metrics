"""
Simplified Forecast Metrics Framework

This script provides a simplified entry point for analyzing forecast accuracy
through multiple metric dimensions.

Usage:
    python main.py [--data=PATH_TO_DATA] [--output=output]
"""
import warnings
warnings.filterwarnings('ignore')

import os
import argparse
import pandas as pd

import metrics_calculator
from data_generator import generate_data
import lag_based_predictions as lag

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Forecast Metrics Analysis')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to forecast data CSV (if not provided, sample data will be generated in output folder)')
    parser.add_argument('--output', type=str, default='output',
                        help='Directory to save output (default: output)')
    parser.add_argument('--generate', action='store_true',
                        help='Generate new sample data (even if existing data file is specified)')
    return parser.parse_args()


def main():
    """Main function to run the forecast metrics analysis."""
    # Parse arguments
    args = parse_arguments()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Determine data file path
    if args.data is None:
        # If no data file specified, use sample_data.csv in the output folder
        data_path = os.path.join(args.output, "sample_data.csv")
    else:
        data_path = args.data

    # Generate sample data if requested, if no data file exists, or if default path is used
    if args.generate or not os.path.exists(data_path) or args.data is None:
        print(f"Generating sample forecast data to {data_path}...")
        data = generate_data()
        data.to_csv(data_path, index=False)
        print(f"Sample data saved to {data_path}")

    # Initialize metrics calculator
    print(f"\nAnalyzing forecast data from {data_path}...")
    calculator = metrics_calculator.MetricsCalculator(data_path)

    # Compute metrics and thresholds
    metrics_df = calculator.compute_metrics()
    sku_thresholds, global_thresholds = calculator.compute_thresholds()

    # Generate reports
    calculator.generate_report(args.output)

    # Calculate lag-based metrics
    print("\nAnalyzing forecast accuracy by forecast horizon...")
    lag_metrics = lag.compute_metrics_by_lag(calculator.data)
    lag_metrics_by_sku = lag.compute_metrics_by_lag_and_sku(calculator.data, timeline_output_file=f"{args.output}/sku_prediction_timeline.csv")

    # Save lag-based metrics
    lag_metrics.to_csv(f"{args.output}/lag_metrics.csv", index=False)
    lag_metrics_by_sku.to_csv(f"{args.output}/lag_metrics_by_sku.csv", index=False)

    print(f"\nAll reports and data saved to {args.output}/")


if __name__ == "__main__":
    main()
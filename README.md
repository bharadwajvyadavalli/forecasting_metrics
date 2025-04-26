# Forecast Metrics Framework

A simplified toolkit for evaluating forecast accuracy across multiple business-relevant dimensions.

## Overview

This framework provides a suite of metrics for assessing forecast quality beyond traditional error measures like MAPE and MAE. It organizes metrics by their business purpose and provides clear explanations of their relevance to decision-making.

## Key Features

- Comprehensive metrics covering bias, anomalies, direction, distribution, and calibration
- Business impact assessment for each metric
- Automated threshold calculation and performance evaluation
- Visualization tools for metric interpretation
- Simple, easy-to-understand structure

## Metrics Categories

### 1. Bias Metrics
Detect systematic over/under-prediction patterns that affect inventory levels and resource allocation efficiency.

- **Mean Bias**: Average deviation between predictions and actuals
- **Tracking Signal**: Ratio of cumulative error to mean absolute error
- **Residual Counts**: Balance of positive vs negative residuals

### 2. Anomaly Metrics
Identify outliers and unusual patterns that may require special attention or intervention.

- **Data Anomaly Rate**: Frequency of anomalies in actual values
- **Residual Anomaly Rate**: Frequency of anomalies in forecast errors

### 3. Directional Metrics
Evaluate trend and directional accuracy, critical for strategic planning and capacity decisions.

- **Direction Accuracy**: Percentage of correctly predicted up/down movements
- **Turning Point F1**: Accuracy in detecting trend reversals

### 4. Distribution Metrics
Assess distributional shifts and stability to identify when historical patterns become less relevant.

- **Sliding JSD**: Jensen-Shannon divergence between sequential time windows

### 5. Calibration Metrics
Evaluate probabilistic forecast accuracy, essential for risk assessment and uncertainty quantification.

- **CRPS**: Continuous Ranked Probability Score for probabilistic forecasts

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/forecast-metrics-framework.git
cd forecast-metrics-framework

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from metrics_calculator import MetricsCalculator

# Initialize calculator with your forecast data
calculator = MetricsCalculator('your_forecast_data.csv')

# Compute metrics and thresholds
metrics_df = calculator.compute_metrics()
sku_thresholds, global_thresholds = calculator.compute_thresholds()

# Generate reports and visualizations
calculator.generate_report('output')
calculator.visualize_key_metrics('output')
```

### Command Line Usage

```bash
# Run the main script with default settings
python main.py

# Generate new sample data
python main.py --generate

# Use custom data file and output directory
python main.py --data=your_data.csv --output=output_folder
```

## Data Format

The input CSV file should contain the following columns:
- `SKU`: Product/item identifier
- `Actual_Month`: Month of the actual value (YYYY-MM format)
- `Actual_Value`: Observed value for the period
- `Prediction_Month`: Month for which the prediction was made
- `Prediction_Value`: Predicted value
- `Prediction_Actual`: Actual value for the prediction month

## Project Structure

```
├── forecast_metrics.py        # Core metrics definitions
├── metrics_calculator.py      # Main calculator class
├── data_generator.py          # Synthetic data generation
├── visualization.py           # Visualization functions
├── main.py                    # Command-line interface
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Example Outputs

The framework generates:

### 1. CSV Reports
- `metrics_results.csv`: Raw metrics for each SKU and time period
- `sku_thresholds.csv`: Performance thresholds for each SKU and metric
- `global_thresholds.csv`: Global performance thresholds
- `sku_performance_summary.csv`: Ranking of SKUs by performance

### 2. Visualizations
- Metric distributions
- SKU performance comparisons
- Anomaly rate analysis
- Performance heatmap
- Threshold charts
- Business impact visualization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

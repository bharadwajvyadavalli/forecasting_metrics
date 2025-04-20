import pandas as pd
import numpy as np

# 1. Load the metrics output file
# Assumes 'sample_data_outputs.csv' is in the working directory

df = pd.read_csv('sample_data_outputs.csv')

# 2. Identify metric columns (exclude SKU and Actual_Month)
metrics = [c for c in df.columns if c not in ['SKU', 'Actual_Month']]

# 3. Define performance metrics (higher is better)
perf_metrics = ['Direction_Accuracy', 'Turning_Point_F1', 'PICP (%)']
# All other metrics are error metrics (lower is better)
error_metrics = [m for m in metrics if m not in perf_metrics]

# 4. Compute SKU-level thresholds (using absolute for signed error metrics)
# Metrics where sign matters (threshold on absolute value)
symmetric_error_metrics = ['Mean_Bias','Tracking_Signal','Cumulative_Bias']
sku_rows = []
for sku, sku_df in df.groupby('SKU'):
    # Error-type metrics: lower |value| is better
    for m in error_metrics:
        series = sku_df[m].dropna()
        if m in symmetric_error_metrics:
            series = series.abs()
        q25, q75 = series.quantile([0.25, 0.75])
        sku_rows.append({
            'SKU': sku,
            'Metric': m,
            'Green': round(q25, 3),
            'Yellow': round(q75, 3),
            'Red_Condition': f'> {q75:.3f}'
        })
    # Performance-type metrics: higher=better
    for m in perf_metrics:
        data = df[m].dropna()
        q25, q75 = data.quantile([0.25, 0.75])
        sku_rows.append({
            'SKU': sku,
            'Metric': m,
            'Green': round(q75, 3),
            'Yellow': round(q25, 3),
            'Red_Condition': f'< {q25:.3f}'
        })
sku_thresh_df = pd.DataFrame(sku_rows)
sku_thresh_df.to_csv('sku_thresholds.csv', index=False)

# 5. Compute global thresholds (using absolute for signed error metrics)
global_rows = []
for m in error_metrics:
    series = df[m].dropna()
    if m in symmetric_error_metrics:
        series = series.abs()
    q25, q75 = series.quantile([0.25, 0.75])
    global_rows.append({
        'Metric': m,
        'Green': round(q25, 3),
        'Yellow': round(q75, 3),
        'Red_Condition': f'> {q75:.3f}'
    })
for m in perf_metrics:
    data = df[m].dropna()
    q25, q75 = data.quantile([0.25, 0.75])
    global_rows.append({
        'Metric': m,
        'Green': round(q75, 3),
        'Yellow': round(q25, 3),
        'Red_Condition': f'< {q25:.3f}'
    })
global_thresh_df = pd.DataFrame(global_rows)
global_thresh_df.to_csv('global_summary.csv', index=False) #(across all SKUs)
global_rows = []
# Error metrics
for m in error_metrics:
    data = df[m].dropna()
    q25, q75 = data.quantile([0.25, 0.75])
    global_rows.append({
        'Metric': m,
        'Green': round(q25, 3),
        'Yellow': round(q75, 3),
        'Red_Condition': f'> {q75:.3f}'
    })
# Performance metrics
for m in perf_metrics:
    data = df[m].dropna()
    q25, q75 = data.quantile([0.25, 0.75])
    global_rows.append({
        'Metric': m,
        'Green': round(q75, 3),
        'Yellow': round(q25, 3),
        'Red_Condition': f'< {q25:.3f}'
    })
global_thresh_df = pd.DataFrame(global_rows)
global_thresh_df.to_csv('global_summary.csv', index=False)

print('Saved sku_thresholds.csv and global_summary.csv')

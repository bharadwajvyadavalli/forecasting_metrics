from typing import Callable

from forecast_metrics_definitions import *


if __name__ == '__main__':
    # 1. Load the sample data
    df = pd.read_csv('sample_data.csv', parse_dates=['Actual_Month', 'Prediction_Month'])

    # 2. Compute metrics per SKU & Actual_Month
    results = []
    for (sku, actual_month), grp in df.groupby(['SKU', 'Actual_Month']):
        y_true = grp['Prediction_Actual'].to_numpy()
        y_pred = grp['Prediction_Value'].to_numpy()
        base_actuals = grp['Actual_Value'].to_numpy()

        metrics = {
            'SKU': sku,
            'Actual_Month': actual_month,
            # 1. Systematic Bias (updated)
            'Mean_Bias': compute_mean_bias(y_true, y_pred),
            #'Tracking_Signal': compute_tracking_signal(y_true, y_pred),
            #'Residual Counts': compute_residual_counts(y_true, y_pred),
            #'Area_Under_Sparsification': compute_area_under_sparsification(y_true, y_pred),
            # 2. Outlier & Anomaly
            'Data_Anomaly_Rate': compute_data_anomaly_rate(base_actuals),
            'Residual_Anomaly_Rate': compute_residual_anomaly_rate(y_true, y_pred),
            #'Mean_Anomaly_Magnitude': compute_mean_anomaly_magnitude(y_true, y_pred),
            #'Time_to_Detect': compute_time_to_detect(y_true, y_pred),
            #'Persistence': compute_persistence(y_true, y_pred),
            # 3. Direction & Velocity
            'Direction_Accuracy': compute_direction_accuracy(y_true, y_pred),
            #'Velocity_Error': compute_velocity_error(y_true, y_pred),
            #'Acceleration_Error': compute_acceleration_error(y_true, y_pred),
            #'Turning_Point_F1': compute_turning_point_f1(y_true, y_pred),
            #'Trend_Change_Delay': compute_trend_change_delay(y_true, y_pred),
            # 4. Probabilistic Calibration
            #'PICP (%)': compute_picp(y_true, y_pred),
            #'Interval_Width': compute_interval_width(y_true, y_pred),
            #'Interval_Score': compute_interval_score(y_true, y_pred),
            #'Winkler_Score': compute_winkler_score(y_true, y_pred),
            'CRPS': compute_crps(y_true, y_pred),
            # 5. Distributional Drift & Stability
            'Sliding_JSD': compute_sliding_jsd(y_true, y_pred)
            #'PSI': compute_psi(y_true, y_pred),
            #'KL_Divergence': compute_kl_divergence(y_true, y_pred),
            #'Rolling_Error_Var': compute_rolling_error_variance(y_true, y_pred),
            #'MMD': compute_mmd(y_true, y_pred),
            # 6. Model Robustness & Change‑Sensitivity
            #'Feature_Drift_Rate': compute_feature_drift_rate(y_true, y_pred)
        }
        results.append(metrics)

    # 3. Build DataFrame and round numeric columns to 2 decimals
    output_df = pd.DataFrame(results)
    numeric_cols = output_df.select_dtypes(include=[np.number]).columns
    output_df[numeric_cols] = output_df[numeric_cols].round(2)

    # 4. Save to CSV
    output_df.to_csv('sample_data_outputs.csv', index=False)

    print("Metrics computed and written to sample_data_outputs.csv")
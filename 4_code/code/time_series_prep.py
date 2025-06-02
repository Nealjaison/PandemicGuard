import pandas as pd
import numpy as np

def create_lag_features(df, column, lags=[1, 3, 7]):
    """
    Create lag features for a time series column.
    lags: list of integers representing lag days
    """
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df

def create_rolling_features(df, column, windows=[3, 7, 14]):
    """
    Create rolling mean features for a time series column.
    windows: list of integers representing rolling window sizes
    """
    for window in windows:
        df[f"{column}_roll_mean_{window}"] = df[column].rolling(window).mean()
    return df

def preprocess_time_series(df, date_column, target_column):
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(by=date_column)

    df = create_lag_features(df, target_column)
    df = create_rolling_features(df, target_column)

    # Fill NA values created by lagging/rolling with zeros or forward fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)

    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Time Series Preprocessing for PandemicGuard")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Output CSV file with time series features")
    parser.add_argument('--date_column', type=str, required=True, help="Name of the date column")
    parser.add_argument('--target_column', type=str, required=True, help="Name of the target time series column")

    args = parser.parse_args()

    data = pd.read_csv(args.input)
    processed = preprocess_time_series(data, args.date_column, args.target_column)
    processed.to_csv(args.output, index=False)
    print(f"Preprocessed time series data saved to {args.output}")

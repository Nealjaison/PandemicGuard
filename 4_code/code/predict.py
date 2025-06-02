import pandas as pd
import joblib
import argparse

def predict(input_csv, model_path, encoders_path, scaler_path, output_csv):
    print(f"Loading new data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Loading preprocessing models...")
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # Encode categorical features
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].fillna('Unknown'))

    # Normalize numerical features
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[num_cols] = scaler.transform(df[num_cols])

    print("Predicting...")
    predictions = model.predict(df)

    df['prediction'] = predictions
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using PandemicGuard model.")
    parser.add_argument('--input', type=str, required=True, help="Path to new data CSV")
    parser.add_argument('--model', type=str, required=True, help="Path to trained model file")
    parser.add_argument('--encoders', type=str, required=True, help="Path to label encoders file")
    parser.add_argument('--scaler', type=str, required=True, help="Path to scaler file")
    parser.add_argument('--output', type=str, required=True, help="Path to save predictions CSV")

    args = parser.parse_args()
    predict(args.input, args.model, args.encoders, args.scaler, args.output)

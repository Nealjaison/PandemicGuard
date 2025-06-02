import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import os
import joblib

def load_data(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file type: use .csv or .json")

def clean_data(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())

    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df

def encode_features(df):
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders

def normalize_features(df):
    df = df.copy()
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler

def preprocess_pipeline(input_file, output_file, save_dir='./models'):
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading data from {input_file}...")
    df = load_data(input_file)

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding categorical features...")
    df, encoders = encode_features(df)

    print("Normalizing numerical features...")
    df, scaler = normalize_features(df)

    print(f"Saving processed data to {output_file}...")
    df.to_csv(output_file, index=False)

    joblib.dump(encoders, os.path.join(save_dir, "label_encoders.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))

    print("Preprocessing complete. Models saved to:", save_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess raw pandemic data.")
    parser.add_argument('--input', type=str, required=True, help="Path to raw input file (.csv or .json)")
    parser.add_argument('--output', type=str, required=True, help="Path to save cleaned output CSV")
    parser.add_argument('--models', type=str, default='./models', help="Directory to save encoders/scaler")

    args = parser.parse_args()
    preprocess_pipeline(args.input, args.output, args.models)
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import argparse
import os

def train(input_csv, model_output_dir='./models'):
    print(f"Loading data from {input_csv}...")
    data = pd.read_csv(input_csv)

    # Example: Assuming last column is label, rest are features
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, 'trained_model.pkl')
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PandemicGuard model.")
    parser.add_argument('--input', type=str, required=True, help="Path to preprocessed CSV data")
    parser.add_argument('--output', type=str, default='./models', help="Directory to save trained model")

    args = parser.parse_args()
    train(args.input, args.output)

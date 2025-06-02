import pandas as pd
import joblib
from sklearn.metrics import classification_report
import argparse
import os

def evaluate(input_csv, model_path):
    print(f"Loading data from {input_csv}...")
    data = pd.read_csv(input_csv)

    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print("Predicting...")
    y_pred = model.predict(X)

    report = classification_report(y_true, y_pred)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PandemicGuard model.")
    parser.add_argument('--input', type=str, required=True, help="Path to preprocessed CSV data")
    parser.add_argument('--model', type=str, required=True, help="Path to trained model file")

    args = parser.parse_args()
    evaluate(args.input, args.model)

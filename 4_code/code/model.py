
# model.py - Core AI model for PandemicGuard

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class PandemicGuardModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        return self.data

    def preprocess(self):
        # Placeholder: Customize based on dataset
        self.data.dropna(inplace=True)
        X = self.data.drop(columns=['target'])
        y = self.data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_test)
        return classification_report(self.y_test, y_pred, output_dict=True)

    def predict(self, new_data):
        return self.model.predict(new_data)

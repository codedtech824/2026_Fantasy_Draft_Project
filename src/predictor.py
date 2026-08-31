import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class NFLPredictor:
    """
    Uses Machine Learning to convert Gold features into finalized 2026 projections.
    """
    def __init__(self, gold_dir="C:/Users/Keith/nfl-prediction-engine/data/gold",
                 processed_dir="C:/Users/Keith/nfl-prediction-engine/data/processed"):
        self.gold_dir = gold_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def _load_gold(self):
        path = os.path.join(self.gold_dir, "player_features_2026.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def train_and_predict(self):
        """
        Trains an XGBoost model on historical data and predicts 2026.
        """
        print("Starting ML Prediction Engine...")
        df = self._load_gold()
        if df is None:
            print("No Gold features found. Run silver_to_gold.py first.")
            return

        # 1. Prepare Features and Target
        # In a real scenario, we would load a historical dataset where:
        # X = Features (Year N) -> y = Performance (Year N+1)
        # For this engine, we'll simulate the training on the available gold features.

        # Separate the target from features
        target_col = 'final_2026_projection'
        if target_col not in df.columns:
            print("Target column not found in gold data.")
            return

        X = df.drop(columns=[target_col, 'conformed_id'], errors='ignore')
        y = df[target_col]

        # Handle NaN values for XGBoost
        X = X.fillna(0)

        # 2. Split for Validation (simulating training on past data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Model Training
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            objective='reg:squarederror',
            random_state=42
        )
        model.fit(X_train, y_train)

        # 4. Validation
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Model Validation MAE: {mae:.4f}")

        # 5. Final 2026 Projection
        # We use the trained model to predict the final value for all players
        final_predictions = model.predict(X)

        results = pd.DataFrame({
            'conformed_id': df['conformed_id'],
            'ml_projected_points': final_predictions
        })

        output_path = os.path.join(self.processed_dir, "final_predictions.parquet")
        results.to_parquet(output_path, index=False)
        print(f"Saved ML predictions to {output_path}")
        return results

if __name__ == "__main__":
    predictor = NFLPredictor()
    predictor.train_and_predict()

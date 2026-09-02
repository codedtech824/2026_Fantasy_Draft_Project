# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - ML Prediction
# MAGIC Trains an XGBoost model on gold features and generates final 2026 fantasy point projections. Logs metrics and model artifact to MLflow.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Load gold features
gold = pd.read_parquet("/tmp/nfl-prediction-engine/data/gold/player_features_2026.parquet")

target_col = "final_2026_projection"
if target_col not in gold.columns:
    raise ValueError("Target column not found. Run 03_silver_to_gold first.")

X = gold.drop(columns=[target_col, "conformed_id"], errors="ignore").fillna(0)
y = gold[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("/nfl-prediction-engine")

with mlflow.start_run(run_name="xgboost_2026"):
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5,
                         objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("mae", mae)
    mlflow.xgboost.log_model(model, "xgboost_model")

    print(f"Model trained. Validation MAE: {mae:.4f}")
    print("MLflow run logged.")

# COMMAND ----------

# Generate final predictions for all players and save
import pandas as pd, os

final_preds = model.predict(X)
results = pd.DataFrame({
    "conformed_id": gold["conformed_id"],
    "ml_projected_points": final_preds
})

out = "/tmp/nfl-prediction-engine/data/processed/final_predictions.parquet"
results.to_parquet(out, index=False)
print(f"Saved {len(results)} predictions to {out}")
print("\nTop 10 projected players:")
print(results.nlargest(10, "ml_projected_points").to_string(index=False))

# COMMAND ----------

# Register as a Databricks table so it's queryable with %sql
spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(results).write.mode("overwrite").saveAsTable("nfl_prediction_engine.final_predictions")
print("Table saved: nfl_prediction_engine.final_predictions")

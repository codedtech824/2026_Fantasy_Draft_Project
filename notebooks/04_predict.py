# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 - ML Prediction
# MAGIC Trains an XGBoost model on gold features and generates final 2026 fantasy point projections.
# MAGIC Logs metrics and the model artifact to MLflow.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

from src.predictor import NFLPredictor

results = NFLPredictor().train_and_predict()

# COMMAND ----------

print(f"Saved {len(results)} predictions")
print("\nTop 10 projected players:")
print(results.nlargest(10, "ml_projected_points").to_string(index=False))

# COMMAND ----------

# Register as a Databricks table so it's queryable with %sql
spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(results).write.mode("overwrite").saveAsTable("nfl_prediction_engine.final_predictions")
print("Table saved: nfl_prediction_engine.final_predictions")

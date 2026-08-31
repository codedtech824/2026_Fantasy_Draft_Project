# Databricks notebook source

# COMMAND ----------

%pip install xgboost scikit-learn pyarrow requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Pipeline — Full End-to-End
# MAGIC Runs all 5 stages in sequence: Bronze → Silver → Gold → ML Prediction → Draft Board.
# MAGIC
# MAGIC **Run this notebook to execute the complete pipeline in one shot.**

# COMMAND ----------

import sys, os, time

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Create all required directories in this session's /tmp
for folder in [
    "/tmp/nfl-prediction-engine/data/bronze/nfl_stats",
    "/tmp/nfl-prediction-engine/data/bronze/injuries",
    "/tmp/nfl-prediction-engine/data/silver",
    "/tmp/nfl-prediction-engine/data/gold",
    "/tmp/nfl-prediction-engine/data/processed",
]:
    os.makedirs(folder, exist_ok=True)

print(f"Repo root: {repo_root}")
print("Directories ready. Starting full pipeline...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1 — Bronze Ingestion

# COMMAND ----------

from src.fetcher import NFLDataFetcher

t0 = time.time()
NFLDataFetcher().run_all()
print(f"Bronze complete ({time.time()-t0:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2 — Bronze to Silver

# COMMAND ----------

from src.bronze_to_silver import BronzeToSilver

t0 = time.time()
BronzeToSilver().run_pipeline()
print(f"Silver complete ({time.time()-t0:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 3 — Silver to Gold

# COMMAND ----------

from src.silver_to_gold import SilverToGold

t0 = time.time()
SilverToGold().run_pipeline()
print(f"Gold complete ({time.time()-t0:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 4 — ML Prediction

# COMMAND ----------

import mlflow, mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

gold = pd.read_parquet("/tmp/nfl-prediction-engine/data/gold/player_features_2026.parquet")
X = gold.drop(columns=["final_2026_projection", "conformed_id"], errors="ignore").fillna(0)
y = gold["final_2026_projection"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("/nfl-prediction-engine")
t0 = time.time()
with mlflow.start_run(run_name="xgboost_pipeline_run"):
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5,
                         objective="reg:squarederror", random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_metric("mae", mae)
    mlflow.xgboost.log_model(model, "xgboost_model")

results = pd.DataFrame({"conformed_id": gold["conformed_id"], "ml_projected_points": model.predict(X)})
results.to_parquet("/tmp/nfl-prediction-engine/data/processed/final_predictions.parquet", index=False)
print(f"Prediction complete. MAE: {mae:.4f} ({time.time()-t0:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 5 — Draft Board

# COMMAND ----------

from src.drafter import NFLDrafter

t0 = time.time()
NFLDrafter().calculate_vbd()
print(f"Draft board complete ({time.time()-t0:.1f}s)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Output

# COMMAND ----------

import pandas as pd

board = pd.read_parquet("/tmp/nfl-prediction-engine/data/processed/final_draft_board.parquet")
cols = ["player_name", "position", "ml_projected_points", "vbd_score", "final_draft_value"]
available = [c for c in cols if c in board.columns]

print(f"Pipeline complete. Draft board: {len(board)} players\n")
print("=== 2026 FINAL DRAFT BOARD (Top 30) ===")
print(board[available].head(30).to_string(index=False))

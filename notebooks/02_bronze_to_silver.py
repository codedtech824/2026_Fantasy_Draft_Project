# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Bronze to Silver
# MAGIC Cleans raw JSON from the bronze layer and writes conformed Parquet tables to `/tmp/nfl-prediction-engine/data/silver/`.

# COMMAND ----------

# Wire up sys.path
import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

# Run the bronze -> silver transformation
from src.bronze_to_silver import BronzeToSilver

transformer = BronzeToSilver()
transformer.run_pipeline()

# COMMAND ----------

# Verify — list parquet files written to silver
import os

silver_root = "/tmp/nfl-prediction-engine/data/silver"
print("Silver layer contents:")
for f in os.listdir(silver_root):
    full = os.path.join(silver_root, f)
    size_kb = os.path.getsize(full) / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")

# COMMAND ----------

# Spot-check: preview game_logs schema and row count
import pandas as pd

game_logs = pd.read_parquet("/tmp/nfl-prediction-engine/data/silver/game_logs.parquet")
print(f"game_logs: {len(game_logs)} rows, {len(game_logs.columns)} columns")
print("\nColumns:", list(game_logs.columns))
print("\nSample:")
print(game_logs.head(3).to_string())

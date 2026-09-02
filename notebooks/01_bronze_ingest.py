# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Bronze Ingestion
# MAGIC Fetches raw data into `/tmp/nfl-prediction-engine/data/bronze/`:
# MAGIC player stats and the 2026 schedule from nfldata.org, the current 2026
# MAGIC roster from nflverse, market data/injuries from LeagueLogs, and
# MAGIC advanced metrics from Muffed.ai.

# COMMAND ----------

# Wire up sys.path and confirm src is importable
import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

# Run the bronze ingestion
from src.fetcher import NFLDataFetcher

fetcher = NFLDataFetcher(base_dir="/tmp/nfl-prediction-engine/data/bronze")
fetcher.run_all()

# COMMAND ----------

# Verify — list files saved to bronze
import os

bronze_root = "/tmp/nfl-prediction-engine/data/bronze"
print("Bronze layer contents:")
for root, dirs, files in os.walk(bronze_root):
    for f in files:
        full = os.path.join(root, f)
        size_kb = os.path.getsize(full) / 1024
        rel = os.path.relpath(full, bronze_root)
        print(f"  {rel}  ({size_kb:.1f} KB)")

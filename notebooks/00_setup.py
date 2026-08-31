# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 - Setup
# MAGIC Run this once after attaching to a new cluster. Installs libraries, wires up sys.path, and creates the local data folders.

# COMMAND ----------

%pip install xgboost scikit-learn pyarrow requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# Add repo root to sys.path so `from src.xxx import ...` works
import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"sys.path root: {repo_root}")

# COMMAND ----------

# Create local medallion folder structure
# DBFS root is disabled on this workspace — /tmp persists for the cluster session
import os

folders = [
    "/tmp/nfl-prediction-engine/data/bronze/nfl_stats",
    "/tmp/nfl-prediction-engine/data/bronze/injuries",
    "/tmp/nfl-prediction-engine/data/silver",
    "/tmp/nfl-prediction-engine/data/gold",
    "/tmp/nfl-prediction-engine/data/processed",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

# COMMAND ----------

# Verify folder structure
import os
print("Local data structure:")
for root, dirs, _ in os.walk("/tmp/nfl-prediction-engine"):
    level = root.replace("/tmp/nfl-prediction-engine", "").count(os.sep)
    print("  " * level + os.path.basename(root) + "/")

# COMMAND ----------

# Smoke-test: confirm all src modules import cleanly
from src.fetcher import NFLDataFetcher
from src.bronze_to_silver import BronzeToSilver
from src.silver_to_gold import SilverToGold
from src.predictor import NFLPredictor
from src.drafter import NFLDrafter

print("All modules imported successfully. Setup complete.")

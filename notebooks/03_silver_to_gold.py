# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Silver to Gold
# MAGIC Applies EWMA time-decay, SOS normalization, injury risk, and matchup modifiers to produce the final predictive feature set.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

from src.silver_to_gold import SilverToGold

transformer = SilverToGold(
    silver_dir="/tmp/nfl-prediction-engine/data/silver",
    gold_dir="/tmp/nfl-prediction-engine/data/gold",
)
transformer.run_pipeline()

# COMMAND ----------

# Verify — list parquet files written to gold
import os

gold_root = "/tmp/nfl-prediction-engine/data/gold"
print("Gold layer contents:")
for f in os.listdir(gold_root):
    full = os.path.join(gold_root, f)
    size_kb = os.path.getsize(full) / 1024
    print(f"  {f}  ({size_kb:.1f} KB)")

# COMMAND ----------

# Spot-check: preview gold features
import pandas as pd

gold = pd.read_parquet("/tmp/nfl-prediction-engine/data/gold/player_features_2026.parquet")
print(f"Gold features: {len(gold)} players, {len(gold.columns)} columns")
print("\nColumns:", list(gold.columns))
print("\nTop 5 by final_2026_projection:")
print(gold.nlargest(5, "final_2026_projection")[["conformed_id", "final_2026_projection", "injury_multiplier", "schedule_modifier", "bye_week"]].to_string())

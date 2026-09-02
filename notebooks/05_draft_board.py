# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 - Draft Board
# MAGIC Applies Value Based Drafting (VBD) and scarcity multipliers to produce the final ranked 2026 fantasy draft board.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"Repo root: {repo_root}")

# COMMAND ----------

from src.drafter import NFLDrafter

drafter = NFLDrafter(
    processed_dir="/tmp/nfl-prediction-engine/data/processed",
    silver_dir="/tmp/nfl-prediction-engine/data/silver",
)
draft_board = drafter.calculate_vbd()

# COMMAND ----------

# Display the full draft board if available
import pandas as pd, os

board_path = "/tmp/nfl-prediction-engine/data/processed/final_draft_board.parquet"
if os.path.exists(board_path):
    board = pd.read_parquet(board_path)
    print(f"Draft board: {len(board)} players\n")

    cols = ["player_name", "position", "ml_projected_points", "vbd_score", "final_draft_value"]
    available = [c for c in cols if c in board.columns]

    print("=== 2026 FINAL DRAFT BOARD (Top 30) ===")
    print(board[available].head(30).to_string(index=False))

    # Register as a Databricks table so it's queryable with %sql
    spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
    spark.createDataFrame(board).write.mode("overwrite").saveAsTable("nfl_prediction_engine.draft_board_2026")
    print("\nTable saved: nfl_prediction_engine.draft_board_2026")
else:
    print("Draft board not found. Check that 04_predict ran successfully.")

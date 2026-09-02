# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 - Predict Real NFL Games
# MAGIC A different prediction target from `08`/`09` -- those predict *fantasy*
# MAGIC team matchups (drafted rosters against each other). This predicts real
# MAGIC NFL games (Chiefs @ Broncos, etc.) and grades those predictions against
# MAGIC actual results, same spirit, separate tables.
# MAGIC
# MAGIC It's a simple baseline, not a real point-spread model: predicted score
# MAGIC for each side is that team's rostered QB/RB/WR/TE `ml_projected_points`
# MAGIC summed (offense), minus the opponent's D/ST `ml_projected_points`
# MAGIC (defense) -- both numbers already sitting in the draft board, no new
# MAGIC modeling. Validated against the real, fully-resolved 2025 season: 59.9%
# MAGIC accuracy (163/272 games) -- meaningfully better than a coin flip, well
# MAGIC below what a real spread model would do, exactly what you'd expect from
# MAGIC a deliberately simple baseline.
# MAGIC
# MAGIC Only depends on the draft board (`run_pipeline.py`), not the fantasy
# MAGIC rosters from `07` -- these are real NFL team predictions, unrelated to
# MAGIC who drafted which player onto a fantasy roster.
# MAGIC
# MAGIC Safe to re-run any time: predictions are static for the season (same
# MAGIC board every time), the graded/accuracy table always recomputes from
# MAGIC whatever's actually been played so far.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import pandas as pd
from src.season_simulator import fetch_nfl_games, predict_nfl_games, grade_nfl_predictions

_PROC = "/tmp/nfl-prediction-engine/data/processed"
_BOARD_TABLE = "nfl_prediction_engine.draft_board_2026"
SEASON = 2026

if spark.catalog.tableExists(_BOARD_TABLE):
    board = spark.table(_BOARD_TABLE).toPandas()
    print(f"Loaded board from Delta table {_BOARD_TABLE}")
elif os.path.exists(f"{_PROC}/final_draft_board.parquet"):
    board = pd.read_parquet(f"{_PROC}/final_draft_board.parquet")
    print(f"Delta table {_BOARD_TABLE} not found -- loaded from /tmp instead")
else:
    raise FileNotFoundError(
        f"Neither the Delta table {_BOARD_TABLE} nor {_PROC}/final_draft_board.parquet exist. "
        "Run run_pipeline.py (or 01-05) at least once first."
    )

# COMMAND ----------

games = fetch_nfl_games(SEASON)
print(f"Fetched {len(games)} games for {SEASON} (played or not)")

predictions = predict_nfl_games(board, games)
print(predictions.head(10).to_string(index=False))

# COMMAND ----------

graded = grade_nfl_predictions(predictions, games)

if graded.empty:
    print(f"No {SEASON} games completed yet -- nothing to grade. Predictions table still saved below.")
else:
    accuracy = graded["correct"].mean()
    print(f"Accuracy so far: {accuracy:.1%} ({graded['correct'].sum()}/{len(graded)})")
    print(graded.to_string(index=False))

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(predictions).write.mode("overwrite").saveAsTable("nfl_prediction_engine.nfl_game_predictions_2026")
print("Table saved: nfl_prediction_engine.nfl_game_predictions_2026 (every game, played or not)")

if not graded.empty:
    spark.createDataFrame(graded).write.mode("overwrite").saveAsTable("nfl_prediction_engine.nfl_game_predictions_graded_2026")
    print("Table saved: nfl_prediction_engine.nfl_game_predictions_graded_2026 (completed games only, with accuracy)")

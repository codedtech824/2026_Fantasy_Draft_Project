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
# MAGIC modeling. Any player nflverse's weekly injury report has marked "Out"
# MAGIC for that week is excluded from their team's offense sum first -- a
# MAGIC benched starter shouldn't count. Validated against the real,
# MAGIC fully-resolved 2025 season: 59.9% accuracy without the injury filter,
# MAGIC 61.4% with it (163 -> 167 of 272 games) -- meaningfully better than a
# MAGIC coin flip, well below what a real spread model would do, exactly what
# MAGIC you'd expect from a deliberately simple baseline. A home-field-advantage
# MAGIC adjustment was also tested and dropped -- across bonus sizes from 2% to
# MAGIC 10% it never beat the baseline by more than 1 game out of 272, noise
# MAGIC rather than signal at this model's scale.
# MAGIC
# MAGIC Only depends on the draft board (`run_pipeline.py`), not the fantasy
# MAGIC rosters from `07` -- these are real NFL team predictions, unrelated to
# MAGIC who drafted which player onto a fantasy roster.
# MAGIC
# MAGIC Safe to re-run any time: predictions are static for the season (same
# MAGIC board every time), the graded/accuracy table always recomputes from
# MAGIC whatever's actually been played so far.
# MAGIC
# MAGIC `predicted_home_score`/`predicted_away_score` are the raw offense-minus-
# MAGIC defense proxy (useful for picking a winner, not realistic point totals --
# MAGIC they run into the hundreds). `*_realistic_score` rescales that into a
# MAGIC real NFL point range using the min/max across every prediction (so
# MAGIC relative team strength is preserved), then breaks it into a plausible
# MAGIC touchdowns/PATs/2pt-conversions/field-goals combination -- one specific
# MAGIC way a real game could reach that total, not the only way.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import pandas as pd
from src.season_simulator import (
    fetch_nfl_games, fetch_weekly_injuries, predict_nfl_games,
    grade_nfl_predictions, add_realistic_scores,
)

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

injuries_by_week = fetch_weekly_injuries(SEASON)
if injuries_by_week:
    total_out = sum(len(ids) for ids in injuries_by_week.values())
    print(f"Fetched {total_out} 'Out' designations across {len(injuries_by_week)} week(s) -- excluding them from that week's offense sum")
else:
    print(f"No {SEASON} injury reports published yet -- predicting with full rosters until they are")

predictions = predict_nfl_games(board, games, injuries_by_week)
predictions = add_realistic_scores(predictions)

score_cols = ["home_team", "away_team", "predicted_winner",
              "home_realistic_score", "home_touchdowns", "home_extra_points",
              "home_two_point_conversions", "home_field_goals",
              "away_realistic_score", "away_touchdowns", "away_extra_points",
              "away_two_point_conversions", "away_field_goals"]
print(predictions[score_cols].head(10).to_string(index=False))

# COMMAND ----------

graded = grade_nfl_predictions(predictions, games)

if graded.empty:
    print(f"No {SEASON} games completed yet -- nothing to grade. Predictions table still saved below.")
else:
    accuracy = graded["correct"].mean()
    print(f"Accuracy so far: {accuracy:.1%} ({graded['correct'].sum()}/{len(graded)})")
    compare_cols = ["home_team", "away_team", "home_realistic_score", "away_realistic_score",
                    "actual_home_score", "actual_away_score", "predicted_winner", "actual_winner", "correct"]
    print(graded[compare_cols].to_string(index=False))

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
# overwriteSchema: the realistic-score/TD-PAT-FG columns are new -- without
# this, Delta rejects the write with DELTA_METADATA_MISMATCH against the
# table's older schema instead of evolving it.
spark.createDataFrame(predictions).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("nfl_prediction_engine.nfl_game_predictions_2026")
print("Table saved: nfl_prediction_engine.nfl_game_predictions_2026 (every game, played or not)")

if not graded.empty:
    spark.createDataFrame(graded).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("nfl_prediction_engine.nfl_game_predictions_graded_2026")
    print("Table saved: nfl_prediction_engine.nfl_game_predictions_graded_2026 (completed games only, with accuracy)")

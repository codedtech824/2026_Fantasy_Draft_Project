# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 - Predict Real NFL Games
# MAGIC A different prediction target from `08`/`09` -- those predict *fantasy*
# MAGIC team matchups (drafted rosters against each other). This predicts real
# MAGIC NFL games (Chiefs @ Broncos, etc.) and grades those predictions against
# MAGIC actual results, same spirit, separate tables.
# MAGIC
# MAGIC Whenever nfldata.org has a real Vegas line for a game (`spread_line`/
# MAGIC `total_line` -- posted as kickoff approaches, not months in advance),
# MAGIC that's what's used: predicted_home_score/predicted_away_score come
# MAGIC straight from the line (home = (total+spread)/2, away = (total-spread)/2),
# MAGIC and predicted_winner is whichever side the spread favors. Validated
# MAGIC against the real, fully-resolved 2025 season at 65.1% accuracy
# MAGIC (177/272) -- a meaningful jump over building our own model, for data
# MAGIC that was already sitting in the API response.
# MAGIC
# MAGIC For any game without a line posted yet, it falls back to the original
# MAGIC simple baseline: that team's rostered QB/RB/WR/TE `ml_projected_points`
# MAGIC summed (offense) minus the opponent's D/ST `ml_projected_points`
# MAGIC (defense), with nflverse's weekly injury report excluding anyone
# MAGIC marked "Out" from the offense sum, plus a flat home-field-advantage
# MAGIC boost (`HOME_FIELD_BONUS_PCT` in season_simulator.py -- backtested as
# MAGIC roughly accuracy-neutral on its own, kept for real-world modeling
# MAGIC completeness). That fallback alone scores 61.8% on the 2025 backtest --
# MAGIC still meaningfully better than a coin flip, just not as good as using
# MAGIC the real line once one exists. Each row's `prediction_source` column
# MAGIC says which method produced it ("vegas" or "roster").
# MAGIC
# MAGIC Only depends on the draft board (`run_pipeline.py`), not the fantasy
# MAGIC rosters from `07` -- these are real NFL team predictions, unrelated to
# MAGIC who drafted which player onto a fantasy roster.
# MAGIC
# MAGIC Safe to re-run any time: as the season progresses, more games flip from
# MAGIC "roster" to "vegas" as their lines get posted, and vegas-sourced rows
# MAGIC pick up line movement on each re-run since the line is refetched fresh
# MAGIC every time -- so, unlike before, predictions aren't fully static across
# MAGIC reruns for games without a final result yet. The graded/accuracy table
# MAGIC always recomputes from whatever's actually been played so far.
# MAGIC
# MAGIC `*_realistic_score` is the real point value directly for vegas-sourced
# MAGIC rows; for roster-sourced rows it's the offense-minus-defense proxy
# MAGIC rescaled into a realistic NFL point range (min/max across that subset
# MAGIC only, so it isn't distorted by vegas rows' different scale). Either
# MAGIC way it's then broken into a plausible touchdowns/PATs/2pt-conversions/
# MAGIC field-goals combination -- one specific way a real game could reach
# MAGIC that total, not the only way.

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

source_counts = predictions["prediction_source"].value_counts().to_dict()
print(f"Prediction source: {source_counts.get('vegas', 0)} game(s) using the real Vegas line, {source_counts.get('roster', 0)} using the roster-based fallback (no line posted yet)")

score_cols = ["home_team", "away_team", "predicted_winner", "prediction_source",
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
    print("By prediction source:")
    print(graded.groupby("prediction_source")["correct"].agg(["mean", "sum", "count"]))
    compare_cols = ["home_team", "away_team", "prediction_source", "home_realistic_score", "away_realistic_score",
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

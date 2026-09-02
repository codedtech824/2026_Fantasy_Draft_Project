# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 08 - Simulate a Full Season (2025 Backtest)
# MAGIC The 2026 season just started, so there's no real season's worth of data
# MAGIC to show yet. This backtests the rosters from `07_draft_league` against
# MAGIC the **real, completed 2025 NFL season** instead: same 13 teams and rosters,
# MAGIC scored week-by-week against what those players actually did in 2025.
# MAGIC
# MAGIC It's a backtest, not a reconstruction of an actual 2025 fantasy league --
# MAGIC these teams and this roster only exist because of the 2026 draft in `07`.
# MAGIC
# MAGIC **Known data gaps** (see `src/season_simulator.py` for detail):
# MAGIC - D/ST weekly scoring is points-allowed only -- sacks/INTs/fumbles/def TDs
# MAGIC   aren't available at week granularity from this API, only as a season
# MAGIC   total (already used correctly elsewhere, e.g. the draft board).
# MAGIC - K weekly scoring is a flat season-average per game, not real week-to-week
# MAGIC   variance -- no week-level kicker data exists from this API at all.
# MAGIC - A handful of skill-position players (~6%) have gaps in some weeks --
# MAGIC   the underlying NGS stats endpoints only cover players above a usage
# MAGIC   threshold each week, not the full league.
# MAGIC - Any 2026 rookie on a roster contributes 0 all season -- they weren't in
# MAGIC   the NFL yet in 2025.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import json
import pandas as pd
from src.league import FANTASY_TEAMS
from src.season_simulator import (
    generate_round_robin, fetch_weekly_offense, fetch_weekly_dst_points_allowed,
    kicker_weekly_average, simulate_season,
)

_SILVER = "/tmp/nfl-prediction-engine/data/silver"
_PROC = "/tmp/nfl-prediction-engine/data/processed"
SEASON = 2025

with open(f"{_PROC}/fantasy_rosters_2026.json") as f:
    rosters = json.load(f)

schedule = generate_round_robin(FANTASY_TEAMS, weeks=13)
print(f"Schedule: {len(FANTASY_TEAMS)} teams, {len(set(r['week'] for r in schedule))} weeks (round-robin, one bye/week)")

# COMMAND ----------

print(f"Fetching real {SEASON} weekly stats...")
weekly_offense = fetch_weekly_offense(SEASON)
weekly_dst = fetch_weekly_dst_points_allowed(SEASON)

game_logs = pd.read_parquet(f"{_SILVER}/game_logs.parquet")
kicker_avg = kicker_weekly_average(game_logs, SEASON)
print(f"Offense rows: {len(weekly_offense)}, D/ST rows: {len(weekly_dst)}, kickers with an average: {len(kicker_avg)}")

# COMMAND ----------

season = simulate_season(rosters, weekly_offense, weekly_dst, kicker_avg, schedule)
print(f"Season simulated: {len(season)} team-weeks")
print(season[season.team == FANTASY_TEAMS[0]].to_string(index=False))

# COMMAND ----------

played = season[season["result"] != "BYE"]
standings = (
    played.groupby("team")
    .agg(
        wins=("result", lambda s: (s == "W").sum()),
        losses=("result", lambda s: (s == "L").sum()),
        ties=("result", lambda s: (s == "T").sum()),
        points_for=("points_for", "sum"),
        points_against=("points_against", "sum"),
    )
    .reset_index()
)
standings[["points_for", "points_against"]] = standings[["points_for", "points_against"]].round(2)
standings = standings.sort_values(["wins", "points_for"], ascending=[False, False]).reset_index(drop=True)
standings.index += 1

print("=== Standings ===")
print(standings.to_string())

# COMMAND ----------

season.to_parquet(f"{_PROC}/fantasy_season_2025.parquet", index=False)
standings.to_parquet(f"{_PROC}/fantasy_standings_2025.parquet", index=False)

spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(season).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_season_2025")
spark.createDataFrame(standings).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_standings_2025")

print("Tables saved:")
print("  nfl_prediction_engine.fantasy_season_2025    (one row per team per week -- for a weekly-trend chart)")
print("  nfl_prediction_engine.fantasy_standings_2025  (one row per team -- for a standings table)")

# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 09 - Update the Live 2026 Season & Standings
# MAGIC The real-life counterpart to `08`'s 2025 backtest: scores the same 13
# MAGIC drafted rosters against whatever 2026 games have **actually completed**
# MAGIC so far, using the same round-robin schedule (13 weeks, one bye/week).
# MAGIC
# MAGIC Safe to re-run at any point in the season -- each run recomputes the
# MAGIC season-to-date from scratch (cheap: ~13 teams x however many weeks have
# MAGIC happened) and overwrites the tables, so there's no drift or partial-update
# MAGIC risk if a past week's stats get corrected. Run this anytime after
# MAGIC `06_update_2026_stats.py` -- weekly, e.g. -- to keep the standings current
# MAGIC through the season.
# MAGIC
# MAGIC Before Week 1 has finished, this will find zero completed weeks and skip
# MAGIC writing (same "nothing yet" pattern as `06`) rather than write an empty
# MAGIC table -- run it again once games start finishing.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import pandas as pd
from src.league import FANTASY_TEAMS
from src.season_simulator import (
    generate_round_robin, completed_weeks_for_season, fetch_weekly_offense,
    fetch_weekly_dst_scores, fetch_weekly_kicker_points, simulate_season,
)

_PROC = "/tmp/nfl-prediction-engine/data/processed"
_ROSTERS_TABLE = "nfl_prediction_engine.fantasy_rosters_2026"
SEASON = 2026
FANTASY_WEEKS = 13

if spark.catalog.tableExists(_ROSTERS_TABLE):
    rosters_df = spark.table(_ROSTERS_TABLE).toPandas()
    rosters = {}
    for team, team_df in rosters_df.groupby("fantasy_team"):
        rosters[team] = {
            slot: slot_df.drop(columns=["fantasy_team", "slot"]).to_dict("records")
            for slot, slot_df in team_df.groupby("slot")
        }
    print(f"Loaded rosters from Delta table {_ROSTERS_TABLE}")
else:
    raise FileNotFoundError(f"{_ROSTERS_TABLE} doesn't exist yet. Run 07_draft_league.py first.")

full_schedule = generate_round_robin(FANTASY_TEAMS, weeks=FANTASY_WEEKS)

# COMMAND ----------

played_weeks = completed_weeks_for_season(SEASON, max_week=FANTASY_WEEKS)
print(f"Completed 2026 weeks so far (of {FANTASY_WEEKS}): {sorted(played_weeks) if played_weeks else 'none yet'}")

if not played_weeks:
    print("Nothing to update -- no 2026 games have finished yet. Re-run this after Week 1 completes.")
else:
    schedule = [row for row in full_schedule if row["week"] in played_weeks]

    weekly_offense = fetch_weekly_offense(SEASON)
    weekly_dst = fetch_weekly_dst_scores(SEASON)
    weekly_kicker = fetch_weekly_kicker_points(SEASON)
    print(f"Offense rows: {len(weekly_offense)}, D/ST rows: {len(weekly_dst)}, K rows: {len(weekly_kicker)}")

    season = simulate_season(rosters, weekly_offense, weekly_dst, weekly_kicker, schedule)
    print(f"\nSeason-to-date: {len(season)} team-weeks across {len(played_weeks)} completed week(s)")
    print(season[season.team == FANTASY_TEAMS[0]].to_string(index=False))

# COMMAND ----------

if played_weeks:
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

    print("=== Standings (through week", max(played_weeks), ") ===")
    print(standings.to_string())

    # Delta tables first -- the durable output; /tmp copies are just
    # within-session convenience.
    spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
    spark.createDataFrame(season).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_season_2026")
    spark.createDataFrame(standings).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_standings_2026")

    os.makedirs(_PROC, exist_ok=True)
    season.to_parquet(f"{_PROC}/fantasy_season_2026.parquet", index=False)
    standings.to_parquet(f"{_PROC}/fantasy_standings_2026.parquet", index=False)

    print("\nTables saved:")
    print("  nfl_prediction_engine.fantasy_season_2026    (one row per team per completed week)")
    print("  nfl_prediction_engine.fantasy_standings_2026  (one row per team -- season-to-date)")

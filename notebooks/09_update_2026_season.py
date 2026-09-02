# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 09 - Update the Live 2026 Season, Predictions & Championship Tracker
# MAGIC The real-life counterpart to `08`'s 2025 backtest: scores the same 13
# MAGIC drafted rosters against whatever 2026 games have **actually completed**
# MAGIC so far, using the same round-robin schedule (13 weeks, one bye/week).
# MAGIC Also predicts every matchup and grades those predictions against real
# MAGIC results as they come in, and tracks how the model's predicted
# MAGIC champion changes week over week as the standings evolve.
# MAGIC
# MAGIC Safe to re-run at any point in the season -- season/standings/predictions
# MAGIC always recompute from scratch and overwrite, so there's no drift or
# MAGIC partial-update risk if a past week's stats get corrected. Run this
# MAGIC anytime after `06_update_2026_stats.py` -- weekly, e.g. -- to keep
# MAGIC everything current through the season.
# MAGIC
# MAGIC Before Week 1 has finished, this will find zero completed weeks and skip
# MAGIC writing (same "nothing yet" pattern as `06`) rather than write empty
# MAGIC tables -- run it again once games start finishing.

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
    compute_standings, predict_matchup_outcomes, compare_predictions_to_actual,
    simulate_playoff_bracket, actual_scores_for_weeks,
)

_PROC = "/tmp/nfl-prediction-engine/data/processed"
_ROSTERS_TABLE = "nfl_prediction_engine.fantasy_rosters_2026"
_TRACKER_TABLE = "nfl_prediction_engine.fantasy_championship_tracker_2026"
SEASON = 2026
FANTASY_WEEKS = 13
PLAYOFF_WEEKS = [14, 15, 16]

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
print(f"Completed 2026 regular-season weeks so far (of {FANTASY_WEEKS}): {sorted(played_weeks) if played_weeks else 'none yet'}")

if not played_weeks:
    print("Nothing to update -- no 2026 games have finished yet. Re-run this after Week 1 completes.")
else:
    schedule = [row for row in full_schedule if row["week"] in played_weeks]

    weekly_offense = fetch_weekly_offense(SEASON)
    weekly_dst = fetch_weekly_dst_scores(SEASON)
    weekly_kicker = fetch_weekly_kicker_points(SEASON)
    print(f"Offense rows: {len(weekly_offense)}, D/ST rows: {len(weekly_dst)}, K rows: {len(weekly_kicker)}")

    season = simulate_season(rosters, weekly_offense, weekly_dst, weekly_kicker, schedule)
    standings = compute_standings(season)
    print(f"\nSeason-to-date: {len(season)} team-weeks across {len(played_weeks)} completed week(s)")
    print(season[season.team == FANTASY_TEAMS[0]].to_string(index=False))
    print("\n=== Standings (through week", max(played_weeks), ") ===")
    print(standings.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Predicted vs. actual -- regular season matchups played so far

# COMMAND ----------

if played_weeks:
    predictions = predict_matchup_outcomes(rosters, full_schedule)
    graded = compare_predictions_to_actual(predictions, season)
    accuracy = graded["correct"].mean()
    mae = graded["points_for_error"].abs().mean()
    print(f"Prediction accuracy so far: {accuracy:.1%} ({graded['correct'].sum()}/{len(graded)})")
    print(f"Mean absolute points error: {mae:.2f}")
    print(graded.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Championship tracker -- "if the playoffs started today"
# MAGIC Logged once per completed week (a Delta MERGE keyed on `as_of_week`, so
# MAGIC re-running the same week updates that row instead of duplicating it):
# MAGIC the current top-6 seeds from the season-to-date standings, and the
# MAGIC model's predicted champion for that seeding. Watch this table grow one
# MAGIC row per week to see how the predicted champion shifts as results come
# MAGIC in -- it naturally stops changing once the regular season (week 13)
# MAGIC ends and seeding locks.

# COMMAND ----------

if played_weeks:
    as_of_week = max(played_weeks)
    seeds = standings["team"].head(6).tolist()
    _, predicted_champion = simulate_playoff_bracket(seeds, rosters)

    tracker_row = pd.DataFrame([{
        "as_of_week": as_of_week,
        "seed1": seeds[0], "seed2": seeds[1], "seed3": seeds[2],
        "seed4": seeds[3], "seed5": seeds[4], "seed6": seeds[5],
        "predicted_champion": predicted_champion,
    }])
    print(f"As of week {as_of_week}, predicted champion: {predicted_champion}")

    spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
    spark_tracker_row = spark.createDataFrame(tracker_row)
    if spark.catalog.tableExists(_TRACKER_TABLE):
        from delta.tables import DeltaTable
        target = DeltaTable.forName(spark, _TRACKER_TABLE)
        (
            target.alias("t")
            .merge(spark_tracker_row.alias("s"), "t.as_of_week = s.as_of_week")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        spark_tracker_row.write.format("delta").saveAsTable(_TRACKER_TABLE)

    print("\nFull tracker history so far:")
    display(spark.table(_TRACKER_TABLE).orderBy("as_of_week"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Playoff bracket -- once real playoff-week results exist
# MAGIC Only runs once the regular season (all 13 weeks) is actually complete
# MAGIC and at least one of weeks 14-16 has real results -- before that there's
# MAGIC nothing to compare the tracker's predictions against yet.

# COMMAND ----------

if played_weeks == set(range(1, FANTASY_WEEKS + 1)):
    playoff_weeks_done = completed_weeks_for_season(SEASON, max_week=max(PLAYOFF_WEEKS)) & set(PLAYOFF_WEEKS)
    if not playoff_weeks_done:
        print("Regular season is complete, but no playoff weeks (14-16) have finished yet.")
    else:
        print(f"Playoff weeks completed so far: {sorted(playoff_weeks_done)}")
        final_seeds = standings["team"].head(6).tolist()

        predicted_games, predicted_champion = simulate_playoff_bracket(final_seeds, rosters)
        real_playoff_scores = actual_scores_for_weeks(rosters, sorted(playoff_weeks_done), weekly_offense, weekly_dst, weekly_kicker)
        actual_games, actual_champion = simulate_playoff_bracket(final_seeds, rosters, actual_scores=real_playoff_scores)

        # Kept as two independent scenarios, not merged game-by-game -- once
        # the brackets disagree on who wins a round, the *matchup* itself in
        # the next round differs between them (a different team advances).
        for g in predicted_games:
            g["scenario"] = "predicted"
        for g in actual_games:
            g["scenario"] = "actual"
        bracket = pd.DataFrame(predicted_games + actual_games)[
            ["scenario", "round", "week", "team_a", "score_a", "team_b", "score_b", "winner"]
        ]

        print(f"\nPredicted champion: {predicted_champion}")
        print(f"Actual champion so far: {actual_champion} (based on {len(playoff_weeks_done)}/3 playoff weeks played)")
        print(bracket.to_string(index=False))

        spark.createDataFrame(bracket).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_bracket_2026")
        print("\nTable saved: nfl_prediction_engine.fantasy_bracket_2026")
else:
    print("Regular season isn't complete yet -- no bracket to compute.")

# COMMAND ----------

if played_weeks:
    spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
    spark.createDataFrame(season).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_season_2026")
    spark.createDataFrame(standings).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_standings_2026")
    spark.createDataFrame(graded).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_predictions_2026")

    os.makedirs(_PROC, exist_ok=True)
    season.to_parquet(f"{_PROC}/fantasy_season_2026.parquet", index=False)
    standings.to_parquet(f"{_PROC}/fantasy_standings_2026.parquet", index=False)

    print("Tables saved:")
    print("  nfl_prediction_engine.fantasy_season_2026               (one row per team per completed week)")
    print("  nfl_prediction_engine.fantasy_standings_2026             (one row per team -- season-to-date)")
    print("  nfl_prediction_engine.fantasy_predictions_2026           (predicted vs actual per matchup, with accuracy)")
    print("  nfl_prediction_engine.fantasy_championship_tracker_2026  (predicted champion, one row per week)")

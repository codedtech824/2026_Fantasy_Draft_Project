# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 - Update 2026 Stats
# MAGIC Fetches stats for every *completed* 2026-season game -- weekly player
# MAGIC stats from nflverse, completed-game/schedule metadata from nfldata.org --
# MAGIC and upserts them into `nfl_prediction_engine.stats_2026`.
# MAGIC
# MAGIC Safe to re-run any time during the season -- rows are keyed by
# MAGIC `game_id + player + position`, so newly completed games are added and
# MAGIC already-seen games are refreshed in place, with no duplicate rows.
# MAGIC
# MAGIC fantasy_points is computed from raw yards/TDs/receptions using the
# MAGIC scoring system in `src/scoring.py` (defaults to FantasyData's standard
# MAGIC PPR values), not read from the API's own precomputed field -- so
# MAGIC changing `scoring` below and rerunning will rescore every stored row,
# MAGIC not just new ones.
# MAGIC
# MAGIC Run `00_setup` first if this is a fresh cluster session (needs `requests`
# MAGIC installed and `src` on the path).

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

from src.stats_updater import Stats2026Updater
from src.scoring import FULL_PPR, HALF_PPR, STANDARD, ScoringSystem

# Swap in HALF_PPR, STANDARD, or a custom ScoringSystem(...) to change how
# fantasy_points is calculated -- see src/scoring.py for every point value.
updater = Stats2026Updater(spark, scoring=FULL_PPR)
summary = updater.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview the table

# COMMAND ----------

if summary["table_rows"]:
    display(spark.table("nfl_prediction_engine.stats_2026").orderBy("week", "game_id", "player_name"))
else:
    print("Table not created yet -- no completed 2026 games with stats found.")

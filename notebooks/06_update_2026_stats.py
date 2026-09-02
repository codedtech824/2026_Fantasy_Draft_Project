# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 - Update 2026 Stats
# MAGIC Fetches stats for every *completed* 2026-season game from nfldata.org and
# MAGIC upserts them into `nfl_prediction_engine.stats_2026`.
# MAGIC
# MAGIC Safe to re-run any time during the season -- rows are keyed by
# MAGIC `game_id + player + position`, so newly completed games are added and
# MAGIC already-seen games are refreshed in place, with no duplicate rows.
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

updater = Stats2026Updater(spark)
summary = updater.run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview the table

# COMMAND ----------

if summary["table_rows"]:
    display(spark.table("nfl_prediction_engine.stats_2026").orderBy("week", "game_id", "player_name"))
else:
    print("Table not created yet -- no completed 2026 games with stats found.")

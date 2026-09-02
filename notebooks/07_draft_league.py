# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 - Draft the Fantasy League
# MAGIC Runs a 13-team, 15-round snake draft (9 starters -- QB/RB/RB/WR/WR/TE/FLEX/K/DST
# MAGIC -- plus 6 bench) against the current `final_draft_board.parquet`, using
# MAGIC `src/league.py`. The draft order is randomized once with a fixed seed so
# MAGIC re-running this notebook reproduces the same league instead of reshuffling
# MAGIC it every time -- change or remove `seed` below if you want a fresh draft.

# COMMAND ----------

import sys, os

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# COMMAND ----------

import json
import pandas as pd
from src.league import FANTASY_TEAMS, generate_snake_order, run_draft, TOTAL_ROSTER_SIZE

_PROC = "/tmp/nfl-prediction-engine/data/processed"

board = pd.read_parquet(f"{_PROC}/final_draft_board.parquet")
board = board[board["position"].isin(["QB", "RB", "WR", "TE", "DST", "K"])].copy()
board = board.sort_values("final_draft_value", ascending=False).reset_index(drop=True)
print(f"Draftable pool: {len(board)} ({board['position'].value_counts().to_dict()})")

snake_order = generate_snake_order(FANTASY_TEAMS, rounds=TOTAL_ROSTER_SIZE, seed=2026)
rosters, pick_log = run_draft(board, snake_order)

print(f"\nDrafted {len(pick_log)} / {len(snake_order)} picks")
for team, r in rosters.items():
    filled = sum(len(v) for v in r.values())
    print(f"  {team}: {filled}/{TOTAL_ROSTER_SIZE}")

# COMMAND ----------

# Save locally (used by 08) and as a Delta table (queryable pick history)
os.makedirs(_PROC, exist_ok=True)
with open(f"{_PROC}/fantasy_rosters_2026.json", "w") as f:
    json.dump(rosters, f, indent=2)

picks_df = pd.DataFrame(pick_log)
picks_df.to_parquet(f"{_PROC}/draft_picks_2026.parquet", index=False)

spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(picks_df).write.mode("overwrite").saveAsTable("nfl_prediction_engine.draft_picks_2026")
print("\nTable saved: nfl_prediction_engine.draft_picks_2026")

# COMMAND ----------

print("=== Round 1 ===")
print(picks_df[picks_df["round"] == 1][["pick", "team", "player", "position", "nfl_team"]].to_string(index=False))

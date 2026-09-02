# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 - Draft the Fantasy League
# MAGIC Runs a 13-team, 15-round snake draft (9 starters -- QB/RB/RB/WR/WR/TE/FLEX/K/DST
# MAGIC -- plus 6 bench) against the current draft board, using `src/league.py`.
# MAGIC The draft order is randomized once with a fixed seed so re-running this
# MAGIC notebook reproduces the same league instead of reshuffling it every time --
# MAGIC change or remove `seed` below if you want a fresh draft.
# MAGIC
# MAGIC Reads the board from the `nfl_prediction_engine.draft_board_2026` Delta
# MAGIC table (falling back to the `/tmp` parquet file only if that table doesn't
# MAGIC exist yet) -- **not** straight from `/tmp`, since `/tmp` only persists for
# MAGIC the current cluster session and gets wiped on a restart/detach, while the
# MAGIC Delta table survives. Needs `run_pipeline.py` (or `01`-`05`) to have run
# MAGIC at least once, ever, on this workspace -- doesn't need to be this session.

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
_BOARD_TABLE = "nfl_prediction_engine.draft_board_2026"

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

# Save locally (handy within this session) and as Delta tables (durable --
# 08 reads the roster table, not the /tmp JSON, so it survives a cluster
# restart/detach between running 07 and 08).
os.makedirs(_PROC, exist_ok=True)
with open(f"{_PROC}/fantasy_rosters_2026.json", "w") as f:
    json.dump(rosters, f, indent=2)

picks_df = pd.DataFrame(pick_log)
picks_df.to_parquet(f"{_PROC}/draft_picks_2026.parquet", index=False)

# Flatten the nested {team: {slot: [player, ...]}} roster dict into one row
# per roster spot -- a real table, not a JSON blob, so it's directly queryable.
# "fantasy_team" (not "team") to avoid colliding with the player's own "team"
# key, which is their *NFL* team.
roster_rows = [
    {"fantasy_team": team, "slot": slot, **player}
    for team, slots in rosters.items()
    for slot, players in slots.items()
    for player in players
]
rosters_df = pd.DataFrame(roster_rows)

spark.sql("CREATE DATABASE IF NOT EXISTS nfl_prediction_engine")
spark.createDataFrame(picks_df).write.mode("overwrite").saveAsTable("nfl_prediction_engine.draft_picks_2026")
spark.createDataFrame(rosters_df).write.mode("overwrite").saveAsTable("nfl_prediction_engine.fantasy_rosters_2026")
print("\nTables saved:")
print("  nfl_prediction_engine.draft_picks_2026")
print("  nfl_prediction_engine.fantasy_rosters_2026")

# COMMAND ----------

print("=== Round 1 ===")
print(picks_df[picks_df["round"] == 1][["pick", "team", "player", "position", "nfl_team"]].to_string(index=False))

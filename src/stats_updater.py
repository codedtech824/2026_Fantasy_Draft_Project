import time
from datetime import datetime, timezone

import pandas as pd
import requests

from src.scoring import FULL_PPR

_EMPTY_RAW = dict(
    pass_yards=0, pass_tds=0, interceptions=0,
    rush_yards=0, rush_tds=0,
    rec_yards=0, rec_tds=0, receptions=0,
)


def _abbreviate_name(full_name):
    """'Christian McCaffrey' -> 'C.McCaffrey' -- the "F.Last" format used
    everywhere else in this pipeline (game_logs, players_master, the draft
    board). Handles multi-word surnames ('Amon-Ra St. Brown' -> 'A.St.
    Brown') by taking the first token as the given name and joining
    everything else as the surname."""
    parts = full_name.strip().split(" ", 1)
    if len(parts) < 2 or not parts[0]:
        return full_name.strip()
    return f"{parts[0][0]}.{parts[1]}"


class Stats2026Updater:
    """
    Fetches player stats for completed 2026-season games from nfldata.org and
    upserts them into a single Delta table (default: nfl_prediction_engine.stats_2026).

    Safe to re-run at any point during the season: rows are keyed by
    game_id + conformed_id (player+position), so already-seen games are
    refreshed in place (e.g. post-game stat corrections, or a rescored table
    after changing `scoring`) and only newly completed games add new rows --
    nothing is ever duplicated.

    fantasy_points is computed from raw yards/TDs/receptions via a
    ScoringSystem (src/scoring.py), not read from the API's own precomputed
    field, so the point values are ours to configure -- e.g.
    Stats2026Updater(spark, scoring=HALF_PPR).
    """

    BASE_URL = "https://api.nfldata.org/v1"
    STANDARD_ENDPOINTS = ["/stats/passing", "/stats/receiving", "/stats/rushing"]
    NGS_ENDPOINTS = {
        "passing": "/stats/ngs/passing",
        "rushing": "/stats/ngs/rushing",
        "receiving": "/stats/ngs/receiving",
    }

    def __init__(self, spark, season=2026, scoring=None, db_name="nfl_prediction_engine", table="stats_2026"):
        self.spark = spark
        self.season = season
        self.scoring = scoring or FULL_PPR
        self.db_name = db_name
        self.table = table
        self.full_table = f"{db_name}.{table}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NFL-Fantasy-Pipeline/1.0"})

    # ---- games / completion --------------------------------------------

    def fetch_completed_games(self):
        """Games for self.season where both scores are populated (i.e. played)."""
        resp = self.session.get(
            f"{self.BASE_URL}/games", params={"season": self.season, "limit": 500}, timeout=30
        )
        resp.raise_for_status()
        games = resp.json().get("data", [])
        return [g for g in games if g.get("home_score") is not None and g.get("away_score") is not None]

    @staticmethod
    def _build_game_maps(completed_games):
        """(team, week) -> opponent / is_home / game_id / game_date, for completed games only."""
        opponent, is_home, game_id, game_date = {}, {}, {}, {}
        for g in completed_games:
            week, home, away = g.get("week"), g.get("home_team"), g.get("away_team")
            if not (week and home and away):
                continue
            gid, gdate = g.get("game_id"), g.get("gameday")
            opponent[(home, week)] = away
            opponent[(away, week)] = home
            is_home[(home, week)] = True
            is_home[(away, week)] = False
            game_id[(home, week)] = gid
            game_id[(away, week)] = gid
            game_date[(home, week)] = gdate
            game_date[(away, week)] = gdate
        return opponent, is_home, game_id, game_date

    # ---- raw fetch ---------------------------------------------------------

    def _fetch_paginated(self, endpoint):
        records, offset, limit = [], 0, 500
        while True:
            resp = self.session.get(
                f"{self.BASE_URL}{endpoint}",
                params={"season": self.season, "limit": limit, "offset": offset},
                timeout=60,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            page = data.get("data", [])
            records.extend(page)
            if len(records) >= data.get("total", 0) or len(page) < limit:
                break
            offset += limit
        return records

    # ---- per-source field extraction ---------------------------------------
    # The standard endpoints (/stats/passing, /stats/receiving, /stats/rushing)
    # are NOT three slices of the same category -- each returns a different,
    # partially-overlapping set of player-weeks (e.g. a pure passer with 0
    # carries never appears in /stats/rushing), but every returned row already
    # carries the player's FULL combined stat line (pass+rush+rec together, 0
    # where not applicable). So a player who appears in more than one of the
    # three lists (e.g. a rushing QB) shows the *same* complete line each
    # time -- we overwrite by key, we never sum across these three fetches.
    #
    # The NGS endpoints are the opposite: each one only carries that single
    # category's fields (verified against the live API), so a dual-threat
    # player's rushing and receiving lines arrive as two separate records
    # that must be added together to get their full game total.

    @staticmethod
    def _extract_standard(stat):
        return dict(
            pass_yards=stat.get("passing_yards") or 0,
            pass_tds=stat.get("passing_tds") or 0,
            interceptions=stat.get("interceptions") or 0,
            rush_yards=stat.get("rushing_yards") or 0,
            rush_tds=stat.get("rushing_tds") or 0,
            rec_yards=stat.get("receiving_yards") or 0,
            rec_tds=stat.get("receiving_tds") or 0,
            receptions=stat.get("receptions") or 0,
        )

    @staticmethod
    def _infer_position_from_standard(stat):
        """The standard endpoint has no position field. Infer one from which
        category the player was most active in -- position doesn't affect
        scoring, this is just for a readable table."""
        if (stat.get("attempts") or 0) > 0:
            return "QB"
        if (stat.get("carries") or 0) > 0 and (stat.get("carries") or 0) >= (stat.get("targets") or 0):
            return "RB"
        if (stat.get("targets") or 0) > 0:
            return "WR"
        return "UNK"

    @staticmethod
    def _extract_ngs(stat, category):
        raw = dict(_EMPTY_RAW)
        if category == "passing":
            raw["pass_yards"] = stat.get("pass_yards") or 0
            raw["pass_tds"] = stat.get("pass_touchdowns") or 0
            raw["interceptions"] = stat.get("interceptions") or 0
        elif category == "rushing":
            raw["rush_yards"] = stat.get("rush_yards") or 0
            raw["rush_tds"] = stat.get("rush_touchdowns") or 0
        elif category == "receiving":
            raw["rec_yards"] = stat.get("yards") or 0  # NOT 'rec_yards' -- verified against the live API
            raw["rec_tds"] = stat.get("rec_touchdowns") or 0
            raw["receptions"] = stat.get("receptions") or 0
        return raw

    # ---- combine -------------------------------------------------------------

    def _new_entry(self, gid, week, player_name, position, team, maps):
        opponent_map, home_away_map, _, game_date_map = maps
        return {
            "stat_key": f"{gid}_{player_name.strip().lower()}_{position.strip().lower()}",
            "game_id": gid,
            "season": self.season,
            "week": week,
            "game_date": game_date_map.get((team, week)),
            "player_name": player_name,
            "position": position,
            "team": team,
            "opponent": opponent_map.get((team, week)),
            "is_home": home_away_map.get((team, week)),
            "_raw": dict(_EMPTY_RAW),
        }

    def build_stats_dataframe(self):
        completed_games = self.fetch_completed_games()
        if not completed_games:
            print(f"No completed games yet for season {self.season}.")
            return pd.DataFrame()

        maps = self._build_game_maps(completed_games)
        opponent_map, home_away_map, game_id_map, game_date_map = maps
        print(f"{len(completed_games)} completed game(s) found for season {self.season}.")

        combined = {}

        standard_rows = []
        for endpoint in self.STANDARD_ENDPOINTS:
            standard_rows.extend(self._fetch_paginated(endpoint))
            time.sleep(0.3)

        if standard_rows:
            print(f"  standard: {len(standard_rows)} rows across passing/receiving/rushing listings")
            for stat in standard_rows:
                player_name, team, week = stat.get("player_name", ""), stat.get("recent_team", ""), stat.get("week")
                if not player_name or not team or week is None:
                    continue
                player_name = _abbreviate_name(player_name)
                gid = game_id_map.get((team, week))
                if gid is None:
                    continue
                position = self._infer_position_from_standard(stat)
                entry = combined.setdefault(
                    f"{gid}_{player_name.strip().lower()}_{position.strip().lower()}",
                    self._new_entry(gid, week, player_name, position, team, maps),
                )
                entry["_raw"] = self._extract_standard(stat)  # each row is already the full combined line
        else:
            for category, endpoint in self.NGS_ENDPOINTS.items():
                records = self._fetch_paginated(endpoint)
                print(f"  {category} (ngs): {len(records)} records")
                for stat in records:
                    first, last = stat.get("player_first_name"), stat.get("player_last_name")
                    if first and last:
                        player_name = f"{first[0]}.{last}"
                    else:
                        player_name = _abbreviate_name(stat.get("player_display_name", ""))
                    team = stat.get("team_abbr", "")
                    week = stat.get("week")
                    position = stat.get("player_position") or category[:3].upper()
                    if not player_name or not team or week is None:
                        continue
                    gid = game_id_map.get((team, week))
                    if gid is None:
                        continue
                    entry = combined.setdefault(
                        f"{gid}_{player_name.strip().lower()}_{position.strip().lower()}",
                        self._new_entry(gid, week, player_name, position, team, maps),
                    )
                    raw = self._extract_ngs(stat, category)
                    for k, v in raw.items():
                        entry["_raw"][k] += v
                time.sleep(0.3)

        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for entry in combined.values():
            raw = entry.pop("_raw")
            entry["fantasy_points"] = self.scoring.score(**raw)
            entry["updated_at"] = now
            rows.append(entry)

        return pd.DataFrame(rows)

    # ---- upsert ------------------------------------------------------------

    def upsert(self, pdf):
        if pdf.empty:
            print("Nothing to upsert.")
            return {"fetched": 0, "action": "none", "table_rows": None}

        self.spark.sql(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")
        spark_df = self.spark.createDataFrame(pdf)

        if self.spark.catalog.tableExists(self.full_table):
            from delta.tables import DeltaTable

            target = DeltaTable.forName(self.spark, self.full_table)
            (
                target.alias("t")
                .merge(spark_df.alias("s"), "t.stat_key = s.stat_key")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            action = "merged"
        else:
            spark_df.write.format("delta").saveAsTable(self.full_table)
            action = "created"

        table_rows = self.spark.table(self.full_table).count()
        return {"fetched": len(pdf), "action": action, "table_rows": table_rows}

    def run(self):
        print(f"Updating {self.full_table} for season {self.season} (scoring: {self.scoring})...")
        pdf = self.build_stats_dataframe()
        summary = self.upsert(pdf)
        if summary["table_rows"] is not None:
            print(
                f"Done. {summary['fetched']} row(s) fetched this run ({summary['action']}). "
                f"{self.full_table} now has {summary['table_rows']} total rows."
            )
        return summary

import time
from datetime import datetime, timezone

import pandas as pd
import requests


class Stats2026Updater:
    """
    Fetches player stats for completed 2026-season games from nfldata.org and
    upserts them into a single Delta table (default: nfl_prediction_engine.stats_2026).

    Safe to re-run at any point during the season: rows are keyed by
    game_id + conformed_id (player+position), so already-seen games are
    refreshed in place (e.g. post-game stat corrections) and only newly
    completed games add new rows -- nothing is ever duplicated.
    """

    BASE_URL = "https://api.nfldata.org/v1"
    POSITIONS = ["passing", "receiving", "rushing"]

    def __init__(self, spark, season=2026, db_name="nfl_prediction_engine", table="stats_2026"):
        self.spark = spark
        self.season = season
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

    # ---- stats -----------------------------------------------------------

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

    def fetch_stats_for_position(self, position):
        """Standard endpoint first; falls back to NGS, which is what's actually
        populated for the current season."""
        standard = {"passing": "/stats/passing", "receiving": "/stats/receiving", "rushing": "/stats/rushing"}
        ngs = {"passing": "/stats/ngs/passing", "receiving": "/stats/ngs/receiving", "rushing": "/stats/ngs/rushing"}

        records = self._fetch_paginated(standard[position])
        if records:
            return records, "standard"
        return self._fetch_paginated(ngs[position]), "ngs"

    @staticmethod
    def _fantasy_points(stat, position, source):
        """Full-PPR: pass yd/25, pass TD*4, INT*-2, rush/rec yd/10, rush/rec TD*6, rec*1.
        Matches this project's redraft-1qb-12t-ppr1 market data convention.
        NOTE: the NGS receiving endpoint's yardage field is 'yards', not 'rec_yards'."""
        if source == "standard" and stat.get("fantasy_points") is not None:
            return stat["fantasy_points"]

        fp = 0.0
        if position == "passing":
            fp += (stat.get("pass_yards") or 0) / 25
            fp += (stat.get("pass_touchdowns") or 0) * 4
            fp -= (stat.get("interceptions") or 0) * 2
        elif position == "rushing":
            fp += (stat.get("rush_yards") or 0) / 10
            fp += (stat.get("rush_touchdowns") or 0) * 6
        elif position == "receiving":
            fp += (stat.get("yards") or 0) / 10
            fp += (stat.get("rec_touchdowns") or 0) * 6
            fp += (stat.get("receptions") or 0) * 1.0
        return fp

    def build_stats_dataframe(self):
        completed_games = self.fetch_completed_games()
        if not completed_games:
            print(f"No completed games yet for season {self.season}.")
            return pd.DataFrame()

        opponent_map, home_away_map, game_id_map, game_date_map = self._build_game_maps(completed_games)
        print(f"{len(completed_games)} completed game(s) found for season {self.season}.")

        rows = []
        for position in self.POSITIONS:
            stats, source = self.fetch_stats_for_position(position)
            print(f"  {position}: {len(stats)} records ({source})")

            for stat in stats:
                if source == "ngs":
                    player_name = stat.get("player_display_name", "")
                    team = stat.get("team_abbr", "")
                    week = stat.get("week")
                    pos = stat.get("player_position") or position[:3].upper()
                else:
                    player_name = stat.get("player_name", "")
                    team = stat.get("recent_team", "")
                    week = stat.get("week")
                    pos = stat.get("position") or position[:3].upper()

                if not player_name or not team or week is None:
                    continue

                gid = game_id_map.get((team, week))
                if gid is None:
                    continue  # stat row for a game not in our completed set -- skip

                conformed_id = f"{player_name.strip().lower()}_{pos.strip().lower()}"
                rows.append({
                    "stat_key": f"{gid}_{conformed_id}",
                    "game_id": gid,
                    "season": self.season,
                    "week": week,
                    "game_date": game_date_map.get((team, week)),
                    "player_name": player_name,
                    "position": pos,
                    "team": team,
                    "opponent": opponent_map.get((team, week)),
                    "is_home": home_away_map.get((team, week)),
                    "fantasy_points": float(self._fantasy_points(stat, position, source)),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
            time.sleep(0.3)  # rate limiting between position endpoints

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.drop_duplicates(subset="stat_key", keep="last")
        return df

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
        print(f"Updating {self.full_table} for season {self.season}...")
        pdf = self.build_stats_dataframe()
        summary = self.upsert(pdf)
        if summary["table_rows"] is not None:
            print(
                f"Done. {summary['fetched']} row(s) fetched this run ({summary['action']}). "
                f"{self.full_table} now has {summary['table_rows']} total rows."
            )
        return summary

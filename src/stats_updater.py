import io
from datetime import datetime, timezone

import pandas as pd
import requests

from src.scoring import FULL_PPR

_TEAM_ALIASES = {"WSH": "WAS"}  # nfldata.org uses WSH, nflverse uses WAS


class Stats2026Updater:
    """
    Fetches player stats for completed 2026-season games and upserts them
    into a single Delta table (default: nfl_prediction_engine.stats_2026).

    Weekly stats come from nflverse's stats_player_week_{season}.csv --
    comprehensive (every rostered player, not a usage-threshold cutoff) and
    already in the "F.Last" name format used everywhere else in this
    pipeline. Completed-game/opponent/schedule metadata still comes from
    nfldata.org's /games, since nflverse's player-stats file doesn't carry
    final scores.

    Safe to re-run at any point during the season: rows are keyed by
    game_id + conformed_id (player+position), so already-seen games are
    refreshed in place (e.g. post-game stat corrections, or a rescored table
    after changing `scoring`) and only newly completed games add new rows --
    nothing is ever duplicated.

    fantasy_points is computed from raw yards/TDs/receptions via a
    ScoringSystem (src/scoring.py), not read from nflverse's own precomputed
    fantasy_points_ppr column, so the point values are ours to configure --
    e.g. Stats2026Updater(spark, scoring=HALF_PPR).
    """

    GAMES_URL = "https://api.nfldata.org/v1/games"
    NFLVERSE_WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv"
    OFFENSE_POSITIONS = ("QB", "RB", "WR", "TE")

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
        resp = self.session.get(self.GAMES_URL, params={"season": self.season, "limit": 500}, timeout=30)
        resp.raise_for_status()
        games = resp.json().get("data", [])
        return [g for g in games if g.get("home_score") is not None and g.get("away_score") is not None]

    @staticmethod
    def _build_game_maps(completed_games):
        """(team, week) -> opponent / is_home / game_id / game_date, for completed games only."""
        opponent, is_home, game_id, game_date = {}, {}, {}, {}
        for g in completed_games:
            week = g.get("week")
            home = _TEAM_ALIASES.get(g.get("home_team"), g.get("home_team"))
            away = _TEAM_ALIASES.get(g.get("away_team"), g.get("away_team"))
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

    # ---- weekly stats -----------------------------------------------------

    def fetch_weekly_stats(self):
        """
        Downloads nflverse's comprehensive weekly player-stats file for
        self.season. Returns None (not raises) if it doesn't exist yet --
        nflverse publishes this once a season's first games have been
        played, so early in a season (or off-season) there may be nothing
        there yet.
        """
        url = self.NFLVERSE_WEEKLY_URL.format(season=self.season)
        resp = self.session.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        df["team"] = df["team"].replace(_TEAM_ALIASES)
        return df

    def build_stats_dataframe(self):
        completed_games = self.fetch_completed_games()
        if not completed_games:
            print(f"No completed games yet for season {self.season}.")
            return pd.DataFrame()

        opponent_map, home_away_map, game_id_map, game_date_map = self._build_game_maps(completed_games)
        print(f"{len(completed_games)} completed game(s) found for season {self.season}.")

        weekly = self.fetch_weekly_stats()
        if weekly is None:
            print(f"nflverse hasn't published stats_player_week_{self.season}.csv yet.")
            return pd.DataFrame()

        offense = weekly[weekly["position"].isin(self.OFFENSE_POSITIONS)]
        print(f"  nflverse weekly stats: {len(offense)} offensive player-weeks")

        now = datetime.now(timezone.utc).isoformat()
        rows = []
        for stat in offense.to_dict("records"):
            player_name, team, week, position = stat.get("player_name"), stat.get("team"), stat.get("week"), stat.get("position")
            if not player_name or not team or week is None or pd.isna(week):
                continue
            week = int(week)
            gid = game_id_map.get((team, week))
            if gid is None:
                continue  # not among the games we've confirmed completed

            conformed_id = f"{player_name.strip().lower()}_{position.strip().lower()}"
            fantasy_points = self.scoring.score(
                pass_yards=stat.get("passing_yards") or 0,
                pass_tds=stat.get("passing_tds") or 0,
                interceptions=stat.get("interceptions") or 0,
                rush_yards=stat.get("rushing_yards") or 0,
                rush_tds=stat.get("rushing_tds") or 0,
                rec_yards=stat.get("receiving_yards") or 0,
                rec_tds=stat.get("receiving_tds") or 0,
                receptions=stat.get("receptions") or 0,
            )
            rows.append({
                "stat_key": f"{gid}_{conformed_id}",
                "game_id": gid,
                "season": self.season,
                "week": week,
                "game_date": game_date_map.get((team, week)),
                "player_name": player_name,
                "position": position,
                "team": team,
                "opponent": opponent_map.get((team, week)),
                "is_home": home_away_map.get((team, week)),
                "fantasy_points": float(fantasy_points),
                "updated_at": now,
            })

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
        print(f"Updating {self.full_table} for season {self.season} (scoring: {self.scoring})...")
        pdf = self.build_stats_dataframe()
        summary = self.upsert(pdf)
        if summary["table_rows"] is not None:
            print(
                f"Done. {summary['fetched']} row(s) fetched this run ({summary['action']}). "
                f"{self.full_table} now has {summary['table_rows']} total rows."
            )
        return summary

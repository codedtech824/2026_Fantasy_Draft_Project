import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class BronzeToSilver:
    """
    Transforms raw JSON data from the Bronze layer into cleaned, conformed
    Parquet tables in the Silver layer.
    """
    def __init__(self, bronze_dir=None, silver_dir=None):
        self.bronze_dir = bronze_dir or os.path.join(_PROJECT_ROOT, "data", "bronze")
        self.silver_dir = silver_dir or os.path.join(_PROJECT_ROOT, "data", "silver")
        os.makedirs(self.silver_dir, exist_ok=True)

    def _load_json_files(self, subfolder, pattern="*.json"):
        """Helper to load all JSON files matching a pattern from a bronze subfolder.
        Returns a list of tuples (filename, data_list).
        """
        path = os.path.join(self.bronze_dir, subfolder, pattern)
        files = glob.glob(path)
        all_data_with_files = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data_with_files.append((os.path.basename(file), data))
                    else:
                        all_data_with_files.append((os.path.basename(file), [data]))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file {file}")
        return all_data_with_files

    def process_stats(self):
        """
        Cleans raw NFL stats.
        Extracts season from filename and adds it as a column.
        """
        print("Processing Bronze Stats -> Silver...")

        # We ONLY want the leaders files for the game logs
        files_data = self._load_json_files("nfl_stats", "leaders_*.json")
        if not files_data:
            print("No stats leaders data found in Bronze. Skipping...")
            return None

        all_dfs = []
        for filename, data in files_data:
            # Extract season from filename (e.g., 'leaders_2022.json' -> '2022')
            try:
                season = filename.split('_')[1].split('.')[0]
            except:
                season = "Unknown"

            df = pd.DataFrame(data)
            df['season'] = season

            # Unnest paginated API response (e.g. NFLData.org returns {"data": [...], "total": ...})
            if 'data' in df.columns:
                unnested = []
                for _, row in df.iterrows():
                    if isinstance(row.get('data'), list):
                        for player in row['data']:
                            if isinstance(player, dict):
                                player.setdefault('season', row.get('season', season))
                                unnested.append(player)
                if unnested:
                    df = pd.DataFrame(unnested)

            all_dfs.append(df)

        df = pd.concat(all_dfs, ignore_index=True)

        # 1. Schema Enforcement
        numeric_cols = ['yards', 'tds', 'receptions', 'carries', 'targets', 'interceptions', 'epa']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # 2. Identity Resolution
        if 'player_name' in df.columns and 'position' in df.columns:
            df['conformed_id'] = df['player_name'].str.lower().str.strip() + "_" + df['position'].str.lower()
        elif 'playerId' in df.columns:
            df['conformed_id'] = df['playerId'].astype(str)
        else:
            df['conformed_id'] = df.index.astype(str)

        # 3. Deduplication - Handle unhashable types (lists)
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        df = df.drop_duplicates()

        # Save to Silver
        output_path = os.path.join(self.silver_dir, "game_logs.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved cleaned game logs to {output_path}")
        return df

    @staticmethod
    def _points_allowed_tier(points_per_game):
        """Standard D/ST points-allowed scoring tier."""
        if points_per_game is None:
            return 0
        if points_per_game <= 0:
            return 10
        if points_per_game <= 6:
            return 7
        if points_per_game <= 13:
            return 4
        if points_per_game <= 20:
            return 1
        if points_per_game <= 27:
            return 0
        if points_per_game <= 34:
            return -1
        return -4

    def process_dst(self):
        """
        Cleans raw team-defense data (dst_*.json) into Silver rows shaped
        like the offensive game logs, so they flow through the same
        EWMA/SOS/gold pipeline unchanged. fantasy_points is computed here
        (not in bronze) since points-allowed tiering is a scoring
        transformation, not raw data: sacks*1, INT*2, fumble rec*2,
        safety*2, def TD*6, plus the tiered points-allowed score per game.
        """
        print("Processing Bronze DST -> Silver...")
        files_data = self._load_json_files("nfl_stats", "dst_*.json")
        if not files_data:
            print("No DST data found in Bronze. Skipping...")
            return None

        rows = []
        for filename, data in files_data:
            try:
                season = filename.split('_')[1].split('.')[0]
            except Exception:
                season = "Unknown"
            for row in data:
                games = row.get("games") or 0
                pa_tier_total = self._points_allowed_tier(row.get("points_allowed_per_game")) * games
                fantasy_points = (
                    (row.get("sacks") or 0) * 1
                    + (row.get("interceptions") or 0) * 2
                    + (row.get("fumble_recoveries") or 0) * 2
                    + (row.get("safeties") or 0) * 2
                    + (row.get("def_tds") or 0) * 6
                    + pa_tier_total
                )
                team = row.get("team")
                rows.append({
                    "season": row.get("season", season),
                    "team": team,
                    "player_name": f"{team} D/ST",
                    "position": "DST",
                    "conformed_id": f"{str(team).lower()}_dst",
                    "games": games,
                    "sacks": row.get("sacks"),
                    "interceptions": row.get("interceptions"),
                    "fumble_recoveries": row.get("fumble_recoveries"),
                    "def_tds": row.get("def_tds"),
                    "safeties": row.get("safeties"),
                    "points_allowed_per_game": row.get("points_allowed_per_game"),
                    "fantasy_points": fantasy_points,
                    "fantasy_points_ppr": fantasy_points,  # PPR doesn't apply to DST -- kept for schema consistency
                })

        df = pd.DataFrame(rows)
        output_path = os.path.join(self.silver_dir, "dst_logs.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved cleaned DST logs to {output_path}")
        return df

    def process_injuries(self):
        """
        Cleans raw injury reports.
        """
        print("Processing Bronze Injuries -> Silver...")
        path = os.path.join(self.bronze_dir, "injuries", "*.json")
        files = glob.glob(path)

        all_data = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                except Exception as e:
                    print(f"Error loading injury file {file}: {e}")

        if not all_data:
            print("No injury data found in Bronze. Skipping...")
            return None

        df = pd.DataFrame(all_data)

        # Convert lists to strings to avoid unhashable type error
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        df = df.drop_duplicates()

        # Save to Silver
        output_path = os.path.join(self.silver_dir, "injury_logs.parquet")
        df.to_parquet(output_path, index=False)
        print(f"Saved cleaned injury logs to {output_path}")
        return df

    def create_players_master(self, stats_df):
        """
        Creates a master lookup table for all players.
        Prefers the nflverse roster (real current team assignments + a
        reliable ACT/CUT/DEV/RES/RET/EXE status), falling back to
        players_master_raw.json (LeagueLogs/Sleeper) if nflverse wasn't
        fetched, and finally to stats_df if neither is available.
        """
        print("Creating Players Master table...")

        # Preferred: nflverse roster -- see fetcher.fetch_nflverse_roster for
        # why this replaces the LeagueLogs/Sleeper source as the source of
        # truth for team assignment and active/retired status.
        nflverse_path = os.path.join(self.bronze_dir, "nfl_stats", "nflverse_roster_2026.csv")
        if os.path.exists(nflverse_path):
            master = pd.read_csv(nflverse_path)
            master = master.dropna(subset=["full_name", "team", "position"])
            # Multiple rows per player across weeks -- keep the latest
            master = master.sort_values("week").drop_duplicates(subset="full_name", keep="last")

            # Build player_name in the same "F.Last" abbreviated format used
            # everywhere else in this pipeline (nfldata.org's player_name /
            # player_display_name fields), so conformed_id joins correctly
            # against game_logs and the draft board.
            master["player_name"] = master["first_name"].str[0] + "." + master["last_name"]
            master["conformed_id"] = (
                master["player_name"].str.lower().str.strip() + "_" +
                master["position"].str.lower().str.strip()
            )

            # The first-initial.lastname scheme isn't always unique -- e.g.
            # Bijan Robinson and Brian Robinson Jr. both reduce to
            # "b.robinson_rb". A duplicate conformed_id here would fan out
            # the merge in drafter.py (one prediction row matching two master
            # rows), silently duplicating a player onto the draft board with
            # identical stats. Collapse to one row per key rather than risk
            # that -- the dropped player just falls through to the game_logs
            # fallback for metadata (no roster_status, so excluded from the
            # ACT filter), which is a safer failure mode than misattribution.
            dupes = master["conformed_id"].duplicated(keep=False).sum()
            if dupes:
                print(f"  {dupes} rows share a conformed_id with another player (name collision) -- keeping one each")
            master = master.drop_duplicates(subset="conformed_id", keep="first")

            master = master.rename(columns={"status": "roster_status"})
            keep_cols = ["conformed_id", "player_name", "full_name", "position", "team",
                         "roster_status", "years_exp", "college"]
            master = master[[c for c in keep_cols if c in master.columns]]

            output_path = os.path.join(self.silver_dir, "players_master.parquet")
            master.to_parquet(output_path, index=False)
            print(f"Saved players master to {output_path} ({len(master)} players from nflverse)")
            return

        # Fall back: try to load the dedicated players master file from Bronze
        path = os.path.join(self.bronze_dir, "nfl_stats", "players_master_raw.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    raw_players = json.load(f)
                    if not isinstance(raw_players, list):
                        if isinstance(raw_players, dict):
                            for key in ['players', 'data', 'results']:
                                if key in raw_players and isinstance(raw_players[key], list):
                                    raw_players = raw_players[key]
                                    break

                    if isinstance(raw_players, list):
                        master = pd.DataFrame(raw_players)

                        # Normalize Name
                        name_cols = ['player_name', 'full_name', 'name', 'playerName']
                        for col in name_cols:
                            if col in master.columns:
                                master = master.rename(columns={col: 'player_name'})
                                break

                        # Normalize Position
                        pos_cols = ['position', 'pos']
                        for col in pos_cols:
                            if col in master.columns:
                                master = master.rename(columns={col: 'position'})
                                break

                        # Build conformed_id to match process_stats() format: name_position
                        if 'player_name' in master.columns and 'position' in master.columns:
                            master['conformed_id'] = (
                                master['player_name'].str.lower().str.strip() + "_" +
                                master['position'].str.lower().str.strip()
                            )
                        elif 'player_name' in master.columns:
                            master['conformed_id'] = master['player_name'].str.lower().str.strip()
                        elif 'id' in master.columns:
                            master['conformed_id'] = master['id'].astype(str)
                        else:
                            master['conformed_id'] = master.index.astype(str)

                        # Convert lists to strings
                        for col in master.columns:
                            if master[col].apply(lambda x: isinstance(x, list)).any():
                                master[col] = master[col].apply(lambda x: str(x) if isinstance(x, list) else x)

                        output_path = os.path.join(self.silver_dir, "players_master.parquet")
                        master.to_parquet(output_path, index=False)
                        print(f"Saved players master to {output_path}")
                        return
                except: pass

        if stats_df is not None:
            master = stats_df.groupby('conformed_id').agg({
                'player_name': 'first',
                'position': 'first',
                'team': 'last'
            }).reset_index()
            output_path = os.path.join(self.silver_dir, "players_master.parquet")
            master.to_parquet(output_path, index=False)
            print(f"Saved players master to {output_path}")

    def run_pipeline(self):
        """Orchestrates the Bronze to Silver flow."""
        stats_df = self.process_stats()
        dst_df = self.process_dst()

        if dst_df is not None and not dst_df.empty:
            combined = pd.concat([stats_df, dst_df], ignore_index=True) if stats_df is not None else dst_df
            output_path = os.path.join(self.silver_dir, "game_logs.parquet")
            combined.to_parquet(output_path, index=False)
            print(f"Merged DST rows into {output_path}")
            stats_df = combined

        self.process_injuries()
        self.create_players_master(stats_df)
        print("Silver transformation complete.")

if __name__ == "__main__":
    transformer = BronzeToSilver()
    transformer.run_pipeline()

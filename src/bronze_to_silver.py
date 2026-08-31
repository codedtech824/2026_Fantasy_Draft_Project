import os
import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime

class BronzeToSilver:
    """
    Transforms raw JSON data from the Bronze layer into cleaned, conformed
    Parquet tables in the Silver layer.
    """
    def __init__(self, bronze_dir="C:/Users/Keith/nfl-prediction-engine/data/bronze",
                 silver_dir="C:/Users/Keith/nfl-prediction-engine/data/silver"):
        self.bronze_dir = bronze_dir
        self.silver_dir = silver_dir
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
        Incorporates data from players_master_raw.json if available.
        """
        print("Creating Players Master table...")

        # Try to load the dedicated players master file from Bronze
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

                        # Ensure conformed_id exists
                        if 'id' in master.columns:
                            master['conformed_id'] = master['id'].astype(str)
                        elif 'playerId' in master.columns:
                            master['conformed_id'] = master['playerId'].astype(str)
                        else:
                            if 'player_name' in master.columns:
                                master['conformed_id'] = master['player_name'].str.lower().str.strip()
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
        self.process_injuries()
        self.create_players_master(stats_df)
        print("Silver transformation complete.")

if __name__ == "__main__":
    transformer = BronzeToSilver()
    transformer.run_pipeline()

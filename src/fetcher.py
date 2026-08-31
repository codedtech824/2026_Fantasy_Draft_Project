import os
import requests
import pandas as pd
import json
from datetime import datetime

class NFLDataFetcher:
    """
    Handles the ingestion of raw NFL data from verified APIs into the Bronze layer.
    """
    def __init__(self, base_dir="C:/Users/Keith/nfl-prediction-engine/data/bronze"):
        self.base_dir = base_dir
        self.session = requests.Session()
        # Common headers to avoid 403s and 406s
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def _save_raw(self, data, folder, filename):
        """Saves raw data as JSON to the bronze layer."""
        path = os.path.join(self.base_dir, folder)
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, filename)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Saved raw data to {full_path}")
        return full_path

    def fetch_league_logs_data(self):
        """
        Pulls market values and player metadata from LeagueLogs.
        """
        print("Fetching data from LeagueLogs...")
        base_url = "https://developer.leaguelogs.com/v1"

        # 1. Fetch Market Values
        try:
            market_url = f"{base_url}/market/redraft-1qb-12t-ppr1"
            resp = self.session.get(market_url)
            resp.raise_for_status()
            self._save_raw(resp.json(), "nfl_stats", "market_values_2026.json")
        except Exception as e:
            print(f"Error fetching LeagueLogs market: {e}")

        # 2. Fetch all players metadata
        try:
            players_url = f"{base_url}/players"
            resp = self.session.get(players_url)
            resp.raise_for_status()
            self._save_raw(resp.json(), "nfl_stats", "players_master_raw.json")
        except Exception as e:
            print(f"Error fetching LeagueLogs players: {e}")

    def fetch_nfl_data_stats(self, seasons=[2022, 2023, 2024, 2025]):
        """
        Pulls historical stats from NFLData.org.
        """
        print("Fetching historical stats from NFLData.org...")
        base_url = "https://api.nfldata.org/v1"

        for season in seasons:
            try:
                leaders_url = f"{base_url}/stats/season?season={season}"
                resp = self.session.get(leaders_url)
                resp.raise_for_status()
                data = resp.json()
                self._save_raw(data, "nfl_stats", f"leaders_{season}.json")
            except Exception as e:
                print(f"Error fetching NFLData stats for {season}: {e}")

    def fetch_muffed_metrics(self):
        """
        Pulls advanced metrics from Muffed.ai.
        """
        print("Fetching advanced metrics from Muffed.ai...")
        url = "https://muffed.ai/api/mcp"

        # Simplified payload
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "query_stat_leaders",
                "arguments": {
                    "metric": "epa",
                    "season": 2025
                }
            }
        }

        try:
            # Use a very clean set of headers for this specific request
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            resp = self.session.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            self._save_raw(resp.json(), "nfl_stats", "muffed_epa_leaders.json")
        except Exception as e:
            print(f"Error fetching Muffed metrics: {e}")

    def fetch_injuries_via_leaguelogs(self):
        """
        Pulls player blurbs from LeagueLogs.
        """
        print("Fetching player status/injuries from LeagueLogs...")
        base_url = "https://developer.leaguelogs.com/v1"

        try:
            players_path = os.path.join(self.base_dir, "nfl_stats", "players_master_raw.json")
            if not os.path.exists(players_path):
                print("No players_master_raw.json found. Run fetch_league_logs_data first.")
                return

            with open(players_path, 'r', encoding='utf-8') as f:
                players_raw = json.load(f)

            # Handle dictionary responses (e.g. {'players': [...]})
            players = players_raw
            if isinstance(players_raw, dict):
                for key in ['players', 'data', 'results']:
                    if key in players_raw and isinstance(players_raw[key], list):
                        players = players_raw[key]
                        break

            if not isinstance(players, list):
                print(f"Error: Could not find a list of players in the raw data. Got {type(players)}")
                return

            top_players = players[:100]
            blurbs = []

            for p in top_players:
                p_id = p.get('id') if isinstance(p, dict) else None
                if not p_id: continue

                try:
                    url = f"{base_url}/players/{p_id}/blurb"
                    resp = self.session.get(url)
                    if resp.status_code == 200:
                        blurbs.append(resp.json())
                except:
                    continue

            self._save_raw(blurbs, "injuries", "player_blurbs.json")

        except Exception as e:
            print(f"Error fetching injuries: {e}")

    def run_all(self):
        """Runs the corrected bronze ingestion pipeline."""
        self.fetch_league_logs_data()
        self.fetch_nfl_data_stats()
        self.fetch_muffed_metrics()
        self.fetch_injuries_via_leaguelogs()
        print("Bronze ingestion complete.")

if __name__ == "__main__":
    fetcher = NFLDataFetcher()
    fetcher.run_all()

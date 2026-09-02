import os
import requests
import pandas as pd
import json
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class NFLDataFetcher:
    """
    Handles the ingestion of raw NFL data from verified APIs into the Bronze layer.
    """
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.join(_PROJECT_ROOT, "data", "bronze")
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

    def fetch_nflverse_roster(self, season=2026):
        """
        Pulls the current-season roster from nflverse -- the authoritative,
        community-maintained source for real team assignments, sourced from
        official transaction data and refreshed regularly. This replaces the
        Sleeper-derived players_master_raw.json as the source of truth for
        "is this player actually on a 2026 roster": Sleeper's player index
        never prunes retired players (Tom Brady still shows status='active'
        with no team years after retiring), while nflverse's status field
        (ACT/CUT/DEV/RES/RET/EXE) is reliable and Brady doesn't even appear
        in it. Same fix already validated in the cfb-stats-pipeline repo.
        """
        print(f"Fetching {season} roster from nflverse...")
        url = f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{season}.csv"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            path = os.path.join(self.base_dir, "nfl_stats")
            os.makedirs(path, exist_ok=True)
            full_path = os.path.join(path, f"nflverse_roster_{season}.csv")
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            print(f"Saved raw data to {full_path}")
        except Exception as e:
            print(f"Error fetching nflverse roster for {season}: {e}")

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
        Pulls historical stats from NFLData.org, paginating through every
        record. /stats/season returns ~1900+ player-seasons but only 50 per
        page -- an unpaginated single request silently keeps just the top 50
        by score, which starves every position of depth (worst for TE, since
        TEs score lower on average and get squeezed out of a combined top-50
        cut first).
        """
        print("Fetching historical stats from NFLData.org...")
        base_url = "https://api.nfldata.org/v1"

        for season in seasons:
            try:
                all_records = []
                offset, limit = 0, 500
                while True:
                    resp = self.session.get(
                        f"{base_url}/stats/season",
                        params={"season": season, "limit": limit, "offset": offset},
                    )
                    resp.raise_for_status()
                    page = resp.json()
                    records = page.get("data", [])
                    all_records.extend(records)
                    if len(all_records) >= page.get("total", 0) or len(records) < limit:
                        break
                    offset += limit

                self._save_raw({"data": all_records, "total": len(all_records)}, "nfl_stats", f"leaders_{season}.json")
                print(f"  {season}: {len(all_records)} player-seasons")
            except Exception as e:
                print(f"Error fetching NFLData stats for {season}: {e}")

    def fetch_dst_stats(self, seasons=[2022, 2023, 2024, 2025]):
        """
        Builds team-defense (D/ST) season totals from two sources:
        - Per-player defensive counting stats (sacks, INTs, fumble
          recoveries, def/return TDs, safeties) already present on every
          player row from /stats/season, summed by team.
        - Points allowed per team, from /games (home_score/away_score of
          every game that team played).
        There's no dedicated team-defense endpoint on this API, so this is
        assembled rather than fetched as one payload.
        """
        print("Building team defense (D/ST) stats from player + game data...")
        base_url = "https://api.nfldata.org/v1"

        for season in seasons:
            try:
                games_resp = self.session.get(f"{base_url}/games", params={"season": season, "limit": 500})
                games_resp.raise_for_status()
                games = games_resp.json().get("data", [])
                points_allowed = {}
                for g in games:
                    if g.get("home_score") is None or g.get("away_score") is None:
                        continue
                    home, away = g.get("home_team"), g.get("away_team")
                    points_allowed.setdefault(home, []).append(g["away_score"])
                    points_allowed.setdefault(away, []).append(g["home_score"])

                leaders_path = os.path.join(self.base_dir, "nfl_stats", f"leaders_{season}.json")
                if not os.path.exists(leaders_path):
                    print(f"  {season}: no leaders file yet -- run fetch_nfl_data_stats first, skipping DST")
                    continue
                with open(leaders_path, "r", encoding="utf-8") as f:
                    players = json.load(f).get("data", [])

                def_fields = ["def_sacks", "def_interceptions", "def_tds", "def_safeties"]
                team_totals = {}
                for p in players:
                    team = p.get("recent_team")
                    if not team:
                        continue
                    t = team_totals.setdefault(team, {k: 0 for k in def_fields} | {"fumble_recoveries": 0})
                    for k in def_fields:
                        t[k] += p.get(k) or 0
                    t["fumble_recoveries"] += (p.get("fumble_recovery_opp") or 0)

                dst_rows = []
                for team, allowed in points_allowed.items():
                    stats = team_totals.get(team, {k: 0 for k in def_fields} | {"fumble_recoveries": 0})
                    dst_rows.append({
                        "team": team,
                        "season": season,
                        "games": len(allowed),
                        "sacks": stats["def_sacks"],
                        "interceptions": stats["def_interceptions"],
                        "fumble_recoveries": stats["fumble_recoveries"],
                        "def_tds": stats["def_tds"],
                        "safeties": stats["def_safeties"],
                        "points_allowed_per_game": sum(allowed) / len(allowed) if allowed else None,
                        "points_allowed_total": sum(allowed),
                    })

                self._save_raw(dst_rows, "nfl_stats", f"dst_{season}.json")
                print(f"  {season}: {len(dst_rows)} team defenses")
            except Exception as e:
                print(f"Error building DST stats for {season}: {e}")

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
        self.fetch_nflverse_roster()
        self.fetch_nfl_data_stats()
        self.fetch_dst_stats()
        self.fetch_muffed_metrics()
        self.fetch_injuries_via_leaguelogs()
        print("Bronze ingestion complete.")

if __name__ == "__main__":
    fetcher = NFLDataFetcher()
    fetcher.run_all()

import os
import pandas as pd
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class NFLDrafter:
    """
    Implements Value Based Drafting (VBD) and Scarcity Analysis.
    """
    def __init__(self, processed_dir=None, silver_dir=None):
        self.processed_dir = processed_dir or os.path.join(_PROJECT_ROOT, "data", "processed")
        self.silver_dir = silver_dir or os.path.join(_PROJECT_ROOT, "data", "silver")

    def _load_predictions(self):
        path = os.path.join(self.processed_dir, "final_predictions.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def _load_players_master(self):
        path = os.path.join(self.silver_dir, "players_master.parquet")
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def calculate_vbd(self):
        """
        Converts projected points into VBD scores.
        """
        print("Calculating Value Based Drafting (VBD) scores...")
        preds = self._load_predictions()
        master = self._load_players_master()

        if preds is None:
            print("Missing predictions. Run the predictor first.")
            return

        # Merge predictions with player metadata
        # Try players_master first; fall back to game_logs if too few matches
        if master is not None:
            df = preds.merge(master, on='conformed_id', how='left')
        else:
            df = preds.copy()

        # Fall back to game_logs for player metadata on a per-row basis, for
        # whichever rows didn't match players_master -- e.g. nflverse groups
        # defensive/O-line positions more broadly (DB/OL/DL) than game_logs'
        # granular labels (CB/SAF/OT/DT), so those rows won't match on
        # conformed_id even though the player is real and current. This
        # doesn't affect roster_status filtering below: unmatched rows simply
        # have no roster_status, which is the right outcome for the fantasy
        # positions we actually care about (QB/RB/WR/TE/K), where the match
        # rate against players_master is high.
        if 'position' not in df.columns or df['position'].isna().any():
            game_logs_path = os.path.join(self.silver_dir, "game_logs.parquet")
            if os.path.exists(game_logs_path):
                logs = pd.read_parquet(game_logs_path)
                meta_cols = [c for c in ['conformed_id', 'player_name', 'position', 'team'] if c in logs.columns]
                player_meta = logs[meta_cols].drop_duplicates('conformed_id').set_index('conformed_id')
                missing = df['position'].isna() if 'position' in df.columns else pd.Series(True, index=df.index)
                unmatched_count = int(missing.sum())
                for col in ['player_name', 'position', 'team']:
                    if col in player_meta.columns:
                        if col not in df.columns:
                            df[col] = pd.NA
                        df.loc[missing, col] = df.loc[missing, 'conformed_id'].map(player_meta[col])
                print(f"Filled {unmatched_count} rows unmatched in players_master from game_logs metadata.")

        # 0. Exclude players not on a current active NFL roster (e.g. Tom
        # Brady's final 2022 season is still in the training data and would
        # otherwise get projected forward as a startable QB). Uses the
        # nflverse roster_status column (ACT/CUT/DEV/RES/RET/EXE) when
        # available -- it's reliable, unlike the old LeagueLogs/Sleeper
        # roster snapshot, whose status field marks Brady "active" with no
        # team while also showing no team for plenty of players who
        # definitely are active (e.g. Tyreek Hill). D/ST rows are exempt --
        # they're team defenses, not individual players, so they never match
        # a roster row. Falls back to a small manual list if nflverse data
        # wasn't fetched (roster_status column absent).
        if "roster_status" in df.columns:
            is_dst = df.get("position", "") == "DST"
            is_active = df["roster_status"] == "ACT"
            df = df[is_dst | is_active]
        elif "conformed_id" in df.columns:
            RETIRED_PLAYER_IDS = {"t.brady_qb"}
            df = df[~df["conformed_id"].isin(RETIRED_PLAYER_IDS)]

        # 1. Define Replacement Level (Baseline) per position.
        # Computed from the actual distribution of ml_projected_points rather
        # than a fixed guess -- baseline = the projected value of the last
        # startable player at that position in a 13-team league (standard VBD
        # replacement level). Different positions land on very different raw
        # point scales here (e.g. DST's "points" -- a mean across counting
        # stats like sacks/INTs -- is nowhere near a QB's yardage-heavy mean),
        # so a fixed baseline like "250 for every QB" ends up meaning
        # different things per position and can make a whole position appear
        # uniformly over- or under-valued regardless of who's actually good.
        starters_per_position = {'QB': 13, 'RB': 26, 'WR': 26, 'TE': 13, 'DST': 13}
        baselines = {}
        for pos, rank in starters_per_position.items():
            pos_values = df.loc[df['position'] == pos, 'ml_projected_points'].sort_values(ascending=False)
            if len(pos_values) >= rank:
                baselines[pos] = pos_values.iloc[rank - 1]
            elif len(pos_values):
                baselines[pos] = pos_values.min()
            else:
                baselines[pos] = 0
        default_baseline = df['ml_projected_points'].median() if not df.empty else 0

        # 2. Calculate Value
        def get_value(row):
            pos = str(row.get('position', 'Unknown')).upper()
            baseline = baselines.get(pos, default_baseline)
            return row['ml_projected_points'] - baseline

        df['vbd_score'] = df.apply(get_value, axis=1)

        # 3. Scarcity Multiplier
        scarcity_boost = {
            'TE': 1.2,
            'RB': 1.1,
            'WR': 1.0,
            'QB': 0.9,
            'DST': 1.0
        }
        df['final_draft_value'] = df.apply(
            lambda r: r['vbd_score'] * scarcity_boost.get(str(r.get('position', 'Unknown')).upper(), 1.0),
            axis=1
        )

        # Sort by final value
        draft_board = df.sort_values(by='final_draft_value', ascending=False)

        # Safely determine which columns we can print
        cols_to_show = ['player_name', 'position', 'ml_projected_points', 'final_draft_value']
        available_cols = [col for col in cols_to_show if col in draft_board.columns]

        print("\n--- 2026 FINAL DRAFT BOARD (Top 10) ---")
        print(draft_board[available_cols].head(10))

        # Save final draft board
        output_path = os.path.join(self.processed_dir, "final_draft_board.parquet")
        draft_board.to_parquet(output_path, index=False)

        return draft_board

if __name__ == "__main__":
    drafter = NFLDrafter()
    drafter.calculate_vbd()

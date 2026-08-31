import os
import pandas as pd
import numpy as np

class NFLDrafter:
    """
    Implements Value Based Drafting (VBD) and Scarcity Analysis.
    """
    def __init__(self, processed_dir="C:/Users/Keith/nfl-prediction-engine/data/processed",
                 silver_dir="C:/Users/Keith/nfl-prediction-engine/data/silver"):
        self.processed_dir = processed_dir
        self.silver_dir = silver_dir

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

        if preds is None or master is None:
            print("Missing predictions or player master data.")
            return

        # Merge predictions with player positions
        df = preds.merge(master, on='conformed_id')

        # 1. Define Replacement Level (Baseline) per position
        baselines = {
            'QB': 250,
            'RB': 150,
            'WR': 140,
            'TE': 100
        }

        # 2. Calculate Value
        def get_value(row):
            pos = str(row.get('position', 'Unknown')).upper()
            baseline = baselines.get(pos, 100)
            return row['ml_projected_points'] - baseline

        df['vbd_score'] = df.apply(get_value, axis=1)

        # 3. Scarcity Multiplier
        scarcity_boost = {
            'TE': 1.2,
            'RB': 1.1,
            'WR': 1.0,
            'QB': 0.9
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

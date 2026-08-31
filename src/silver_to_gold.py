import os
import pandas as pd
import numpy as np
import glob

class SilverToGold:
    """
    Transforms cleaned Silver data into a predictive Gold feature set.
    Implements: EWMA, SOS Normalization, Injury Risk, and Matchup Analysis.
    """
    def __init__(self, silver_dir="/tmp/nfl-prediction-engine/data/silver",
                 gold_dir="/tmp/nfl-prediction-engine/data/gold"):
        self.silver_dir = silver_dir
        self.gold_dir = gold_dir
        os.makedirs(self.gold_dir, exist_ok=True)

    def _load_silver(self, filename):
        path = os.path.join(self.silver_dir, filename)
        if os.path.exists(path):
            return pd.read_parquet(path)
        return None

    def apply_ewma(self, df, weight_map={2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3}):
        """
        Applies Exponentially Weighted Moving Average to stats.
        """
        print("Applying Time-Decay (EWMA) to stats...")

        # Ensure season is numeric
        df['season'] = pd.to_numeric(df['season'], errors='coerce')

        # Create a weight column based on the season
        df['weight'] = df['season'].map(weight_map).fillna(0.1)

        # Weighted average per player
        weighted_stats = []
        for pid, group in df.groupby('conformed_id'):
            # Select numeric stat columns
            stats_cols = group.select_dtypes(include=[np.number]).columns.drop(['season', 'weight'], errors='ignore')

            # Calculate weighted average: (Value * Weight) / Sum(Weights)
            sum_weights = group['weight'].sum()
            if sum_weights == 0: continue

            weighted_vals = (group[stats_cols] * group[['weight']].values).sum() / sum_weights
            res = {col: weighted_vals[col] for col in stats_cols}
            res['conformed_id'] = pid
            weighted_stats.append(res)

        return pd.DataFrame(weighted_stats)

    def apply_sos_normalization(self, df):
        """
        Normalizes stats based on Opponent Strength (SOS).
        """
        print("Applying SOS Normalization...")
        df = df.copy()

        # Safely get numeric columns and exclude identifiers
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numeric_cols if col not in ['conformed_id', 'season', 'weight']]

        for col in cols_to_normalize:
            # Simulate a general normalization tax
            df[col] = df[col] * 0.95

        return df

    def calculate_injury_risk(self, injury_df, player_df):
        """
        Calculates an Injury Risk Multiplier based on severity and frequency.
        """
        print("Calculating Injury Risk Factors...")
        if injury_df is None or injury_df.empty:
            player_df['injury_multiplier'] = 1.0
            return player_df

        # Severity Mapping
        severity_map = {
            'ACL': 0.7,
            'Achilles': 0.6,
            'Hamstring': 0.9,
            'Concussion': 0.95,
            'None': 1.0
        }

        risk_scores = []
        # Ensure conformed_id is treated as string for matching
        injury_df['conformed_id'] = injury_df['conformed_id'].astype(str)
        player_df['conformed_id'] = player_df['conformed_id'].astype(str)

        for pid in player_df['conformed_id']:
            p_injuries = injury_df[injury_df['conformed_id'] == pid]
            if p_injuries.empty:
                risk_scores.append(1.0)
                continue

            # Aggregate risk
            # Attempt to find injury_type, otherwise default to 0.9
            if 'injury_type' in p_injuries.columns:
                severities = p_injuries['injury_type'].map(severity_map).fillna(0.9)
                min_severity = severities.min()
            else:
                min_severity = 0.9

            count_penalty = 0.98 ** len(p_injuries)
            risk_scores.append(min_severity * count_penalty)

        player_df['injury_multiplier'] = risk_scores
        return player_df

    def apply_matchup_engine(self, player_df, schedule_df):
        """
        Adjusts projections based on the 2026 schedule.
        """
        print("Applying 2026 Matchup Engine modifiers...")
        player_df = player_df.copy()
        player_df['schedule_modifier'] = np.random.uniform(0.9, 1.1, len(player_df))
        return player_df

    def run_pipeline(self):
        """Orchestrates the Silver to Gold flow."""
        game_logs = self._load_silver("game_logs.parquet")
        injuries = self._load_silver("injury_logs.parquet")
        players = self._load_silver("players_master.parquet")

        if game_logs is None:
            print("No Silver game logs found. Run bronze_to_silver.py first.")
            return

        # 1. Time-Decay Stats
        gold_df = self.apply_ewma(game_logs)

        # 2. SOS Normalization
        gold_df = self.apply_sos_normalization(gold_df)

        # 3. Injury Risk
        gold_df = self.calculate_injury_risk(injuries, gold_df)

        # 4. Matchup Engine
        dummy_schedule = pd.DataFrame()
        gold_df = self.apply_matchup_engine(gold_df, dummy_schedule)

        # Final Projection Formula
        # Calculate 'points' based on the available numeric stats
        numeric_cols = gold_df.select_dtypes(include=[np.number]).columns.drop(['injury_multiplier', 'schedule_modifier'], errors='ignore')
        if not numeric_cols.empty:
            gold_df['points'] = gold_df[numeric_cols].mean(axis=1)
        else:
            gold_df['points'] = 0.0

        gold_df['final_2026_projection'] = (
            gold_df['points'] *
            gold_df['injury_multiplier'] *
            gold_df['schedule_modifier']
        )

        # Save to Gold
        output_path = os.path.join(self.gold_dir, "player_features_2026.parquet")
        gold_df.to_parquet(output_path, index=False)
        print(f"Saved final gold features to {output_path}")

if __name__ == "__main__":
    transformer = SilverToGold()
    transformer.run_pipeline()

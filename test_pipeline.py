"""
Automated end-to-end test for the NFL Prediction Engine.
Run from the project root: python test_pipeline.py
"""
import os
import sys
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((name, condition))


# ── Stage 1: Bronze Ingestion ────────────────────────────────────────────────
print("\n=== Stage 1: Bronze Ingestion ===")
try:
    from src.fetcher import NFLDataFetcher
    fetcher = NFLDataFetcher()
    fetcher.run_all()

    bronze_nfl = os.path.join(fetcher.base_dir, "nfl_stats")
    leaders = [f for f in os.listdir(bronze_nfl) if f.startswith("leaders_") and f.endswith(".json")]
    check("Leaders JSON files fetched", len(leaders) >= 1, f"{len(leaders)} files")
    check("players_master_raw.json exists", os.path.exists(os.path.join(bronze_nfl, "players_master_raw.json")))
    check("market_values_2026.json exists", os.path.exists(os.path.join(bronze_nfl, "market_values_2026.json")))
except Exception:
    print(traceback.format_exc())
    check("Bronze ingestion", False, "Exception raised")


# ── Stage 2: Bronze → Silver ─────────────────────────────────────────────────
print("\n=== Stage 2: Bronze to Silver ===")
try:
    from src.bronze_to_silver import BronzeToSilver
    transformer = BronzeToSilver()
    transformer.run_pipeline()

    silver_dir = transformer.silver_dir
    game_logs_path = os.path.join(silver_dir, "game_logs.parquet")
    master_path    = os.path.join(silver_dir, "players_master.parquet")

    check("game_logs.parquet written", os.path.exists(game_logs_path))
    check("players_master.parquet written", os.path.exists(master_path))

    if os.path.exists(game_logs_path):
        df = pd.read_parquet(game_logs_path)
        check("game_logs has rows", len(df) > 0, f"{len(df)} rows")
        check("conformed_id column present", "conformed_id" in df.columns)
        check("position column present", "position" in df.columns)
except Exception:
    print(traceback.format_exc())
    check("Bronze to Silver", False, "Exception raised")


# ── Stage 3: Silver → Gold ────────────────────────────────────────────────────
print("\n=== Stage 3: Silver to Gold ===")
try:
    from src.silver_to_gold import SilverToGold
    gold_transformer = SilverToGold()
    gold_transformer.run_pipeline()

    gold_path = os.path.join(gold_transformer.gold_dir, "player_features_2026.parquet")
    check("player_features_2026.parquet written", os.path.exists(gold_path))

    if os.path.exists(gold_path):
        gold = pd.read_parquet(gold_path)
        check("Gold has rows", len(gold) > 0, f"{len(gold)} players")
        check("final_2026_projection column present", "final_2026_projection" in gold.columns)
        check("injury_multiplier column present", "injury_multiplier" in gold.columns)
except Exception:
    print(traceback.format_exc())
    check("Silver to Gold", False, "Exception raised")


# ── Stage 4: ML Prediction ────────────────────────────────────────────────────
print("\n=== Stage 4: ML Prediction ===")
try:
    from src.predictor import NFLPredictor
    predictor = NFLPredictor()
    results_df = predictor.train_and_predict()

    pred_path = os.path.join(predictor.processed_dir, "final_predictions.parquet")
    check("final_predictions.parquet written", os.path.exists(pred_path))

    if os.path.exists(pred_path):
        preds = pd.read_parquet(pred_path)
        check("Predictions has rows", len(preds) > 0, f"{len(preds)} players")
        check("ml_projected_points column present", "ml_projected_points" in preds.columns)
        check("No NaN projections", preds["ml_projected_points"].isna().sum() == 0)
except Exception:
    print(traceback.format_exc())
    check("ML Prediction", False, "Exception raised")


# ── Stage 5: Draft Board ──────────────────────────────────────────────────────
print("\n=== Stage 5: Draft Board ===")
try:
    from src.drafter import NFLDrafter
    drafter = NFLDrafter()
    board = drafter.calculate_vbd()

    board_path = os.path.join(drafter.processed_dir, "final_draft_board.parquet")
    check("final_draft_board.parquet written", os.path.exists(board_path))

    if board is not None and len(board) > 0:
        check("Draft board has players", len(board) > 0, f"{len(board)} players")
        check("final_draft_value column present", "final_draft_value" in board.columns)
        check("Sorted descending by draft value",
              board["final_draft_value"].iloc[0] >= board["final_draft_value"].iloc[-1])
except Exception:
    print(traceback.format_exc())
    check("Draft Board", False, "Exception raised")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"Results: {passed}/{total} checks passed")

if passed == total:
    print("SUCCESS: All checks passed. Pipeline is working end-to-end.")
else:
    failed = [name for name, ok in results if not ok]
    print(f"FAILED checks: {', '.join(failed)}")
    sys.exit(1)

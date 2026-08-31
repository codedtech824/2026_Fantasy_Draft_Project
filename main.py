import sys
import logging
from src.fetcher import NFLDataFetcher
from src.bronze_to_silver import BronzeToSilver
from src.silver_to_gold import SilverToGold
from src.predictor import NFLPredictor
from src.drafter import NFLDrafter

# Setup logging to both file and console
# We specify encoding='utf-8' for the FileHandler to support emojis in the log file
# We use text-based indicators for the console to prevent UnicodeEncodeErrors on Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_full_pipeline():
    """
    Orchestrates the full Medallion Architecture flow:
    Bronze -> Silver -> Gold -> Prediction -> Draft Board
    """
    logger.info("Starting NFL Prediction Engine Pipeline...")
    logger.info("==============================================")

    try:
        # Step 1: Bronze Ingestion
        logger.info("\n[1/5] Ingesting Raw Data (Bronze Layer)...")
        fetcher = NFLDataFetcher()
        fetcher.run_all()

        # Step 2: Silver Transformation
        logger.info("\n[2/5] Cleansing & Conforming Data (Silver Layer)...")
        silver_transformer = BronzeToSilver()
        silver_transformer.run_pipeline()

        # Step 3: Gold Feature Engineering
        logger.info("\n[3/5] Engineering Predictive Features (Gold Layer)...")
        gold_transformer = SilverToGold()
        gold_transformer.run_pipeline()

        # Step 4: ML Projection
        logger.info("\n[4/5] Running ML Prediction Model...")
        predictor = NFLPredictor()
        predictor.train_and_predict()

        # Step 5: VBD Draft Engine
        logger.info("\n[5/5] Generating Final Draft Board...")
        drafter = NFLDrafter()
        drafter.calculate_vbd()

        logger.info("\n==============================================")
        logger.info("SUCCESS: Pipeline Complete! Your 2026 Draft Board is ready.")
        logger.info("Location: data/processed/final_draft_board.parquet")

    except Exception as e:
        logger.error(f"\nFAILED: Pipeline Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_full_pipeline()
    print("\n" + "="*50)
    input("Pipeline process finished. Press ENTER to close this window...")

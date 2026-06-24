"""
run_all.py – Run both Stage 1 and Stage 2 pipelines
=====================================================

Usage examples
--------------
# Single labelled CSV; each stage holds out its own 80/20 split internally
python run_all.py --data data/dataset.csv

# Train on the labelled set, predict on the released (unlabelled) validation set
python run_all.py --train data/dataset.csv --test data/validationset.csv

Outputs are written to results/group55_classifications_1.csv (Stage 1) and
results/group55_classifications_2.csv (Stage 2).
"""

import sys
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
from stage1.stage1_pipeline import run_stage1
from stage2.stage2_pipeline import run_stage2

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    parser = argparse.ArgumentParser(
        description="NLP Project — Fact vs Opinion Classifier (Stage 1 + Stage 2)"
    )
    parser.add_argument("--data",   type=str, help="Single labelled CSV (auto-split 80/20)")
    parser.add_argument("--train",  type=str, help="Training CSV (both stages)")
    parser.add_argument("--test",   type=str, help="Test CSV (both stages)")
    parser.add_argument("--epochs", type=int, default=4, help="Stage 2 fine-tuning epochs")

    args = parser.parse_args()

    train_path = args.train or args.data
    test_path  = args.test

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 1 – Traditional NLP")
    logger.info("=" * 60)
    run_stage1(
        train_path=train_path,
        test_path=test_path,
    )

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STAGE 2 – LLM-Based Classification")
    logger.info("=" * 60)
    run_stage2(
        train_path=train_path,
        test_path=test_path,
        epochs=args.epochs,
    )

    logger.info("=" * 60)
    logger.info("All done. Results saved in: %s", RESULTS_DIR)
    logger.info("  Stage 1 -> %s", RESULTS_DIR / "group55_classifications_1.csv")
    logger.info("  Stage 2 -> %s", RESULTS_DIR / "group55_classifications_2.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

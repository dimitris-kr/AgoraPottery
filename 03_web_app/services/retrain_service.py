"""
retrain_service.py
------------------
FastAPI-side logic for the retraining pipeline.

Responsibilities:
  - Check eligibility (new validated items exist outside current training run)
  - Build stratified train/val split over ALL items (original + new)
  - Create TrainingRun + PotteryItemInTrainingRun records in DB
  - Build the payload for Modal
  - Spawn the Modal job asynchronously
  - Handle the webhook callback from Modal (finalize DB state)
"""
from sqlalchemy.orm import Session

from models import PotteryItemInTrainingRun, PotteryItem
from services import get_current_training_run


# CONFIG

TRAIN_SPLIT    = 0.80
VAL_SPLIT      = 0.10
TEST_SPLIT     = 0.10
RANDOM_STATE   = 42
STRATIFY_COL   = "historical_period"


# ELIGIBILITY

def count_new_validated_items(db: Session) -> int:
    """
    Count validated PotteryItems that have a ChronologyLabel
    but are NOT in the current training run (train or val split).
    These are the items that would be added to the next training run.
    """
    current_run = get_current_training_run(db)
    if not current_run:
        return 0

    # IDs already in current training run (train + val only; test excluded)
    in_current = (
        db.query(PotteryItemInTrainingRun.pottery_item_id)
        .filter(
            PotteryItemInTrainingRun.training_run_id == current_run.id,
            PotteryItemInTrainingRun.split.in_(["train", "val"]),
        )
        .subquery()
    )

    # Items with a chronology label that are NOT in the current run
    count = (
        db.query(PotteryItem)
        .join(PotteryItem.chronology_label)
        .filter(PotteryItem.id.notin_(in_current))
        .count()
    )

    return count

def get_eligibility(db: Session) -> dict:
    new_items = count_new_validated_items(db)
    return {
        "eligible": new_items > 0,
        "new_items_count": new_items,
    }
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
import os

import numpy as np
from fastapi import HTTPException
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from models import PotteryItemInTrainingRun, PotteryItem, ModelVersion, Model, ChronologyLabel, TrainingRun
from schemas import EligibilitySchema, RetrainStartedSchema, TrainingRunSchema
from services import get_current_training_run

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

TRAIN_SPLIT = 0.90
VAL_SPLIT = 0.10
# NO TEST_SPLIT
RANDOM_STATE = 42
STRATIFY_COL = "historical_period"

HF_IMAGE_REPO = os.getenv("HF_IMAGE_REPO")
HF_TFIDF_REPO = os.getenv("TFIDF_REPO")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")


# ──────────────────────────────────────────────
# ELIGIBILITY
# ──────────────────────────────────────────────

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


def get_eligibility(db: Session) -> EligibilitySchema:
    new_items = count_new_validated_items(db)
    return EligibilitySchema(
        eligible=new_items > 0,
        new_items_count=new_items,
    )


# ──────────────────────────────────────────────
# RETRAIN
# ──────────────────────────────────────────────

def trigger_retrain(db: Session) -> RetrainStartedSchema:
    """
    Main entry point called by the /retrain endpoint.
    1. Validate eligibility
    2. Build new split over ALL labelled items
    3. Create TrainingRun + pivot records
    4. Collect model configs from DB
    5. Spawn Modal job
    6. Return job info
    """
    # ── 1. Eligibility ──
    eligibility = get_eligibility(db)
    if not eligibility.eligible:
        raise HTTPException(
            status_code=409,
            detail="No new validated items since the last training run. Retraining not needed.",
        )

    if not WEBHOOK_URL:
        raise HTTPException(
            status_code=500,
            detail="WEBHOOK_URL environment variable is not set.",
        )

    # ── 2. Fetch all items that have a chronology label ──
    all_items = (
        db.query(PotteryItem)
        .join(PotteryItem.chronology_label)
        .join(ChronologyLabel.historical_period)
        .all()
    )

    # ── 3. Build stratified train/val split ──
    # No test split: in production retraining, real predictions with user
    # feedback serve as the effective test set. Every labelled sample should
    # contribute to training.
    item_ids = [item.id for item in all_items]
    strat_labels = [item.chronology_label.historical_period.name for item in all_items]

    idx = np.arange(len(item_ids))
    try:
        idx_train, idx_val = train_test_split(
            idx,
            test_size=VAL_SPLIT,
            stratify=strat_labels,
            random_state=RANDOM_STATE,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=f"Could not stratify split: {e}")

    split_map = {}
    for i in idx_train: split_map[item_ids[i]] = "train"
    for i in idx_val:   split_map[item_ids[i]] = "val"

    # ── 4. Determine new version string ──
    prev_version = _get_current_model_version_string(db)
    new_version = _bump_version(prev_version)

    # ── 5. Create new TrainingRun (not yet current — becomes current after webhook) ──
    db.query(TrainingRun).filter(TrainingRun.is_current == True).update(
        {TrainingRun.is_current: False}
    )
    new_run = TrainingRun(
        split_strategy=f"{TRAIN_SPLIT}-{VAL_SPLIT}",
        random_state=RANDOM_STATE,
        is_current=False,  # set True in webhook after training succeeds
    )
    db.add(new_run)
    db.flush()

    # ── 6. Create PotteryItemInTrainingRun records ──
    db.bulk_insert_mappings(
        PotteryItemInTrainingRun,
        [
            {"training_run_id": new_run.id, "pottery_item_id": pid, "split": split}
            for pid, split in split_map.items()
        ],
    )
    db.commit()

    # ── 7. Build item payloads for Modal ──
    items_by_split = {"train": [], "val": []}
    for item in all_items:
        split = split_map[item.id]  # every item is train or val — no test
        label = item.chronology_label
        items_by_split[split].append({
            "pottery_item_id": item.id,
            "description": item.description,
            "image_path": item.image_path,
            "historical_period": label.historical_period.name,
            "start_year": label.start_year,
            "year_range": label.year_range,
        })

    # ── 8. Collect model configs from DB ──
    model_configs = _build_model_configs(db)

    # ── 9. Build full Modal payload ──
    modal_payload = {
        "items_train": items_by_split["train"],
        "items_val": items_by_split["val"],
        "new_version": new_version,
        "prev_version": prev_version,
        "training_run_id": new_run.id,
        "tfidf_repo": HF_TFIDF_REPO,
        "hf_images_repo": HF_IMAGE_REPO,
        "models": model_configs,
        "webhook_url": WEBHOOK_URL,
    }

    # ── 10. Spawn Modal job ──
    # job_id = _spawn_modal_job(modal_payload)
    job_id = "local-test-fake-job-id"

    return RetrainStartedSchema(
        job_id=job_id,
        new_training_run=TrainingRunSchema.model_validate(new_run),
        new_version=new_version,
        train_size=len(items_by_split["train"]),
        val_size=len(items_by_split["val"]),
        status="training_started",
    )


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────

def _get_current_model_version_string(db: Session) -> str:
    mv = (
        db.query(ModelVersion)
        .filter(ModelVersion.is_current == True)
        .first()
    )
    return mv.version if mv else "v1"


def _bump_version(version: str) -> str:
    """'v1' → 'v2',  'v2' → 'v3', etc."""
    try:
        n = int(version.lstrip("v"))
        return f"v{n + 1}"
    except ValueError:
        return version + "_new"


def _build_model_configs(db: Session) -> list[dict]:
    """
    Build the list of {model_repo, task, feature_keys} dicts
    by reading the Model, Task, and FeatureSet tables.
    """
    models = db.query(Model).all()
    configs = []
    for m in models:
        feature_keys = sorted([fs.feature_type for fs in m.feature_sets])
        configs.append({
            "model_repo": m.hf_repo_id,
            "task": m.task.name.lower(),
            "feature_keys": feature_keys,
        })
    return configs


def _spawn_modal_job(payload: dict) -> str:
    """
    Spawn the Modal training function asynchronously.
    Returns the Modal call ID (used as job_id for status polling).
    """
    try:
        import modal
        # Import the deployed Modal function
        # The app name must match what's in modal_training.py: app = modal.App("agora-pottery-retrain")
        TrainingFunction = modal.Function.lookup("agora-pottery-retrain", "run_training")
        call = TrainingFunction.spawn(payload)
        return call.object_id
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to spawn Modal training job: {e}",
        )


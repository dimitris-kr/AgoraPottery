from fastapi import APIRouter

from database import db_dependency
from schemas import EligibilitySchema, RetrainStartedSchema
from services import auth_dependency, trigger_retrain
from services import get_eligibility

import os

router = APIRouter(prefix="/models/retrain", tags=["Model Retraining"])

# ──────────────────────────────────────────────
# ELIGIBILITY
# ──────────────────────────────────────────────

@router.get("/eligibility", response_model=EligibilitySchema)
def retrain_eligibility(
    db: db_dependency,
    user: auth_dependency,
):
    """
    Returns whether retraining is available and how many new
    validated items exist outside the current training run.
    """
    return get_eligibility(db)


# ──────────────────────────────────────────────
# TRIGGER RETRAIN
# ──────────────────────────────────────────────

@router.post("/trigger", response_model=RetrainStartedSchema)
def retrain(
    db: db_dependency,
    user: auth_dependency,
):
    """
    Triggers the full retraining pipeline:
      1. Validates eligibility
      2. Builds a new stratified split over all labelled items
      3. Creates TrainingRun + PotteryItemInTrainingRun records
      4. Spawns a Modal GPU job
      5. Returns immediately with job_id + training_run_id
    """
    return trigger_retrain(db)
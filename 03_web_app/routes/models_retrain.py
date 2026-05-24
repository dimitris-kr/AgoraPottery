from fastapi import APIRouter, Depends

from database import db_dependency
from schemas import EligibilitySchema, RetrainStartedSchema, RetrainFinalizedSchema, WebhookPayloadSchema
from services import auth_dependency, trigger_retrain, verify_webhook_secret, finalize_retrain
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

# ──────────────────────────────────────────────
# WEBHOOK — called by Modal when run_training finishes
# ──────────────────────────────────────────────

@router.post("/complete", response_model=RetrainFinalizedSchema)
def retrain_complete(
    db: db_dependency,
    payload: WebhookPayloadSchema,
    _: None = Depends(verify_webhook_secret),
):
    """
    Webhook receiver — called by the Modal `run_training` function on completion.

    Authenticated via `Authorization: Bearer <WEBHOOK_SECRET>` header
    (NOT via auth_dependency — this endpoint is hit by Modal, not a logged-in user).

    On success: promotes new ModelVersions + TrainingRun + bumps FeatureSet versions.
    On failure: rolls back the half-created TrainingRun and restores the previous one.
    """
    return finalize_retrain(db, payload)
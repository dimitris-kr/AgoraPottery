from fastapi import APIRouter, Depends

from database import db_dependency
from schemas import (
    EligibilitySchema, RetrainStartedSchema, RetrainFinalizedSchema,
    WebhookPayloadSchema, JobStatusSchema, RunStatusSchema,
)
from services import (
    auth_dependency, trigger_retrain, verify_webhook_secret, finalize_retrain,
    get_eligibility, get_job_status, get_run_status,
)

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


# ──────────────────────────────────────────────
# STATUS POLLING — frontend polls this while a retrain is in flight
# ──────────────────────────────────────────────

@router.get("/status/{job_id}", response_model=JobStatusSchema)
def retrain_status(
    job_id: str,
    user: auth_dependency,
):
    """
    Poll the status of a running Modal training job. Returns one of:
      "running" | "success" | "failure" | "not_found"

    Frontend polls every few seconds to get job progress. Note that
    "success" here means run_training itself returned cleanly — the
    webhook (POST /complete) is what actually flips ModelVersion.is_current.
    """
    return get_job_status(job_id)


# ──────────────────────────────────────────────
# DB FINALIZATION STATUS — polled after Modal reports "success"
# ──────────────────────────────────────────────

@router.get("/run-status/{training_run_id}", response_model=RunStatusSchema)
def retrain_run_status(
    training_run_id: int,
    db: db_dependency,
    user: auth_dependency,
):
    """
    Poll whether the webhook has finalized the retrain in the DB. Returns one of:
      "finalizing" | "finalized" | "failed" | "archived"

    Closes the gap between Modal "success" and the webhook promoting the new
    ModelVersions. Keyed by the TrainingRun id returned from /trigger, which is
    the atomic source of truth for finalization (see services.get_run_status).
    """
    return get_run_status(db, training_run_id)
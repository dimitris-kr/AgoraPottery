from datetime import datetime

from pydantic import BaseModel


# ──────────────────────────────────────────────
# ELIGIBILITY
# ──────────────────────────────────────────────

class EligibilitySchema(BaseModel):
    eligible: bool
    new_items_count: int
    # Soft, advisory guidance of minimum new_items_count for retrain to be meaningful.
    # Does NOT gate eligibility.
    recommended_threshold: int


# ──────────────────────────────────────────────
# TRIGGER RETRAIN  (response)
# ──────────────────────────────────────────────

class TrainingRunSchema(BaseModel):
    id: int
    split_strategy: str
    random_state: int
    is_current: bool
    created_at: datetime

    class Config:
        from_attributes = True

class RetrainStartedSchema(BaseModel):
    job_id: str
    new_training_run: TrainingRunSchema
    new_version: str
    train_size: int
    val_size: int
    status: str


# ──────────────────────────────────────────────
# WEBHOOK PAYLOAD  (incoming from Modal)
# ──────────────────────────────────────────────

class ModelTrainingResultSchema(BaseModel):
    model_id: int            # DB PK — finalize_retrain looks up the Model row by this
    val_loss: float
    train_time: float | None = None
    scores: dict
    train_sample_size: int
    # no per-model `success` flag — run_training is all-or-nothing
    # (any model failing → status="error" → full rollback in finalize_retrain).


class WebhookPayloadSchema(BaseModel):
    training_run_id: int
    new_version: str
    status: str
    error: str | None
    results: list[ModelTrainingResultSchema]
    # shared secret is sent in the `Authorization: Bearer <secret>` header,
    # not the body — see services.retrain_service.verify_webhook_secret.


# ──────────────────────────────────────────────
# WEBHOOK RESPONSE  (finalize retrain)
# ──────────────────────────────────────────────

class RetrainFinalizedSchema(BaseModel):
    status: str
    new_version: str
    models_updated: list[str]


# ──────────────────────────────────────────────
# STATUS POLLING
# ──────────────────────────────────────────────

class JobStatusSchema(BaseModel):
    status: str                 # "running" | "success" | "failure" | "not_found"
    error: str | None = None
    result: dict | None = None
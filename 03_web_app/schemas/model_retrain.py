from datetime import datetime

from pydantic import BaseModel


# ──────────────────────────────────────────────
# ELIGIBILITY
# ──────────────────────────────────────────────

class EligibilitySchema(BaseModel):
    eligible: bool
    new_items_count: int


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
    model_repo: str
    task: str
    feature_keys: list[str]
    val_loss: float
    scores: dict
    train_sample_size: int
    success: bool


class WebhookPayloadSchema(BaseModel):
    training_run_id: int
    new_version: str
    status: str
    error: str | None
    results: list[ModelTrainingResultSchema]
    secret: str


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
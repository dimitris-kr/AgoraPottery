from datetime import datetime
from typing import List

from pydantic import BaseModel, computed_field, Field

from models import ModelHasTarget, ModelUsesFeatureSet, Model


class TaskSchema(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class TargetSchema(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True
        orm_mode = True


class FeatureSetSchema(BaseModel):
    id: int
    feature_type: str
    data_type: str | None
    current_version: str

    class Config:
        from_attributes = True
        orm_mode = True

class ModelSchema(BaseModel):
    id: int
    name: str

    task: TaskSchema

    # Flattened M2M
    targets: list[TargetSchema]
    feature_sets: list[FeatureSetSchema]

    class Config:
        from_attributes = True


class ModelVersionSchema(BaseModel):
    id: int
    model: ModelSchema
    version: str
    train_sample_size: int | None
    val_loss: float | None
    val_accuracy: float | None
    val_mae: float | None
    created_at: datetime

    class Config:
        from_attributes = True
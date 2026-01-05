from typing import Any

from pydantic import BaseModel

from models import Model, ModelVersion


class PredictionResponse(BaseModel):
    prediction: Any
    breakdown: Any
    model: str
    model_version: str
    feature_types: list[str]

    class Config:
        extra = "forbid"
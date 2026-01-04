from typing import Any

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    model_name: str
    model_version: str
    feature_types: list[str]
    features: dict[str, Any]

    class Config:
        extra = "forbid"
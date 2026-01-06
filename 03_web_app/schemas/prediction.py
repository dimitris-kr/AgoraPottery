from typing import Any, Union

from pydantic import BaseModel

from models import Model, ModelVersion


class PredictionResponse(BaseModel):
    prediction: Union[str, list[int]]
    breakdown: dict[str, Any]
    model: str
    model_version: str
    feature_types: list[str]

    class Config:
        extra = "forbid"
from datetime import datetime
from typing import Any, Union

from pydantic import BaseModel

from schemas import ModelVersionSchema, HistoricalPeriodSchema


class PredictionResponse(BaseModel):
    prediction: Union[str, list[int]]
    breakdown: dict[str, Any]
    model: str
    model_version: str
    feature_types: list[str]

    class Config:
        extra = "forbid"


class ChronologyPredictionSchema(BaseModel):
    id: int

    # inputs
    input_text: str | None
    input_image_path: str | None

    # outputs
    historical_period: HistoricalPeriodSchema | None
    start_year: float | None
    end_year: float | None
    midpoint_year: float | None
    year_range: float | None

    breakdown: dict | None
    status: str

    created_at: datetime

    # relations (nested)
    model_version: ModelVersionSchema

    class Config:
        from_attributes = True

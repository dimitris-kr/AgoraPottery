from datetime import datetime
from typing import Any, Union, Literal

from pydantic import BaseModel, computed_field

from schemas import ModelVersionSchema, HistoricalPeriodSchema, PotteryItemSchema
from services import hf_image_url, match_regression


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

    pottery_item: PotteryItemSchema | None

    @computed_field
    @property
    def input_image_url(self) -> str | None:
        return hf_image_url(self.input_image_path)

    @computed_field
    @property
    def match(self) -> Literal["exact", "close", "none", "unknown"]:
        if not self.pottery_item or not self.pottery_item.chronology_label:
            return "unknown"

        true = self.pottery_item.chronology_label

        if self.historical_period:
            return (
                "exact"
                if self.historical_period.id == true.historical_period.id
                else "none"
            )

        return match_regression(self, true)

    class Config:
        from_attributes = True

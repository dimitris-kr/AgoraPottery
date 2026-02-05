from datetime import datetime

from pydantic import BaseModel

class HistoricalPeriodSchema(BaseModel):
    id: int
    name: str
    limit_lower: float
    limit_upper: float

    class Config:
        from_attributes = True

class DataSourceSchema(BaseModel):
    id: int
    description: str

    class Config:
        from_attributes = True

class ChronologyLabelSchema(BaseModel):
    id: int
    # pottery_item: PotteryItemSchema
    historical_period: HistoricalPeriodSchema
    start_year: float
    end_year: float
    midpoint_year: float
    year_range: float

    class Config:
        from_attributes = True

class PotteryItemSchema(BaseModel):
    id: int
    object_id: str | None
    description: str | None
    image_path: str | None
    created_at: datetime | None

    data_source: DataSourceSchema
    chronology_label: ChronologyLabelSchema | None

    class Config:
        from_attributes = True

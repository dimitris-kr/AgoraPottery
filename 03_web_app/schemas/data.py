from pydantic import BaseModel

class HistoricalPeriodSchema(BaseModel):
    id: int
    name: str
    limit_lower: float
    limit_upper: float

    class Config:
        from_attributes = True
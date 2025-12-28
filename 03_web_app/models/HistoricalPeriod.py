from sqlalchemy import Column, String, Float, Integer
from sqlalchemy.orm import relationship
from database import Base

class HistoricalPeriod(Base):
    __tablename__ = "historical_periods"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    chronology_labels = relationship("ChronologyLabel", back_populates="historical_period")

    chronology_predictions = relationship("ChronologyPrediction", back_populates="historical_period")

    # FIELDS
    name = Column(String, nullable=False, unique=True, index=True)
    limit_lower = Column(Float, nullable=False)
    limit_upper = Column(Float, nullable=False)


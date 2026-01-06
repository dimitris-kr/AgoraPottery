from database import Base
from sqlalchemy import Column, Integer, Float, String, Text, ForeignKey, Enum, JSON
from sqlalchemy.orm import relationship


class ChronologyPrediction(Base):
    __tablename__ = "chronology_predictions"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    model_version_id = Column(Integer, ForeignKey("model_versions.id"))
    model_version = relationship("ModelVersion", back_populates="chronology_predictions")

    pottery_item_id = Column(Integer, ForeignKey("pottery_items.id"), nullable=True)
    pottery_item = relationship("PotteryItem", back_populates="chronology_prediction")

    historical_period_id = Column(Integer, ForeignKey("historical_periods.id"), nullable=True)
    historical_period = relationship("HistoricalPeriod", back_populates="chronology_predictions")

    # FIELDS
    input_text = Column(Text, nullable=True)
    input_image_path = Column(String, nullable=True)

    start_year = Column(Float, nullable=True)
    end_year = Column(Float, nullable=True)
    midpoint_year = Column(Float, nullable=True)
    year_range = Column(Float, nullable=True)

    breakdown = Column(JSON, nullable=True)  # probabilities / MC stats

    status = Column(
        Enum("pending", "validated", name="prediction_status"),
        default="pending",
        nullable=False
    )
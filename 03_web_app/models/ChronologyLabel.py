from sqlalchemy import Column, Integer, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from ._timestamps import Timestamps

class ChronologyLabel(Base, Timestamps):
    __tablename__ = "chronology_labels"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    pottery_item_id = Column(Integer, ForeignKey("pottery_items.id"))
    pottery_item = relationship("PotteryItem", back_populates="chronology_label")

    historical_period_id = Column(Integer, ForeignKey("historical_periods.id"))
    historical_period = relationship("HistoricalPeriod", back_populates="chronology_labels")

    # FIELDS
    start_year = Column(Float, nullable=False)
    end_year = Column(Float, nullable=False)
    midpoint_year = Column(Float, nullable=False)
    year_range = Column(Float, nullable=False)



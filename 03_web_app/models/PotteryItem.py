from sqlalchemy import Column, Integer, String, Text, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base
from ._timestamps import Timestamps

class PotteryItem(Base, Timestamps):
    __tablename__ = "pottery_items"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    data_source_id = Column(Integer, ForeignKey("data_sources.id"))
    data_source = relationship("DataSource", back_populates="pottery_items")

    chronology_label = relationship("ChronologyLabel", back_populates="pottery_item", uselist=False)

    chronology_prediction = relationship("ChronologyPrediction", back_populates="pottery_item")

    in_feature_sets = relationship("PotteryItemInFeatureSet", back_populates="pottery_item")

    in_training_run = relationship("PotteryItemInTrainingRun", back_populates="pottery_item")

    # FIELDS
    object_id = Column(String, unique=True, index=True)
    description = Column(Text)
    image_path = Column(String)
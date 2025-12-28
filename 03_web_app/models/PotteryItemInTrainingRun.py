from database import Base
from sqlalchemy import Column, Integer, Float, String, Boolean, Enum, ForeignKey
from sqlalchemy.orm import relationship

# Association(Pivot) Table - Connects PotteryItem and TrainingRun
class PotteryItemInTrainingRun(Base):
    __tablename__ = "pottery_items_in_training_runs"

    # PRIMARY KEY
    # FOREIGN KEYS & RELATIONSHIPS
    training_run_id = Column(Integer, ForeignKey("training_runs.id"), primary_key=True)
    training_run = relationship("TrainingRun", back_populates="pottery_items")

    pottery_item_id = Column(Integer, ForeignKey("pottery_items.id"), primary_key=True)
    pottery_item = relationship("PotteryItem", back_populates="in_training_run")

    # FIELDS
    split = Column(Enum("train", "val", "test", name="dataset_split"), nullable=False)

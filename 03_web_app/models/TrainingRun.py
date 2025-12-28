from database import Base
from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ._timestamps import Timestamps

class TrainingRun(Base, Timestamps):
    __tablename__ = "training_runs"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True)

    # FOREIGN KEYS & RELATIONSHIPS
    model_versions = relationship("ModelVersion", back_populates="training_run")

    pottery_items = relationship("PotteryItemInTrainingRun", back_populates="training_run")

    # FIELDS
    split_strategy = Column(String, nullable=False)  # "80-10-10"
    random_state = Column(Integer, nullable=False)   # 42

    is_current = Column(Boolean, nullable=False, default=False)


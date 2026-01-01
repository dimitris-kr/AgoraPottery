from database import Base
from sqlalchemy import Column, Integer, Float, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from ._timestamps import Timestamps

class ModelVersion(Base, Timestamps):
    __tablename__ = "model_versions"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    model_id = Column(Integer, ForeignKey("models.id"))
    model = relationship("Model", back_populates="model_versions")

    training_run_id = Column(Integer, ForeignKey("training_runs.id"))
    training_run = relationship("TrainingRun", back_populates="model_versions")

    chronology_predictions = relationship("ChronologyPrediction", back_populates="model_version")

    # FIELDS
    version = Column(String, index=True, nullable=False)
    train_sample_size = Column(Integer)
    train_time = Column(Float)
    val_loss = Column(Float)
    val_accuracy = Column(Float, nullable=True)
    val_mae = Column(Float, nullable=True)

    is_current = Column(Boolean, nullable=False, default=False)

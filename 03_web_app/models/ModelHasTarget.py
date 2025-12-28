from database import Base
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship

# Association(Pivot) Table
class ModelHasTarget(Base):
    __tablename__ = "models_have_targets"

    # PRIMARY KEY
    # id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    model_id = Column(Integer, ForeignKey("models.id"), primary_key=True)
    model = relationship("Model", back_populates="targets")

    target_id = Column(Integer, ForeignKey("targets.id"), primary_key=True)
    target = relationship("Target", back_populates="models")
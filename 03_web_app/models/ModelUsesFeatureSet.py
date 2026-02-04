from database import Base
from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship


# Association(Pivot) Table
class ModelUsesFeatureSet(Base):
    __tablename__ = "models_use_feature_sets"

    # PRIMARY KEY
    # id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    model_id = Column(Integer, ForeignKey("models.id"), primary_key=True)
    # model = relationship("Model", back_populates="feature_sets")

    feature_set_id = Column(Integer, ForeignKey("feature_sets.id"), primary_key=True)
    # feature_set = relationship("FeatureSet", back_populates="models")

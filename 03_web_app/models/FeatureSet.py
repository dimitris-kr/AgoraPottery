from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship
from database import Base

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS

    models = relationship("ModelUsesFeatureSet", back_populates="feature_set")

    # FIELDS
    feature_type = Column(String, nullable=False, unique=True, index=True)
    data_type = Column(String)
    hf_repo_id = Column(String, unique=True, nullable=False)
    current_version = Column(String, nullable=False, default="v1")
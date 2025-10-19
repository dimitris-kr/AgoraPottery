from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship
from database import Base

class FeatureSet(Base):
    __tablename__ = "feature_sets"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    pottery_items = relationship("PotteryItemInFeatureSet", back_populates="feature_set")

    # FIELDS
    feature_type = Column(String, nullable=False, unique=True, index=True)
    data_type = Column(String)
    hf_repo_id = Column(String, unique=True)
    path = Column(String)


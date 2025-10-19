from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
from ._timestamps import Timestamps


# Association(Pivot) Table - Connects PotteryItem and FeatureType
class PotteryItemInFeatureSet(Base, Timestamps):
    __tablename__ = "pottery_items_in_feature_sets"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    pottery_item_id = Column(Integer, ForeignKey("pottery_items.id"))
    pottery_item = relationship("PotteryItem", back_populates="in_feature_sets")

    feature_set_id = Column(Integer, ForeignKey("feature_sets.id"))
    feature_set = relationship("FeatureSet", back_populates="pottery_items")

    # FIELDS
    feature_index = Column(Integer, nullable=False)


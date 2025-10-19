from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship
from database import Base

class DataSource(Base):
    __tablename__ = "data_sources"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    pottery_items = relationship("PotteryItem", back_populates="data_source")

    # FIELDS
    description = Column(String, nullable=False, unique=True, index=True)
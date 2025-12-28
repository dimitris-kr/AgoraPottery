from sqlalchemy.orm import relationship

from database import Base
from sqlalchemy import Column, Integer, String


class Target(Base):
    __tablename__ = "targets"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    models = relationship("ModelHasTarget", back_populates="target")

    # FIELDS
    name = Column(String, nullable=False, unique=True, index=True)
from sqlalchemy.orm import relationship

from database import Base
from sqlalchemy import Column, Integer, String


class Target(Base):
    __tablename__ = "targets"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    # models = relationship("ModelHasTarget", back_populates="target")
    models = relationship("Model", secondary="models_have_targets", back_populates="targets")

    # FIELDS
    name = Column(String, nullable=False, unique=True, index=True)
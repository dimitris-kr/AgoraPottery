from sqlalchemy.orm import relationship

from database import Base
from sqlalchemy import Column, Integer, String


class Task(Base):
    __tablename__ = "tasks"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    models = relationship("Model", back_populates="task")

    # FIELDS
    name = Column(String, nullable=False, unique=True, index=True)
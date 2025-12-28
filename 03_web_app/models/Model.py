from database import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship


class Model(Base):
    __tablename__ = "models"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FOREIGN KEYS & RELATIONSHIPS
    task_id = Column(Integer, ForeignKey("tasks.id"))
    task = relationship("Task", back_populates="models")

    targets = relationship("ModelHasTarget", back_populates="model")

    feature_sets = relationship("ModelUsesFeatureSet", back_populates="model")

    model_versions = relationship("ModelVersion", back_populates="model")

    # FIELDS
    name = Column(String, nullable=False, unique=True, index=True)
    hf_repo_id = Column(String, nullable=False)
    model_path = Column(String)
    config_path = Column(String)
    metadata_path = Column(String)
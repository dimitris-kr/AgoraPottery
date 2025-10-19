from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import declarative_mixin

@declarative_mixin
class Timestamps:
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

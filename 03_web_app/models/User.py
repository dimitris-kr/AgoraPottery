from sqlalchemy import Column, String, Integer
from database import Base
from ._timestamps import Timestamps

class User(Base, Timestamps):
    __tablename__ = "users"

    # PRIMARY KEY
    id = Column(Integer, primary_key=True, index=True)

    # FIELDS
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
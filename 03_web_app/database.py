from typing import Annotated

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Neon (serverless Postgres) drops idle connections and auto-suspends compute, so a connection sitting in the pool can be dead by the next request. Guard:
#   pool_pre_ping → test (and transparently reopen) a connection before each use. Fixes "SSL connection has been closed unexpectedly".
#   pool_recycle  → never reuse a connection older than 5 min (Neon's idle window).
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
    # Default 5+10 was too small. 20 + 30 = 50 max: safe on Neon's POOLED endpoint.
    pool_size=20,
    max_overflow=30,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency for routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


db_dependency = Annotated[Session, Depends(get_db)]

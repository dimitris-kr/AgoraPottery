from dotenv import load_dotenv
load_dotenv()

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from routes import router
from services import auth_dependency

# Create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pottery Chronology Predictor API")

# CORS: allow requests from FRONTEND_ORIGINS HF Space variable
# a comma-separated list, e.g. "https://dimitriskr.github.io" (ORIGIN only, no path)
# Falls back to the local dev origins so `ng serve` keeps working locally.
_default_origins = "http://localhost:4200,http://127.0.0.1:4200"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Welcome to the Pottery Chronology Predictor API"
    }

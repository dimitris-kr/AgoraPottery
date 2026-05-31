from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from database import Base, engine
from routes import router
from services import auth_dependency

# Create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Pottery Chronology Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
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

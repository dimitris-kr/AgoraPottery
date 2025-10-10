from fastapi import FastAPI, HTTPException

from database import Base, engine
from routes import router
from services import auth_dependency

# Create tables if not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Agora Pottery Chronology API")

app.include_router(router)


@app.get("/")
def root(user: auth_dependency):
    if not user:
        raise HTTPException(status_code=401, detail="Authentication failed.")
    return {"message": "Welcome to the Pottery Chronology API", "user": user}

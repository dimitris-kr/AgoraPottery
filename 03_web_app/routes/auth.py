from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm

from services import authenticate_user, create_access_token, TOKEN_EXPIRATION
from database import db_dependency
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login")
def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()], db: db_dependency):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access_token = create_access_token(username=user.username, user_id=user.id , expires_delta=timedelta(minutes=TOKEN_EXPIRATION))
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }
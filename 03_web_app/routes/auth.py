from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import OAuth2PasswordRequestForm

from services import authenticate_user, create_access_token, TOKEN_EXPIRATION, change_password_in_users_table, \
    auth_dependency
from database import db_dependency
from datetime import timedelta
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login")
def login(
        db: db_dependency,

        form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    access_token = create_access_token(username=user.username, user_id=user.id,
                                       expires_delta=timedelta(minutes=TOKEN_EXPIRATION))
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

class NewPasswordSchema(BaseModel):
    current_password: str
    new_password: str

@router.post("/change-password")
def change_password(
        db: db_dependency,
        user: auth_dependency,

        payload: NewPasswordSchema,
):
    """Change the logged-in user's password, after verifying current password."""
    change_password_in_users_table(db, user["id"], payload.current_password, payload.new_password)
    return {"message": "Password changed successfully. You can log in with your new password."}

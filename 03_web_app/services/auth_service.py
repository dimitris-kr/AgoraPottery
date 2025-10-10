from datetime import datetime, timedelta
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
import bcrypt
from sqlalchemy.orm import Session

from database import db_dependency
from models import User

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
TOKEN_EXPIRATION = int(os.getenv("TOKEN_EXPIRATION", 60))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
# bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto") # from passlib.context import CryptContext

def hash_password(password: str) -> str:
    # return bcrypt_context.hash(password)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password, hashed_password):
    # return bcrypt_context.verify(plain_password, hashed_password)
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(username: str, user_id: int, expires_delta: timedelta = timedelta(minutes=TOKEN_EXPIRATION)):
    payload = {
        "sub": username,
        'id': user_id,
        'exp': datetime.now() + expires_delta,
    }
    encoded_jwt = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: db_dependency):
    credentials_exception = HTTPException(status_code=401, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user_id = payload.get("id")
        if username is None or user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).get(user_id)
    if not user:
        raise credentials_exception
    return {
        "id": user.id,
        "username": user.username
    }

auth_dependency = Annotated[dict, Depends(get_current_user)]
"""Authentication API endpoints."""
from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from backend.app.core.config import settings
from backend.app.core.security import create_access_token, decode_access_token
from backend.app.services import user_store
from backend.app.services.user_store import DuplicateUserError

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token")


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    username: str
    password: str
    full_name: Optional[str] = None


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None


def _issue_token(username: str) -> dict:
    """Create a bearer-token response for a username."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def get_current_username(token: str = Depends(oauth2_scheme)) -> str:
    """Resolve the bearer token to a registered username.

    Shared dependency for routers that need to know which user is calling
    (music, users, export). Raises 401 for invalid tokens or tokens whose
    user no longer exists.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception
    username: Optional[str] = payload.get("sub")
    if username is None or user_store.get_user(username) is None:
        raise credentials_exception
    return username


@router.post("/register", response_model=Token)
async def register(user: UserCreate):
    """Register a new user and return an access token."""
    try:
        user_store.create_user(
            email=user.email,
            username=user.username,
            password=user.password,
            full_name=user.full_name,
        )
    except DuplicateUserError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    return _issue_token(user.username)


@router.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login and get access token. The password is verified against the
    stored hash; unknown users and wrong passwords are rejected."""
    user = user_store.authenticate(form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _issue_token(user["username"])


@router.get("/me")
async def get_current_user(username: str = Depends(get_current_username)):
    """Get current user information."""
    user = user_store.get_user(username)
    return {
        "username": user["username"],
        "email": user["email"],
        "full_name": user["full_name"],
    }

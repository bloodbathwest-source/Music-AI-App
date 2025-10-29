"""User management API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel, EmailStr
from backend.app.api.auth import oauth2_scheme

router = APIRouter()


class UserProfile(BaseModel):
    """User profile model."""
    id: int
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True


class UserUpdate(BaseModel):
    """User update model."""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    token: str = Depends(oauth2_scheme)  # pylint: disable=unused-argument
):
    """Get user profile."""
    return {
        "id": 1,
        "email": "user@example.com",
        "username": "user",
        "full_name": "Example User",
        "is_active": True
    }


@router.put("/profile", response_model=UserProfile)
async def update_user_profile(
    user_update: UserUpdate,
    token: str = Depends(oauth2_scheme)  # pylint: disable=unused-argument
):
    """Update user profile."""
    return {
        "id": 1,
        "email": user_update.email or "user@example.com",
        "username": "user",
        "full_name": user_update.full_name or "Example User",
        "is_active": True
    }


@router.get("/tracks")
async def get_user_tracks(
    token: str = Depends(oauth2_scheme)  # pylint: disable=unused-argument
):
    """Get user's music tracks."""
    return {"tracks": [], "total": 0}

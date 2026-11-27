"""User management API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from backend.app.api.auth import get_current_username
from backend.app.services import track_store, user_store
from backend.app.services.user_store import DuplicateUserError

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
async def get_user_profile(username: str = Depends(get_current_username)):
    """Get user profile."""
    return user_store.get_user(username)


@router.put("/profile", response_model=UserProfile)
async def update_user_profile(
    user_update: UserUpdate,
    username: str = Depends(get_current_username)
):
    """Update user profile."""
    try:
        return user_store.update_user(
            username,
            full_name=user_update.full_name,
            email=user_update.email,
        )
    except DuplicateUserError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error


@router.get("/tracks")
async def get_user_tracks(username: str = Depends(get_current_username)):
    """Get user's music tracks."""
    tracks = track_store.list_all_tracks(username)
    return {"tracks": tracks, "total": len(tracks)}

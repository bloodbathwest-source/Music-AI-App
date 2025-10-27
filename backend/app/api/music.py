"""Music generation and management API endpoints."""
from typing import List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.app.api.auth import oauth2_scheme
from backend.app.models.music import MusicGenerationRequest, MusicTrack

router = APIRouter()


class GenerationResponse(BaseModel):
    """Music generation response."""
    task_id: str
    status: str
    message: str


@router.post("/generate", response_model=GenerationResponse)
async def generate_music(
    request: MusicGenerationRequest,
    _background_tasks: BackgroundTasks,
    _token: str = Depends(oauth2_scheme)
):
    """Generate music using AI."""
    # Validate track length
    if request.duration < 30 or request.duration > 180:
        raise HTTPException(
            status_code=400,
            detail="Track duration must be between 30 and 180 seconds"
        )

    # In production, this would queue a background task for AI generation
    task_id = f"task_{hash(str(request))}"

    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Music generation task queued successfully"
    }


@router.get("/generate/{task_id}")
async def get_generation_status(
    task_id: str,
    _token: str = Depends(oauth2_scheme)
):
    """Get status of music generation task."""
    # Mock response
    return {
        "task_id": task_id,
        "status": "completed",
        "progress": 100,
        "result": {
            "track_id": "track_123",
            "file_path": "/tracks/generated_music.mp3"
        }
    }


@router.get("/tracks", response_model=List[MusicTrack])
async def list_tracks(
    _skip: int = 0,
    _limit: int = 10,
    _token: str = Depends(oauth2_scheme)
):
    """List user's music tracks."""
    # Mock response
    return []


@router.get("/tracks/{track_id}", response_model=MusicTrack)
async def get_track(
    track_id: str,
    _token: str = Depends(oauth2_scheme)
):
    """Get specific track details."""
    # Mock response
    return {
        "id": track_id,
        "title": "Generated Track",
        "genre": "electronic",
        "duration": 60,
        "file_path": f"/tracks/{track_id}.mp3",
        "created_at": "2025-10-26T17:00:00"
    }


@router.delete("/tracks/{track_id}")
async def delete_track(
    track_id: str,
    _token: str = Depends(oauth2_scheme)
):
    """Delete a track."""
    return {"message": f"Track {track_id} deleted successfully"}


@router.post("/tracks/{track_id}/share")
async def share_track(
    track_id: str,
    is_public: bool = True,
    _token: str = Depends(oauth2_scheme)
):
    """Share or unshare a track."""
    return {
        "track_id": track_id,
        "is_public": is_public,
        "message": "Track sharing settings updated"
    }

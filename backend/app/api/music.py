"""Music generation and management API endpoints."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from fastapi import BackgroundTasks
from pydantic import BaseModel
from backend.app.api.auth import get_current_username
from backend.app.models.music import MusicGenerationRequest
from backend.app.services import track_store
from backend.app.services.ai_service import MusicGenerationService

router = APIRouter()

# One shared generation service for all requests.
generation_service = MusicGenerationService()


class GenerationResponse(BaseModel):
    """Music generation response."""
    task_id: str
    status: str
    message: str


class TrackResponse(BaseModel):
    """Track response model."""
    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str


class ShareRequest(BaseModel):
    """Track sharing request (sent as JSON by the frontend)."""
    is_public: bool = True


def _run_generation(task_id: str):
    """Background task: run the AI service and store the resulting track."""
    task = track_store.get_task(task_id)
    if task is None:
        return
    try:
        result = generation_service.generate_music(task["request"])
        track = track_store.add_track(
            owner=task["owner"],
            genre=result["parameters"]["genre"],
            duration=result["parameters"]["duration"],
            midi_data=result["midi_data"],
            parameters=result["parameters"],
        )
        track_store.complete_task(task_id, track)
    except Exception as error:  # surfaced to the client via the status endpoint
        track_store.fail_task(task_id, str(error))


def _visible_track(track_id: str, username: str) -> dict:
    """Return the full track record if the user owns it or it is public."""
    track = track_store.get_track(track_id)
    if track is None or (track["owner"] != username and not track["is_public"]):
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@router.post("/generate", response_model=GenerationResponse)
async def generate_music(
    request: MusicGenerationRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(get_current_username)
):
    """Generate music using AI. The generation runs as a background task;
    poll GET /generate/{task_id} for the result."""
    # Validate track length. Note: MusicGenerationRequest already enforces
    # 30 <= duration <= 180 via Pydantic, so invalid values are rejected with
    # 422 before this handler runs; this check is defense-in-depth only.
    if request.duration < 30 or request.duration > 180:
        raise HTTPException(
            status_code=400,
            detail="Track duration must be between 30 and 180 seconds"
        )

    task = track_store.create_task(username, request)
    background_tasks.add_task(_run_generation, task["id"])

    return {
        "task_id": task["id"],
        "status": "queued",
        "message": "Music generation task queued successfully"
    }


@router.get("/generate/{task_id}")
async def get_generation_status(
    task_id: str,
    username: str = Depends(get_current_username)
):
    """Get status of music generation task."""
    task = track_store.get_task(task_id)
    if task is None or task["owner"] != username:
        raise HTTPException(status_code=404, detail="Generation task not found")

    response = {
        "task_id": task["id"],
        "status": task["status"],
        "progress": task["progress"],
    }
    if task["status"] == "completed":
        response["result"] = task["result"]
    elif task["status"] == "failed":
        response["error"] = task["error"]
    return response


@router.get("/tracks", response_model=List[TrackResponse])
async def list_tracks(
    skip: int = 0,
    limit: int = 10,
    username: str = Depends(get_current_username)
):
    """List user's music tracks."""
    return track_store.list_tracks(username, skip=skip, limit=limit)


@router.get("/tracks/{track_id}", response_model=TrackResponse)
async def get_track(
    track_id: str,
    username: str = Depends(get_current_username)
):
    """Get specific track details."""
    track = _visible_track(track_id, username)
    return track_store.get_public_track(track["id"])


@router.delete("/tracks/{track_id}")
async def delete_track(
    track_id: str,
    username: str = Depends(get_current_username)
):
    """Delete a track."""
    if not track_store.delete_track(username, track_id):
        raise HTTPException(status_code=404, detail="Track not found")
    return {"message": f"Track {track_id} deleted successfully"}


@router.post("/tracks/{track_id}/share")
async def share_track(
    track_id: str,
    request: ShareRequest,
    username: str = Depends(get_current_username)
):
    """Share or unshare a track."""
    track = track_store.set_track_public(username, track_id, request.is_public)
    if track is None:
        raise HTTPException(status_code=404, detail="Track not found")
    return {
        "track_id": track_id,
        "is_public": track["is_public"],
        "message": "Track sharing settings updated"
    }

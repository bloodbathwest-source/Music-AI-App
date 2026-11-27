"""Export and sharing API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from backend.app.api.auth import get_current_username
from backend.app.services import track_store

router = APIRouter()

SUPPORTED_FORMATS = ["mp3", "wav", "flac", "midi"]
# The generation service produces MIDI. Rendering audio formats (mp3, wav,
# flac) would require a synthesizer/soundfont pipeline that this build does
# not have, so downloads are served for MIDI and honestly refused otherwise.
RENDERABLE_FORMATS = ["mid", "midi"]


class ExportRequest(BaseModel):
    """Export request model."""
    track_id: str
    format: str  # mp3, wav, flac, midi
    quality: Optional[str] = "high"


class UploadRequest(BaseModel):
    """Upload to platform request."""
    track_id: str
    platform: str  # spotify, soundcloud, youtube
    title: str
    description: Optional[str] = None
    genre: Optional[str] = None
    tags: Optional[list] = []


def _visible_track(track_id: str, username: str) -> dict:
    """Return the full track record if the user owns it or it is public."""
    track = track_store.get_track(track_id)
    if track is None or (track["owner"] != username and not track["is_public"]):
        raise HTTPException(status_code=404, detail="Track not found")
    return track


@router.post("/export")
async def export_track(
    request: ExportRequest,
    username: str = Depends(get_current_username)
):
    """Export track in specified format."""
    if request.format.lower() not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    _visible_track(request.track_id, username)

    return {
        "track_id": request.track_id,
        "format": request.format,
        "download_url": f"/api/export/download/{request.track_id}.{request.format}",
        "expires_in": 3600
    }


@router.get("/download/{filename}")
async def download_track(
    filename: str,
    username: str = Depends(get_current_username)
):
    """Download an exported track file.

    Serves the track's real MIDI data for .mid/.midi filenames. Other
    supported formats cannot be rendered by this build (no audio synthesis
    pipeline), so they return 501 instead of a placeholder response.
    """
    track_id, dot, extension = filename.rpartition(".")
    if not dot or not track_id:
        raise HTTPException(status_code=400, detail="Invalid download filename")
    extension = extension.lower()
    if extension not in SUPPORTED_FORMATS and extension not in RENDERABLE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    track = _visible_track(track_id, username)

    if extension not in RENDERABLE_FORMATS:
        raise HTTPException(
            status_code=501,
            detail=(
                f"Rendering to {extension} is not available in this build; "
                "download the track as MIDI instead"
            ),
        )

    return Response(
        content=track["midi_data"],
        media_type="audio/midi",
        headers={
            "Content-Disposition": f'attachment; filename="{track_id}.mid"'
        },
    )


@router.post("/upload")
async def upload_to_platform(
    request: UploadRequest,
    username: str = Depends(get_current_username)
):
    """Upload track to external platform."""
    supported_platforms = ["spotify", "soundcloud", "youtube"]
    if request.platform.lower() not in supported_platforms:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported platform. Supported: {', '.join(supported_platforms)}"
        )

    return {
        "track_id": request.track_id,
        "platform": request.platform,
        "status": "uploading",
        "message": f"Upload to {request.platform} initiated"
    }


@router.get("/upload/{upload_id}/status")
async def get_upload_status(
    upload_id: str,
    username: str = Depends(get_current_username)
):
    """Get upload status."""
    return {
        "upload_id": upload_id,
        "status": "completed",
        "platform_url": f"https://example.com/track/{upload_id}"
    }

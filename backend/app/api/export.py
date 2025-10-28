"""Export and sharing API endpoints."""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from backend.app.api.auth import oauth2_scheme

router = APIRouter()


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


@router.post("/export")
async def export_track(
    request: ExportRequest,
    _token: str = Depends(oauth2_scheme)
):
    """Export track in specified format."""
    supported_formats = ["mp3", "wav", "flac", "midi"]
    if request.format.lower() not in supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Supported formats: {', '.join(supported_formats)}"
        )

    return {
        "track_id": request.track_id,
        "format": request.format,
        "download_url": f"/api/export/download/{request.track_id}.{request.format}",
        "expires_in": 3600
    }


@router.get("/download/{filename}")
async def download_track(
    filename: str,
    _token: str = Depends(oauth2_scheme)
):
    """Download exported track."""
    # In production, this would serve the actual file
    return {"message": f"Download {filename}", "status": "ready"}


@router.post("/upload")
async def upload_to_platform(
    request: UploadRequest,
    _token: str = Depends(oauth2_scheme)
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
    _token: str = Depends(oauth2_scheme)
):
    """Get upload status."""
    return {
        "upload_id": upload_id,
        "status": "completed",
        "platform_url": f"https://example.com/track/{upload_id}"
    }

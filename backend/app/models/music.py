"""Music-related data models."""
from typing import Optional

from pydantic import BaseModel


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""

    genre: str
    mood: str
    tempo: int
    duration: int
    key: Optional[str] = "C"
    scale: Optional[str] = "major"


class MusicTrack(BaseModel):
    """Model representing a music track."""

    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str

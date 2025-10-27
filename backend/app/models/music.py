"""Music-related data models."""
from typing import Optional
from pydantic import BaseModel, Field


class MusicGenerationRequest(BaseModel):
    """Model for music generation requests."""

    genre: str = Field(..., description="Music genre (pop, jazz, classical, etc.)")
    mood: str = Field(..., description="Mood of the music (happy, sad, energetic, etc.)")
    key: str = Field(..., description="Musical key (C, D, E, etc.)")
    tempo: int = Field(..., ge=60, le=200, description="Tempo in BPM")
    duration: int = Field(..., ge=30, le=180, description="Duration in seconds")
    style: Optional[str] = Field(None, description="Additional style parameters")


class MusicTrack(BaseModel):
    """Model for music track metadata."""

    id: str = Field(..., description="Unique track identifier")
    title: str = Field(..., description="Track title")
    genre: str = Field(..., description="Music genre")
    duration: int = Field(..., description="Duration in seconds")
    file_path: str = Field(..., description="Path to audio file")
    created_at: str = Field(..., description="Creation timestamp")
    user_id: Optional[str] = Field(None, description="Owner user ID")

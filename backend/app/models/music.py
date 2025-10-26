"""Music-related Pydantic models."""
from typing import Optional
from pydantic import BaseModel, Field


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    genre: str = Field(..., description="Music genre (pop, jazz, classical, rock, electronic, ambient)")
    mood: str = Field(..., description="Mood (happy, sad, energetic, calm, suspenseful)")
    key: str = Field(default="C", description="Musical key (C, D, E, F, G, A, B, etc.)")
    tempo: int = Field(default=120, ge=60, le=200, description="Tempo in BPM (60-200)")
    duration: int = Field(default=60, ge=30, le=180, description="Duration in seconds (30-180)")
    mode: Optional[str] = Field(default="major", description="Mode (major, minor)")


class MusicTrack(BaseModel):
    """Model representing a music track."""
    id: str
    title: str
    genre: str
    mood: Optional[str] = None
    key: Optional[str] = None
    tempo: Optional[int] = None
    duration: int
    file_path: str
    user_id: int
    created_at: str
    is_public: bool = False

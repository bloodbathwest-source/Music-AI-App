"""Music-related data models."""
from typing import Optional
from pydantic import BaseModel, Field


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    genre: str = Field(..., description="Music genre (e.g., pop, jazz, classical, rock, electronic)")
    mood: str = Field(..., description="Mood of the music (e.g., happy, sad, energetic, calm)")
    key: str = Field(..., description="Musical key (e.g., C, D, E, F, G, A, B)")
    tempo: int = Field(..., description="Tempo in BPM (beats per minute)", ge=60, le=200)
    duration: int = Field(..., description="Duration in seconds", ge=30, le=180)


class MusicTrack(BaseModel):
    """Model for a music track."""
    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str
    user_id: Optional[str] = None
    is_public: bool = False

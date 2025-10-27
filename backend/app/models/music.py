"""Music-related data models."""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    genre: str = Field(..., description="Music genre (e.g., pop, jazz, classical, rock, electronic)")
    mood: str = Field(..., description="Mood of the music (e.g., happy, sad, energetic, calm)")
    key: Literal["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] = Field(
        ..., description="Musical key (e.g., C, C#, D, E, F, F#, G, A, A#, B)"
    )
    tempo: int = Field(..., description="Tempo in BPM (beats per minute)", ge=60, le=200)
    duration: int = Field(..., description="Duration in seconds", ge=30, le=180)


class MusicTrack(BaseModel):
    """Model for a music track."""
    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str = Field(..., description="ISO 8601 formatted datetime string (e.g., 2025-10-26T17:00:00)")
    user_id: Optional[str] = None
    is_public: bool = False

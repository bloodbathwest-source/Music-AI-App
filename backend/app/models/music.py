"""Music domain models and request/response schemas."""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "genre": "pop",
                "mood": "happy",
                "key": "C",
                "tempo": 120,
                "duration": 60,
                "mode": "major"
            }
        }
    )

    genre: str = Field(..., description="Music genre (pop, jazz, classical, rock, electronic, ambient)")
    mood: str = Field(..., description="Mood/emotion (happy, sad, energetic, calm, suspenseful)")
    key: str = Field(..., description="Musical key (C, D, E, F, G, A, B)")
    tempo: int = Field(..., ge=60, le=200, description="Tempo in BPM (60-200)")
    duration: int = Field(..., ge=30, le=180, description="Duration in seconds (30-180)")
    mode: Optional[str] = Field(default="major", description="Mode (major, minor, dorian)")


class MusicTrack(BaseModel):
    """Music track model."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "track_123",
                "title": "Generated Track",
                "genre": "pop",
                "mood": "happy",
                "key": "C",
                "tempo": 120,
                "duration": 60,
                "file_path": "/tracks/track_123.mid",
                "created_at": "2025-10-28T23:56:46Z",
                "user_id": "user_456",
                "is_public": False
            }
        }
    )

    id: str = Field(..., description="Unique track identifier")
    title: str = Field(..., description="Track title")
    genre: str = Field(..., description="Music genre")
    mood: str = Field(..., description="Mood/emotion")
    key: str = Field(..., description="Musical key")
    tempo: int = Field(..., description="Tempo in BPM")
    duration: int = Field(..., description="Duration in seconds")
    file_path: str = Field(..., description="Path to the generated file")
    created_at: str = Field(..., description="Creation timestamp")
    user_id: Optional[str] = Field(default=None, description="User who created the track")
    is_public: bool = Field(default=False, description="Whether track is publicly shared")

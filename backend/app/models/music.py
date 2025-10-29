"""
Music generation models and request schemas.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict


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

    genre: str
    mood: str
    key: str
    tempo: int
    duration: int
    mode: Optional[str] = "major"


class MusicTrack(BaseModel):
    """Model representing a music track."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "track_123",
                "title": "My Generated Track",
                "genre": "electronic",
                "duration": 60,
                "file_path": "/tracks/track_123.mp3",
                "created_at": "2025-10-26T17:00:00",
                "is_public": False
            }
        }
    )

    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str
    is_public: Optional[bool] = False



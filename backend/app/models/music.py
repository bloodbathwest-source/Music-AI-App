"""Music-related data models."""
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "genre": "pop",
                "mood": "happy",
                "key": "C",
                "tempo": 120,
                "duration": 60
            }
        }
    )
    
    genre: str = Field(..., description="Music genre (e.g., pop, jazz, classical)")
    mood: str = Field(..., description="Mood of the music (e.g., happy, sad, energetic)")
    key: str = Field(..., description="Musical key (e.g., C, D, E)")
    tempo: int = Field(..., ge=40, le=200, description="Tempo in BPM")
    duration: int = Field(..., ge=30, le=180, description="Duration in seconds")


class MusicTrack(BaseModel):
    """Music track model."""
    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str
    user_id: Optional[str] = None
    is_public: bool = False

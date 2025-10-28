"""Music-related data models."""
from typing import Optional
from pydantic import BaseModel, Field


class MusicGenerationRequest(BaseModel):
    """Request model for music generation."""
    
    genre: str = Field(..., description="Music genre (e.g., pop, jazz, classical, rock, electronic, ambient)")
    mood: str = Field(..., description="Mood of the music (e.g., happy, sad, energetic, calm, suspenseful)")
    key: str = Field(..., description="Musical key (C, D, E, F, G, A, B)")
    tempo: int = Field(..., ge=60, le=200, description="Tempo in BPM (60-200)")
    duration: int = Field(..., ge=30, le=180, description="Duration in seconds (30-180)")
    mode: Optional[str] = Field("major", description="Musical mode (major or minor)")


class MusicTrack(BaseModel):
    """Music track model."""
    
    id: str
    title: str
    genre: str
    duration: int
    file_path: str
    created_at: str

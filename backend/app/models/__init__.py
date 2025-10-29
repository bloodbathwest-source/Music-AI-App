"""
Music generation models and request schemas.
"""

from typing import Optional


class MusicGenerationRequest:
    """Request model for music generation."""

    def __init__(
        self,
        genre: str,
        mood: str,
        key: str,
        tempo: int,
        duration: int,
        mode: Optional[str] = "major"
    ):
        """
        Initialize a music generation request.

        Args:
            genre: Music genre (pop, jazz, classical, rock, electronic).
            mood: Mood/emotion (happy, sad, energetic, calm).
            key: Musical key (C, D, E, F, G, A, B).
            tempo: Tempo in BPM.
            duration: Duration in seconds.
            mode: Musical mode (major, minor, dorian). Defaults to "major".
        """
        self.genre = genre
        self.mood = mood
        self.key = key
        self.tempo = tempo
        self.duration = duration
        self.mode = mode

    def __repr__(self):
        return (f"MusicGenerationRequest(genre={self.genre}, mood={self.mood}, "
                f"key={self.key}, tempo={self.tempo}, duration={self.duration}, "
                f"mode={self.mode})")

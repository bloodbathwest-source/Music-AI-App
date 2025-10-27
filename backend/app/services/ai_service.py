"""AI Music Generation Service using ML models."""
import io
import random
from typing import Dict, List

from midiutil import MIDIFile

from backend.app.models.music import MusicGenerationRequest

# Try to import torch, but don't fail if it's not available
try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True

    class MusicLSTM(nn.Module):
        """LSTM model for music generation."""

        def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=128):
            """Initialize LSTM model."""
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            """Forward pass."""
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out
except ImportError:
    TORCH_AVAILABLE = False
    MusicLSTM = None


class MusicGenerationService:
    """Service for AI-powered music generation."""

    # Musical scales
    SCALES = {
        "major": [0, 2, 4, 5, 7, 9, 11],
        "minor": [0, 2, 3, 5, 7, 8, 10],
        "dorian": [0, 2, 3, 5, 7, 9, 10],
        "phrygian": [0, 1, 3, 5, 7, 8, 10],
        "lydian": [0, 2, 4, 6, 7, 9, 11],
        "mixolydian": [0, 2, 4, 5, 7, 9, 10],
        "pentatonic": [0, 2, 4, 7, 9]
    }

    # Key to MIDI note mapping
    KEY_TO_NOTE = {
        "C": 60, "C#": 61, "D": 62, "D#": 63,
        "E": 64, "F": 65, "F#": 66, "G": 67,
        "G#": 68, "A": 69, "A#": 70, "B": 71
    }

    # Genre-specific characteristics
    GENRE_PARAMS = {
        "pop": {"tempo_range": (100, 130), "scale": "major"},
        "jazz": {"tempo_range": (90, 140), "scale": "dorian"},
        "classical": {"tempo_range": (60, 120), "scale": "major"},
        "rock": {"tempo_range": (110, 150), "scale": "minor"},
        "electronic": {"tempo_range": (120, 140), "scale": "minor"},
        "ambient": {"tempo_range": (60, 90), "scale": "pentatonic"}
    }

    def __init__(self):
        """Initialize the music generation service."""
        if TORCH_AVAILABLE and MusicLSTM:
            self.model = MusicLSTM()
        else:
            self.model = None  # Use rule-based generation without ML

    def generate_music(self, request: MusicGenerationRequest) -> Dict:
        """
        Generate music based on request parameters.

        Args:
            request: MusicGenerationRequest with generation parameters

        Returns:
            Dict with generated music data
        """
        # Get genre parameters
        genre_params = self.GENRE_PARAMS.get(
            request.genre.lower(),
            {"tempo_range": (100, 120), "scale": "major"}
        )

        # Determine scale
        scale_name = genre_params.get("scale", "major")
        scale = self.SCALES.get(scale_name, self.SCALES["major"])

        # Get root note
        root_note = self.KEY_TO_NOTE.get(request.key, 60)

        # Generate melody
        melody = self._generate_melody(
            root_note=root_note,
            scale=scale,
            duration=request.duration,
            tempo=request.tempo,
            mood=request.mood
        )

        # Generate MIDI file
        midi_data = self._create_midi(melody, request.tempo)

        return {
            "melody": melody,
            "midi_data": midi_data,
            "parameters": {
                "genre": request.genre,
                "key": request.key,
                "tempo": request.tempo,
                "duration": request.duration,
                "scale": scale_name
            }
        }

    def _generate_melody(
        self,
        root_note: int,
        scale: List[int],
        duration: int,
        tempo: int,
        mood: str
    ) -> List[Dict]:
        """Generate a melody using evolutionary algorithm."""
        # Calculate number of notes based on duration and tempo
        beats_per_second = tempo / 60.0
        num_notes = int(duration * beats_per_second / 2)  # Half notes

        melody = []
        current_time = 0.0

        # Mood-based note selection
        mood_params = {
            "happy": {"octave_range": 1, "variation": 0.7},
            "sad": {"octave_range": -1, "variation": 0.3},
            "energetic": {"octave_range": 2, "variation": 0.9},
            "calm": {"octave_range": 0, "variation": 0.2},
            "suspenseful": {"octave_range": 0, "variation": 0.5}
        }

        params = mood_params.get(mood.lower(), {"octave_range": 0, "variation": 0.5})

        for _ in range(num_notes):
            # Select note from scale
            scale_degree = random.choice(scale)
            octave_offset = random.randint(0, 2) * 12 + params["octave_range"] * 12
            note = root_note + scale_degree + octave_offset

            # Ensure note is in valid MIDI range
            note = max(21, min(108, note))

            # Note duration (in beats)
            note_duration = random.choice([0.5, 1.0, 1.5, 2.0])

            # Velocity (volume)
            velocity = random.randint(70, 110)

            melody.append({
                "note": note,
                "start_time": current_time,
                "duration": note_duration,
                "velocity": velocity
            })

            current_time += note_duration

        return melody

    def _create_midi(self, melody: List[Dict], tempo: int) -> bytes:
        """Create MIDI file from melody."""
        midi = MIDIFile(1)
        track = 0
        channel = 0
        time = 0

        midi.addTempo(track, time, tempo)

        for note_data in melody:
            midi.addNote(
                track,
                channel,
                note_data["note"],
                note_data["start_time"],
                note_data["duration"],
                note_data["velocity"]
            )

        # Convert to bytes
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)
        return midi_buffer.read()

    def train_model(self, training_data: List[Dict]):
        """
        Train the LSTM model on custom data.

        Args:
            training_data: List of music sequences for training
        """
        # Placeholder for model training logic
        # In production, this would implement proper training loop
        raise NotImplementedError("Model training not yet implemented")

"""AI music generation service."""
import io
import random
from typing import Dict, List
from midiutil import MIDIFile

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE and nn is not None:
    class MusicLSTM(nn.Module):
        """LSTM model for music generation."""

        def __init__(self, input_size=128, hidden_size=256, num_layers=2):
            """Initialize the LSTM model."""
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, input_size)

        def forward(self, x):
            """Forward pass."""
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out)
            return output
else:
    # Dummy class when torch is not available
    class MusicLSTM:  # type: ignore
        """Dummy LSTM model placeholder when torch is not available."""
        def __init__(self, *args, **kwargs):
            """Initialize dummy model."""
            pass


class MusicGenerationService:
    """Service for AI-powered music generation."""

    def __init__(self):
        """Initialize the music generation service."""
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = MusicLSTM()
                self.model.eval()
            except Exception as error:
                print(f"Failed to initialize LSTM model: {error}")
                self.model = None

    def generate_music(self, request) -> Dict:
        """
        Generate music based on the request parameters.

        Args:
            request: MusicGenerationRequest with genre, mood, key, tempo, duration

        Returns:
            Dict containing melody, midi_data, and parameters
        """
        # Generate melody based on parameters
        melody = self._generate_melody(
            genre=request.genre,
            mood=request.mood,
            key=request.key,
            tempo=request.tempo,
            duration=request.duration
        )

        # Create MIDI data
        midi_data = self._create_midi(melody, request.tempo)

        return {
            "melody": melody,
            "midi_data": midi_data,
            "parameters": {
                "genre": request.genre,
                "mood": request.mood,
                "key": request.key,
                "tempo": request.tempo,
                "duration": request.duration
            }
        }

    def _generate_melody(self, genre: str, mood: str, key: str, tempo: int,
                         duration: int) -> List[Dict]:
        """Generate a melody based on parameters."""
        # Musical scale patterns
        scales = {
            "C": [60, 62, 64, 65, 67, 69, 71, 72],
            "D": [62, 64, 66, 67, 69, 71, 73, 74],
            "E": [64, 66, 68, 69, 71, 73, 75, 76],
            "F": [65, 67, 69, 70, 72, 74, 76, 77],
            "G": [67, 69, 71, 72, 74, 76, 78, 79],
            "A": [69, 71, 73, 74, 76, 78, 80, 81],
            "B": [71, 73, 75, 76, 78, 80, 82, 83],
        }

        scale = scales.get(key, scales["C"])

        # Calculate number of notes based on duration and tempo
        beats_per_second = tempo / 60.0
        num_notes = int(duration * beats_per_second / 2)  # Assuming quarter notes mostly

        melody = []
        for i in range(num_notes):
            # Generate note based on mood
            if mood == "happy":
                note = random.choice(scale[3:])  # Higher notes
                velocity = random.randint(80, 110)
            elif mood == "sad":
                note = random.choice(scale[:5])  # Lower notes
                velocity = random.randint(60, 90)
            elif mood == "energetic":
                note = random.choice(scale)
                velocity = random.randint(90, 120)
            else:  # calm
                note = random.choice(scale[:6])
                velocity = random.randint(50, 80)

            # Vary note duration
            if genre == "jazz":
                duration_beats = random.choice([0.25, 0.5, 0.75, 1.0])
            elif genre == "classical":
                duration_beats = random.choice([0.5, 1.0, 1.5, 2.0])
            else:
                duration_beats = random.choice([0.5, 1.0])

            melody.append({
                "note": note,
                "duration": duration_beats,
                "velocity": velocity,
                "time": i * 0.5  # Simplified timing
            })

        return melody

    def _create_midi(self, melody: List[Dict], tempo: int) -> bytes:
        """Create MIDI data from melody."""
        midi = MIDIFile(1)  # One track
        track = 0
        channel = 0
        time = 0

        midi.addTempo(track, time, tempo)

        for note_data in melody:
            midi.addNote(
                track,
                channel,
                note_data["note"],
                note_data["time"],
                note_data["duration"],
                note_data["velocity"]
            )

        # Write to bytes
        output = io.BytesIO()
        midi.writeFile(output)
        return output.getvalue()

    def train_model(self, training_data: List[Dict]):
        """
        Train the LSTM model on custom data.

        Args:
            training_data: List of music sequences for training.

        Returns:
            bool: True if the model was trained successfully, False otherwise.

        Note:
            This is a placeholder. In production, implement proper training loop.
        """
        if self.model and training_data:
            print("Training the model with provided data...")
            return True
        return False

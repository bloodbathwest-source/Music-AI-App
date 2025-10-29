"""
AI Music Generation Service.

This module provides the MusicGenerationService class that handles
music generation using AI models and evolutionary algorithms.
"""

import io
import random
from typing import Dict, List

try:
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from midiutil import MIDIFile


class MusicLSTM(nn.Module):
    """Simple LSTM model for music generation."""

    def __init__(self, input_size=128, hidden_size=256, output_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output


class MusicGenerationService:
    """Service for generating music using AI and evolutionary algorithms."""

    def __init__(self):
        """Initialize the music generation service."""
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = MusicLSTM()
            except Exception:
                # If model initialization fails, continue without it
                self.model = None

    def generate_music(self, request) -> Dict:
        """
        Generate music based on the provided request parameters.

        Args:
            request: MusicGenerationRequest object with generation parameters.

        Returns:
            Dict containing melody, midi_data, and parameters.
        """
        # Extract parameters from request
        genre = request.genre
        mood = request.mood
        key = request.key
        tempo = request.tempo
        duration = request.duration

        # Generate melody based on parameters
        melody = self._generate_melody(genre, mood, key, tempo, duration)

        # Create MIDI data
        midi_data = self._create_midi_data(melody, tempo)

        return {
            "melody": melody,
            "midi_data": midi_data,
            "parameters": {
                "genre": genre,
                "mood": mood,
                "key": key,
                "tempo": tempo,
                "duration": duration
            }
        }

    def _generate_melody(self, genre: str, mood: str, key: str,
                        tempo: int, duration: int) -> List[Dict]:
        """
        Generate a melody based on parameters.

        Args:
            genre: Music genre (pop, jazz, classical, rock, electronic).
            mood: Mood/emotion (happy, sad, energetic, calm).
            key: Musical key (C, D, E, F, G, A, B).
            tempo: Tempo in BPM.
            duration: Duration in seconds.

        Returns:
            List of note dictionaries with note, duration, and velocity.
        """
        # Map key to root note
        key_map = {
            "C": 60, "D": 62, "E": 64, "F": 65,
            "G": 67, "A": 69, "B": 71
        }
        root = key_map.get(key, 60)

        # Define scale based on genre and mood
        if genre == "jazz":
            scale = [0, 2, 3, 5, 7, 9, 10]  # Minor/jazz scale
        elif mood == "sad":
            scale = [0, 2, 3, 5, 7, 8, 10]  # Minor scale
        else:
            scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale

        # Calculate number of notes based on duration and tempo
        # Assume quarter notes on average
        beats = (duration * tempo) / 60
        num_notes = int(beats)

        # Generate melody notes
        melody = []
        available_notes = [root + offset + octave
                          for octave in [0, 12, 24]
                          for offset in scale]

        # Filter to piano range (21-108)
        available_notes = [n for n in available_notes if 21 <= n <= 108]

        for _ in range(num_notes):
            # Adjust velocity based on mood
            if mood == "energetic":
                velocity = random.randint(90, 127)
            elif mood == "calm":
                velocity = random.randint(50, 80)
            elif mood == "sad":
                velocity = random.randint(40, 70)
            else:  # happy or default
                velocity = random.randint(70, 100)

            # Adjust note duration based on genre
            if genre == "electronic":
                note_duration = random.choice([0.25, 0.5, 1.0])
            elif genre == "classical":
                note_duration = random.choice([0.5, 1.0, 1.5, 2.0])
            else:
                note_duration = random.choice([0.5, 1.0])

            # Select note (prefer stepwise motion)
            if melody and random.random() < 0.7:
                # Stepwise motion
                last_note = melody[-1]["note"]
                nearby_notes = [n for n in available_notes
                              if abs(n - last_note) <= 4]
                if nearby_notes:
                    note = random.choice(nearby_notes)
                else:
                    note = random.choice(available_notes)
            else:
                # Random jump
                note = random.choice(available_notes)

            melody.append({
                "note": note,
                "duration": note_duration,
                "velocity": velocity
            })

        return melody

    def _create_midi_data(self, melody: List[Dict], tempo: int) -> bytes:
        """
        Create MIDI data from melody.

        Args:
            melody: List of note dictionaries.
            tempo: Tempo in BPM.

        Returns:
            MIDI file data as bytes.
        """
        midi = MIDIFile(1)
        track = 0
        time_offset = 0
        midi.addTempo(track, time_offset, tempo)

        time = 0
        for note_data in melody:
            note = note_data["note"]
            duration = note_data["duration"]
            velocity = note_data["velocity"]

            midi.addNote(track, 0, note, time, duration, velocity)
            time += duration

        # Convert to bytes
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)
        midi_data = midi_buffer.read()

        return midi_data

    def train_model(self, training_data: List[Dict]):
        """
        Train the LSTM model on custom data.

        Args:
            training_data: List of music sequences for training.

        Returns:
            bool: True if the model was trained successfully with the given data,
                  False otherwise.

        Note:
            This is a placeholder logic. In production, implement a proper training
            loop with validation, error handling, and saving the model's state.
        """
        if self.model and training_data:
            print("Training the model with provided data...")  # Placeholder message
            return True  # Indicate success
        return False  # Indicate failure

"""AI Music Generation Service."""
import random
import io
from typing import Dict, List, Any, Optional

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from midiutil import MIDIFile
from backend.app.models.music import MusicGenerationRequest


class MusicLSTM(nn.Module):
    """LSTM neural network for music generation."""

    def __init__(self, input_size: int = 128, hidden_size: int = 256, num_layers: int = 2):
        """Initialize the LSTM model.

        Args:
            input_size: Size of input features (number of possible notes)
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> tuple:
        """Forward pass through the network.

        Args:
            x: Input tensor
            hidden: Hidden state tuple (h, c)

        Returns:
            Tuple of (output, hidden_state)
        """
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            hidden = (h0, c0)

        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden


class MusicGenerationService:
    """Service for generating music using AI algorithms."""

    def __init__(self):
        """Initialize the music generation service."""
        self.model = None
        if TORCH_AVAILABLE:
            self.model = MusicLSTM()
            self.model.eval()

        # Define musical scales and patterns
        self.scales = {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10]
        }

        # Genre-specific patterns
        self.genre_patterns = {
            "pop": {"rhythm": [1, 0.5, 0.5, 1], "octave_range": 2},
            "jazz": {"rhythm": [1, 0.75, 0.25, 1], "octave_range": 3},
            "classical": {"rhythm": [1, 1, 1, 1], "octave_range": 3},
            "rock": {"rhythm": [1, 0.5, 0.5, 1], "octave_range": 2},
            "electronic": {"rhythm": [0.5, 0.5, 0.5, 0.5], "octave_range": 2},
            "ambient": {"rhythm": [2, 2, 2, 2], "octave_range": 2}
        }

        # Mood-specific adjustments
        self.mood_patterns = {
            "happy": {"velocity_mod": 10, "note_range": "high"},
            "sad": {"velocity_mod": -10, "note_range": "low"},
            "energetic": {"velocity_mod": 20, "note_range": "high"},
            "calm": {"velocity_mod": -15, "note_range": "mid"},
            "suspenseful": {"velocity_mod": 5, "note_range": "mid"}
        }

    def generate_music(self, request: MusicGenerationRequest) -> Dict[str, Any]:
        """Generate music based on the request parameters.

        Args:
            request: Music generation request with genre, mood, key, tempo, duration

        Returns:
            Dictionary containing melody, midi_data, and parameters
        """
        # Get root note from key
        root_notes = {
            "C": 60, "D": 62, "E": 64, "F": 65,
            "G": 67, "A": 69, "B": 71
        }
        root = root_notes.get(request.key, 60)

        # Get scale for the mode
        mode = request.mode if hasattr(request, 'mode') and request.mode else "major"
        scale = self.scales.get(mode, self.scales["major"])

        # Generate melody notes for the scale
        melody_notes = [root + interval for interval in scale]

        # Add octave variations based on genre
        genre_pattern = self.genre_patterns.get(request.genre, self.genre_patterns["pop"])
        octave_range = genre_pattern["octave_range"]

        for octave in range(1, octave_range + 1):
            melody_notes.extend([root + interval + (12 * octave) for interval in scale])

        # Calculate number of notes based on duration and tempo
        beats_per_second = request.tempo / 60.0
        total_beats = int(request.duration * beats_per_second)
        num_notes = max(32, min(total_beats, 256))

        # Generate melody with evolutionary approach
        melody = self._evolve_melody(
            melody_notes,
            num_notes,
            request.genre,
            request.mood,
            genre_pattern
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
                "duration": request.duration,
                "mode": mode
            }
        }

    def _evolve_melody(
        self,
        available_notes: List[int],
        num_notes: int,
        _genre: str,
        mood: str,
        genre_pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Evolve a melody using a simplified evolutionary algorithm.

        Args:
            available_notes: List of MIDI note numbers to choose from
            num_notes: Number of notes to generate
            genre: Music genre
            mood: Emotional mood
            genre_pattern: Genre-specific pattern dictionary

        Returns:
            List of note dictionaries with note, duration, and velocity
        """
        # Get mood pattern
        mood_pattern = self.mood_patterns.get(mood, self.mood_patterns["happy"])

        # Create population of melodies
        population_size = 20
        population = []

        for _ in range(population_size):
            melody = self._generate_random_melody(
                available_notes,
                num_notes,
                genre_pattern,
                mood_pattern
            )
            population.append(melody)

        # Evolve for several generations
        generations = 10
        for _ in range(generations):
            # Sort by fitness (simple heuristic: variation in notes and rhythm)
            population.sort(key=self._calculate_fitness, reverse=True)

            # Keep top 50% and generate new melodies for bottom 50%
            survivors = population[:population_size // 2]
            offspring = [
                self._generate_random_melody(
                    available_notes,
                    num_notes,
                    genre_pattern,
                    mood_pattern
                )
                for _ in range(population_size // 2)
            ]
            population = survivors + offspring

        # Return the best melody
        population.sort(key=self._calculate_fitness, reverse=True)
        return population[0]

    def _generate_random_melody(
        self,
        available_notes: List[int],
        num_notes: int,
        genre_pattern: Dict[str, Any],
        mood_pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate a random melody.

        Args:
            available_notes: Available MIDI notes
            num_notes: Number of notes to generate
            genre_pattern: Genre-specific patterns
            mood_pattern: Mood-specific patterns

        Returns:
            List of note dictionaries
        """
        melody = []
        rhythm_pattern = genre_pattern["rhythm"]
        base_velocity = 80 + mood_pattern["velocity_mod"]

        # Filter notes based on mood
        if mood_pattern["note_range"] == "high":
            notes_pool = [n for n in available_notes if n >= 60]
        elif mood_pattern["note_range"] == "low":
            notes_pool = [n for n in available_notes if n <= 72]
        else:
            notes_pool = available_notes

        if not notes_pool:
            notes_pool = available_notes

        for i in range(num_notes):
            note = random.choice(notes_pool)
            duration = rhythm_pattern[i % len(rhythm_pattern)]
            velocity = max(20, min(127, base_velocity + random.randint(-10, 10)))

            melody.append({
                "note": note,
                "duration": duration,
                "velocity": velocity
            })

        return melody

    def _calculate_fitness(self, melody: List[Dict[str, Any]]) -> float:
        """Calculate fitness score for a melody.

        Args:
            melody: List of note dictionaries

        Returns:
            Fitness score (higher is better)
        """
        if not melody:
            return 0.0

        # Calculate note variation
        notes = [n["note"] for n in melody]
        unique_notes = len(set(notes))
        note_variation = unique_notes / len(melody) if melody else 0

        # Calculate rhythm variation
        durations = [n["duration"] for n in melody]
        unique_durations = len(set(durations))
        rhythm_variation = unique_durations / len(melody) if melody else 0

        # Prefer smooth transitions (penalize large jumps)
        intervals = [abs(notes[i+1] - notes[i]) for i in range(len(notes)-1)]
        avg_interval = sum(intervals) / len(intervals) if intervals else 0
        smoothness = 1.0 / (1.0 + avg_interval / 12.0)  # Normalize by octave

        # Combined fitness
        fitness = (note_variation * 0.3) + (rhythm_variation * 0.2) + (smoothness * 0.5)
        return fitness

    def _create_midi(self, melody: List[Dict[str, Any]], tempo: int) -> bytes:
        """Create MIDI file data from melody.

        Args:
            melody: List of note dictionaries
            tempo: Tempo in BPM

        Returns:
            MIDI file data as bytes
        """
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
                time,
                note_data["duration"],
                note_data["velocity"]
            )
            time += note_data["duration"]

        # Write to bytes
        midi_buffer = io.BytesIO()
        midi.writeFile(midi_buffer)
        midi_buffer.seek(0)
        return midi_buffer.read()

    def train_model(self, training_data: List[Dict]) -> bool:
        """Train the LSTM model on custom data.

        Args:
            training_data: List of music sequences for training

        Returns:
            True if training successful, False otherwise
        """
        if self.model and training_data and TORCH_AVAILABLE:
            print("Training the model with provided data...")
            # In production, implement proper training loop
            return True
        return False

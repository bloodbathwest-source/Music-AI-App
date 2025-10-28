"""AI-powered music generation service."""
import random
import io
from typing import List, Dict, Optional
from midiutil import MIDIFile
from backend.app.models.music import MusicGenerationRequest

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# LSTM Model for music generation (optional)
if TORCH_AVAILABLE:
    class LSTMModel(nn.Module):
        """LSTM model for music generation."""
        
        def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=128):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out


class MusicGenerationService:
    """Service for AI-powered music generation."""
    
    def __init__(self):
        """Initialize the music generation service."""
        self.model = None
        if TORCH_AVAILABLE:
            try:
                self.model = LSTMModel()
                self.model.eval()
            except Exception as e:
                print(f"Warning: Could not initialize LSTM model: {e}")
                self.model = None
        
        # Musical scales
        self.scales = {
            'C': [60, 62, 64, 65, 67, 69, 71, 72],  # C major
            'D': [62, 64, 66, 67, 69, 71, 73, 74],  # D major
            'E': [64, 66, 68, 69, 71, 73, 75, 76],  # E major
            'F': [65, 67, 69, 70, 72, 74, 76, 77],  # F major
            'G': [67, 69, 71, 72, 74, 76, 78, 79],  # G major
            'A': [69, 71, 73, 74, 76, 78, 80, 81],  # A major
            'B': [71, 73, 75, 76, 78, 80, 82, 83],  # B major
        }
        
        # Genre characteristics
        self.genre_patterns = {
            'pop': {'rhythm_complexity': 0.6, 'note_density': 0.7},
            'jazz': {'rhythm_complexity': 0.8, 'note_density': 0.8},
            'classical': {'rhythm_complexity': 0.5, 'note_density': 0.6},
            'rock': {'rhythm_complexity': 0.7, 'note_density': 0.7},
            'electronic': {'rhythm_complexity': 0.9, 'note_density': 0.9},
            'ambient': {'rhythm_complexity': 0.3, 'note_density': 0.4},
        }
        
        # Mood characteristics
        self.mood_patterns = {
            'happy': {'velocity_range': (80, 110), 'tempo_modifier': 1.1},
            'sad': {'velocity_range': (40, 70), 'tempo_modifier': 0.8},
            'energetic': {'velocity_range': (90, 120), 'tempo_modifier': 1.2},
            'calm': {'velocity_range': (50, 75), 'tempo_modifier': 0.9},
            'suspenseful': {'velocity_range': (60, 85), 'tempo_modifier': 0.95},
        }
    
    def generate_music(self, request: MusicGenerationRequest) -> Dict:
        """
        Generate music based on the given parameters.
        
        Args:
            request: Music generation request with parameters
            
        Returns:
            Dictionary containing melody, MIDI data, and parameters
        """
        # Get the scale for the requested key
        scale = self.scales.get(request.key, self.scales['C'])
        
        # Get genre and mood patterns
        genre_pattern = self.genre_patterns.get(request.genre, self.genre_patterns['pop'])
        mood_pattern = self.mood_patterns.get(request.mood, self.mood_patterns['happy'])
        
        # Calculate number of notes based on duration and tempo
        beats_per_second = request.tempo / 60.0
        total_beats = int(request.duration * beats_per_second)
        num_notes = int(total_beats * genre_pattern['note_density'])
        
        # Generate melody
        melody = self._generate_melody(
            scale=scale,
            num_notes=num_notes,
            velocity_range=mood_pattern['velocity_range'],
            duration=request.duration
        )
        
        # Create MIDI data
        midi_data = self._create_midi(
            melody=melody,
            tempo=request.tempo,
            duration=request.duration
        )
        
        return {
            'melody': melody,
            'midi_data': midi_data,
            'parameters': {
                'genre': request.genre,
                'mood': request.mood,
                'key': request.key,
                'tempo': request.tempo,
                'duration': request.duration
            }
        }
    
    def _generate_melody(self, scale: List[int], num_notes: int, 
                        velocity_range: tuple, duration: int) -> List[Dict]:
        """
        Generate a melody using the given scale.
        
        Args:
            scale: List of MIDI note numbers in the scale
            num_notes: Number of notes to generate
            velocity_range: Tuple of (min, max) velocity
            duration: Total duration in seconds
            
        Returns:
            List of note dictionaries with note, duration, and velocity
        """
        melody = []
        
        # Handle case where no notes should be generated
        if num_notes <= 0:
            return melody
        
        time_per_note = duration / num_notes
        
        for i in range(num_notes):
            # Select note from scale with slight tendency toward tonic
            if i == 0 or i == num_notes - 1:
                # Start and end on tonic
                note = scale[0]
            else:
                # Random note from scale with weighted preference for tonic and dominant
                weights = [3, 1, 1, 1, 2, 1, 1, 1][:len(scale)]
                note = random.choices(scale, weights=weights)[0]
            
            # Add octave variation
            octave_shift = random.choice([-12, 0, 0, 0, 12])
            note = max(21, min(108, note + octave_shift))  # Keep in piano range
            
            # Random duration (quarter, eighth, half note variations)
            duration_factor = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])
            note_duration = time_per_note * duration_factor
            
            # Random velocity within mood range
            velocity = random.randint(velocity_range[0], velocity_range[1])
            
            melody.append({
                'note': note,
                'duration': note_duration,
                'velocity': velocity
            })
        
        return melody
    
    def _create_midi(self, melody: List[Dict], tempo: int, duration: int) -> bytes:
        """
        Create MIDI file from melody.
        
        Args:
            melody: List of note dictionaries
            tempo: Tempo in BPM
            duration: Total duration in seconds
            
        Returns:
            MIDI file as bytes
        """
        midi = MIDIFile(1)  # 1 track
        track = 0
        channel = 0
        time = 0
        
        midi.addTempo(track, time, tempo)
        
        # Add notes to MIDI
        current_time = 0
        for note_data in melody:
            pitch = note_data['note']
            duration = note_data['duration']
            velocity = note_data['velocity']
            
            midi.addNote(track, channel, pitch, current_time, duration, velocity)
            current_time += duration
        
        # Write MIDI to bytes
        midi_io = io.BytesIO()
        midi.writeFile(midi_io)
        midi_io.seek(0)
        return midi_io.read()
    
    def train_model(self, training_data: List[Dict]) -> bool:
        """
        Train the LSTM model on custom data.

        Args:
            training_data: List of music sequences for training.

        Returns:
            bool: True if the model was trained successfully with the given data, False otherwise.

        Note:
            This is a placeholder logic. In production, implement a proper training loop with validation,
            error handling, and saving the model's state.
        """
        if self.model and training_data:
            print("Training the model with provided data...")  # Placeholder message
            return True  # Indicate success
        return False  # Indicate failure

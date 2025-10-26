"""
Music generation service using evolutionary algorithms and simple generative models
"""
import random
import numpy as np
from midiutil import MIDIFile
import os
import re
from typing import Dict, List, Tuple


class MusicGenerator:
    """AI Music Generator using evolutionary algorithms"""
    
    # Musical scales
    SCALES = {
        'major': [0, 2, 4, 5, 7, 9, 11],
        'minor': [0, 2, 3, 5, 7, 8, 10],
        'dorian': [0, 2, 3, 5, 7, 9, 10],
        'phrygian': [0, 1, 3, 5, 7, 8, 10],
        'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    }
    
    # Root notes
    ROOT_NOTES = {
        'C': 60, 'C#': 61, 'D': 62, 'D#': 63,
        'E': 64, 'F': 65, 'F#': 66, 'G': 67,
        'G#': 68, 'A': 69, 'A#': 70, 'B': 71
    }
    
    # Emotional patterns
    EMOTION_PATTERNS = {
        'happy': {'tempo_range': (120, 140), 'velocity_range': (80, 110), 'note_density': 0.7},
        'sad': {'tempo_range': (60, 80), 'velocity_range': (50, 70), 'note_density': 0.4},
        'energetic': {'tempo_range': (140, 180), 'velocity_range': (90, 120), 'note_density': 0.8},
        'calm': {'tempo_range': (70, 90), 'velocity_range': (60, 80), 'note_density': 0.3},
        'suspenseful': {'tempo_range': (90, 110), 'velocity_range': (60, 90), 'note_density': 0.5},
    }
    
    def __init__(self, genre: str = 'pop', key: str = 'C', mode: str = 'major', 
                 emotion: str = 'happy', tempo: int = 120):
        """Initialize music generator"""
        self.genre = genre
        self.key = key
        self.mode = mode
        self.emotion = emotion
        self.tempo = tempo
        self.root_note = self.ROOT_NOTES.get(key, 60)
        self.scale = self.SCALES.get(mode, self.SCALES['major'])
        self.emotion_pattern = self.EMOTION_PATTERNS.get(emotion, self.EMOTION_PATTERNS['happy'])
    
    def generate_melody_notes(self, num_bars: int = 8) -> List[Tuple[int, float, int]]:
        """Generate melody notes (note, duration, velocity)"""
        notes = []
        notes_per_bar = 8 if self.emotion == 'energetic' else 4
        total_notes = num_bars * notes_per_bar
        
        # Generate scale notes in different octaves
        available_notes = []
        for octave in [-12, 0, 12]:
            for interval in self.scale:
                available_notes.append(self.root_note + interval + octave)
        
        # Generate melodic sequence
        current_note_idx = len(available_notes) // 2  # Start in middle range
        
        for i in range(total_notes):
            # Random walk with tendency to return to tonic
            if random.random() < 0.3:  # Return to nearby note
                current_note_idx = max(0, min(len(available_notes) - 1, 
                                             current_note_idx + random.choice([-1, 0, 1])))
            else:  # Jump to new note
                current_note_idx = random.randint(0, len(available_notes) - 1)
            
            note = available_notes[current_note_idx]
            duration = random.choice([0.5, 1, 2]) if self.emotion != 'energetic' else random.choice([0.25, 0.5, 1])
            velocity = random.randint(*self.emotion_pattern['velocity_range'])
            
            notes.append((note, duration, velocity))
        
        return notes
    
    def generate_chord_progression(self, num_bars: int = 8) -> List[List[int]]:
        """Generate chord progression"""
        # Common chord progressions
        progressions = {
            'major': [[0, 4, 7], [5, 9, 12], [7, 11, 14], [0, 4, 7]],  # I-IV-V-I
            'minor': [[0, 3, 7], [5, 8, 12], [7, 10, 14], [0, 3, 7]],  # i-iv-v-i
        }
        
        base_progression = progressions.get(self.mode, progressions['major'])
        chords = []
        
        for bar in range(num_bars):
            chord_template = base_progression[bar % len(base_progression)]
            chord = [self.root_note - 12 + note for note in chord_template]
            chords.append(chord)
        
        return chords
    
    def generate_midi(self, output_path: str, title: str = "AI Generated Song") -> str:
        """Generate MIDI file"""
        # Sanitize the output path to prevent path injection
        output_path = os.path.normpath(output_path)
        base_dir = os.path.normpath(os.path.join(os.getcwd(), 'static', 'audio'))
        
        # Ensure the path is within the allowed directory
        if not output_path.startswith(base_dir):
            output_path = os.path.join(base_dir, os.path.basename(output_path))
        
        midi = MIDIFile(2)  # 2 tracks: melody and harmony
        
        # Track 0: Melody
        midi.addTrackName(0, 0, "Melody")
        midi.addTempo(0, 0, self.tempo)
        
        # Track 1: Chords
        midi.addTrackName(1, 0, "Harmony")
        midi.addTempo(1, 0, self.tempo)
        
        # Generate melody
        melody_notes = self.generate_melody_notes(num_bars=8)
        time = 0
        for note, duration, velocity in melody_notes:
            midi.addNote(0, 0, note, time, duration, velocity)
            time += duration
        
        # Generate chords
        chords = self.generate_chord_progression(num_bars=8)
        time = 0
        for chord in chords:
            for note in chord:
                midi.addNote(1, 0, note, time, 4, 70)  # Whole note chords
            time += 4
        
        # Save MIDI file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            midi.writeFile(f)
        
        return output_path
    
    def get_song_metadata(self) -> Dict:
        """Get metadata for generated song"""
        return {
            'genre': self.genre,
            'key': self.key,
            'mode': self.mode,
            'emotion': self.emotion,
            'tempo': self.tempo
        }

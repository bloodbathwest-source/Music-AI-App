"""Utilities for music file generation and processing."""
import os
from datetime import datetime
from typing import Dict, List, Tuple

from midiutil import MIDIFile
import numpy as np


class MusicFileGenerator:
    """Generate MIDI and audio files from music data."""

    def __init__(self, output_dir: str = "generated_music"):
        """Initialize the file generator.

        Args:
            output_dir: Directory to save generated files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_midi_file(
        self,
        music_data: Dict,
        filename: str = None
    ) -> str:
        """Create a MIDI file from music data.

        Args:
            music_data: Dictionary containing melody, harmony, tempo, etc.
            filename: Output filename (auto-generated if None)

        Returns:
            Path to the created MIDI file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"music_{timestamp}.mid"

        filepath = os.path.join(self.output_dir, filename)

        # Create MIDI file with 2 tracks (melody and harmony)
        num_tracks = 2 if music_data.get('harmony') else 1
        midi = MIDIFile(num_tracks)

        # Set tempo
        tempo = music_data.get('tempo', 120)
        midi.addTempo(0, 0, tempo)

        # Add track name
        metadata = music_data.get('metadata', {})
        track_name = f"{metadata.get('genre', 'Unknown')} - {metadata.get('mood', 'Unknown')}"
        midi.addTrackName(0, 0, track_name)

        # Add melody
        time = 0
        for note, duration, velocity in music_data['melody']:
            midi.addNote(
                track=0,
                channel=0,
                pitch=int(note),
                time=time,
                duration=duration,
                volume=int(velocity)
            )
            time += duration

        # Add harmony if present
        if music_data.get('harmony'):
            midi.addTrackName(1, 0, "Harmony")
            time = 0
            for note, duration, velocity in music_data['harmony']:
                midi.addNote(
                    track=1,
                    channel=1,
                    pitch=int(note),
                    time=time,
                    duration=duration,
                    volume=int(velocity)
                )
                time += duration

        # Write MIDI file
        with open(filepath, 'wb') as output_file:
            midi.writeFile(output_file)

        return filepath

    def generate_waveform_data(
        self,
        music_data: Dict,
        sample_rate: int = 22050
    ) -> Tuple[np.ndarray, int]:
        """Generate waveform data for visualization.

        Args:
            music_data: Dictionary containing melody data
            sample_rate: Audio sample rate

        Returns:
            Tuple of (waveform array, sample rate)
        """
        melody = music_data['melody']
        tempo = music_data.get('tempo', 120)

        # Calculate total duration
        total_duration = sum(duration for _, duration, _ in melody)

        # Convert to samples
        total_samples = int(total_duration * sample_rate * 60 / tempo)
        waveform = np.zeros(total_samples)

        # Generate simple sine wave representation
        sample_idx = 0
        for note, duration, velocity in melody:
            # Convert note to frequency (A4 = 440 Hz is MIDI note 69)
            frequency = 440 * (2 ** ((note - 69) / 12))

            # Duration in samples
            note_samples = int(duration * sample_rate * 60 / tempo)

            # Generate sine wave
            t = np.linspace(0, duration * 60 / tempo, note_samples, endpoint=False)
            amplitude = velocity / 127.0  # Normalize velocity
            wave = amplitude * np.sin(2 * np.pi * frequency * t)

            # Add envelope (simple ADSR)
            envelope = self._create_envelope(note_samples)
            wave *= envelope

            # Add to waveform
            end_idx = min(sample_idx + note_samples, total_samples)
            waveform[sample_idx:end_idx] = wave[:end_idx - sample_idx]
            sample_idx = end_idx

            if sample_idx >= total_samples:
                break

        return waveform, sample_rate

    def _create_envelope(self, num_samples: int) -> np.ndarray:
        """Create a simple ADSR envelope.

        Args:
            num_samples: Number of samples in the envelope

        Returns:
            Envelope array
        """
        attack = int(num_samples * 0.1)
        decay = int(num_samples * 0.2)
        sustain_level = 0.7
        release = int(num_samples * 0.3)

        envelope = np.ones(num_samples)

        # Attack
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)

        # Decay
        if decay > 0:
            envelope[attack:attack+decay] = np.linspace(1, sustain_level, decay)

        # Sustain
        sustain_start = attack + decay
        sustain_end = num_samples - release
        if sustain_end > sustain_start:
            envelope[sustain_start:sustain_end] = sustain_level

        # Release
        if release > 0 and sustain_end < num_samples:
            envelope[sustain_end:] = np.linspace(sustain_level, 0, num_samples - sustain_end)

        return envelope

    def get_music_notation_data(self, music_data: Dict) -> List[Dict]:
        """Convert music data to notation-friendly format.

        Args:
            music_data: Dictionary containing melody data

        Returns:
            List of note dictionaries with notation information
        """
        melody = music_data['melody']
        notation_data = []

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        for note, duration, velocity in melody:
            # Convert MIDI note to note name and octave
            note_name = note_names[note % 12]
            octave = (note // 12) - 1

            notation_data.append({
                'note': f"{note_name}{octave}",
                'duration': duration,
                'velocity': velocity,
                'midi_note': note
            })

        return notation_data
